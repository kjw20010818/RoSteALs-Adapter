"""
small_ae_adapter.py
===================
RoSteALS Post-G SmallAE Adapter (standalone module).
VQGAN 디코더(G) 출력에 연결되는 경량 Autoencoder 워터마크 어댑터.

구조:
  image_wm [3,256,256]
    → Encoder (3→32→64ch, 256→128)  ← 공간 bottleneck
    → FiLM (secret 주입)
    → Decoder (64→32ch, 128→256)
    → delta (zero-init)
  image_out = clamp(image_wm + exp(log_alpha) * delta)
"""
"""
Post-G Adapter for RoSteALS
============================
G(decoder) 출력 이미지에 경량 어댑터를 붙여 강건성 개선.

파이프라인:
  x → E → z → z + F(s) → VQ → G → image_wm → Adapter(image_wm, s) → image_out → D → bits
       (frozen)  (frozen)          (frozen)    (학습 대상)                              (frozen)

FreqJNDAdapter (Idea 1 + 2 합성):
  Step 1. FiLM CNN  : image_wm와 s를 받아 spatial residual 생성
  Step 2. FFT 밴드패스 : residual을 중간 주파수 대역으로 필터링
            → JPEG·blur 공격은 고주파를 제거하므로, 중주파에 신호 배치
  Step 3. JND 마스크 : 국소 분산(텍스처 복잡도)으로 시각적 허용 영역 가중
            → 텍스처 풍부한 영역에 더 강한 residual 허용
  최종   : image_out = image_wm + alpha * jnd * bandpass(residual)

Usage:
  python scripts/train_post_g_adapter.py \\
      --config models/VQ4_post_g_adapter.yaml \\
      --pretrain_ckpt /hdd/watermark/model/adapter_ablation/continued_FD_only_100k/checkpoints/last_bn_calibrated.ckpt \\
      --output /hdd/watermark/model/post_g_adapter \\
      --gpus 2 --batch_size 4
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import pytorch_lightning as pl
from torch.nn import functional as thf

from ldm.util import instantiate_from_config
from ldm.models.autoencoder import VQModelInterface
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from cldm.ae import disabled_train
from cldm.compat import safe_torch_load


# ── FiLM Block ────────────────────────────────────────────────────────────────



# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Core SmallAE building blocks
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class _AEResBlock(nn.Module):
    """Lightweight residual block for SmallAEAdapter (no cond)."""
    def __init__(self, ch_in: int, ch_out: int = None):
        super().__init__()
        ch_out = ch_out or ch_in
        n_groups = min(8, ch_in)
        self.norm1  = nn.GroupNorm(n_groups, ch_in)
        self.conv1  = nn.Conv2d(ch_in, ch_out, 3, 1, 1)
        n_groups2   = min(8, ch_out)
        self.norm2  = nn.GroupNorm(n_groups2, ch_out)
        self.conv2  = nn.Conv2d(ch_out, ch_out, 3, 1, 1)
        self.skip   = nn.Conv2d(ch_in, ch_out, 1) if ch_in != ch_out else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class _AEFiLM(nn.Module):
    """Zero-init FiLM layer (scale/shift)."""
    def __init__(self, cond_dim: int, ch: int):
        super().__init__()
        self.linear = nn.Linear(cond_dim, ch * 2)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        out = self.linear(cond)[:, :, None, None]
        scale, shift = out.chunk(2, dim=1)
        return (1.0 + scale) * x + shift


class SmallAEAdapter(nn.Module):
    """
    Small Autoencoder Adapter (Post-G).

    Post-G CNN 어댑터와의 결정적 차이:
      CNN adapter : 픽셀 국소 수용장 (local receptive field)
      SmallAE     : 공간 압축 후 전역 복원 → 이미지 전체 맥락에서 워터마크 배치

    파이프라인 (n_down=1, base_ch=32):
        image_wm [3, 256, 256]
          → conv_in(3→32) + ResBlock×2           → [32, 256, 256]
          → stride-2 conv(32→64) + ResBlock×2    → [64, 128, 128]  ← bottleneck
          → FiLM(s)
          → upsample + conv(64→32) + ResBlock×2  → [32, 256, 256]
          → FiLM(s) + conv_out(32→3) zero-init   → delta [3, 256, 256]
        image_out = clamp(image_wm + exp(log_alpha) * delta)

    파라미터 수 (base_ch=32, n_down=1, n_res_blocks=2): ~362K
    """

    def __init__(
        self,
        img_channels: int     = 3,
        base_ch: int          = 32,
        n_down: int           = 1,          # 다운샘플 횟수 (1→H/2, 2→H/4)
        n_res_blocks: int     = 2,
        secret_len: int       = 100,
        cond_dim: int         = 128,
        log_alpha_init: float = -2.0,
        max_residual: float   = 0.12,
        smooth_delta: bool    = True,       # delta에 Gaussian blur 적용 (블록 아티팩트 제거)
        smooth_sigma: float   = 1.0,        # Gaussian sigma (픽셀 단위)
    ):
        super().__init__()
        self.n_down = n_down
        self.smooth_delta = smooth_delta

        # Gaussian kernel 미리 생성 (고정 가중치, 학습 안 함)
        if smooth_delta:
            ksize = 5
            coords = torch.arange(ksize, dtype=torch.float32) - ksize // 2
            g = torch.exp(-coords**2 / (2 * smooth_sigma**2))
            g = g / g.sum()
            kernel = (g[:, None] * g[None, :])             # (ksize, ksize)
            kernel = kernel[None, None].repeat(img_channels, 1, 1, 1)  # (C,1,k,k)
            self.register_buffer('_gauss_kernel', kernel)
            self._gauss_pad = ksize // 2

        # Secret → conditioning
        self.secret_enc = nn.Sequential(
            nn.Linear(secret_len, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        # ── Encoder ────────────────────────────────────────────────────────
        # Stage 0: conv_in (no spatial change)
        self.enc_in = nn.Conv2d(img_channels, base_ch, 3, 1, 1)
        self.enc_in_res = nn.Sequential(*[_AEResBlock(base_ch) for _ in range(n_res_blocks)])

        # Downsample stages
        self.enc_down = nn.ModuleList()
        ch = base_ch
        for i in range(n_down):
            ch_out = base_ch * (2 ** (i + 1))
            blk = nn.Sequential(
                nn.Conv2d(ch, ch_out, 3, stride=2, padding=1),
                *[_AEResBlock(ch_out) for _ in range(n_res_blocks)],
            )
            self.enc_down.append(blk)
            ch = ch_out
        # ch = bottleneck channels = base_ch * 2^n_down

        self.film_bn = _AEFiLM(cond_dim, ch)

        # ── Decoder ────────────────────────────────────────────────────────
        self.dec_up = nn.ModuleList()
        for i in reversed(range(n_down)):
            ch_out = base_ch * (2 ** i)
            blk = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(ch, ch_out, 3, 1, 1),
                *[_AEResBlock(ch_out) for _ in range(n_res_blocks)],
            )
            self.dec_up.append(blk)
            ch = ch_out
        # ch = base_ch after all upsamples

        self.film_dec = _AEFiLM(cond_dim, ch)

        # ── Output (zero-init → identity start) ────────────────────────────
        self.conv_out = nn.Conv2d(ch, img_channels, 3, 1, 1)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

        self.log_alpha    = nn.Parameter(torch.tensor(float(log_alpha_init)))
        self.max_residual = max_residual

    def forward(self, image_wm: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        cond = self.secret_enc(s)

        # Encode
        h = F.silu(self.enc_in(image_wm))
        h = self.enc_in_res(h)
        for blk in self.enc_down:
            h = blk(h)

        # Bottleneck FiLM
        h = self.film_bn(h, cond)

        # Decode
        for blk in self.dec_up:
            h = blk(h)

        # Decoder FiLM + output
        h     = self.film_dec(h, cond)
        delta = self.conv_out(h)

        # Gaussian smoothing → 128×128 stride 경계 블록 아티팩트 제거
        if self.smooth_delta:
            p = self._gauss_pad
            delta = F.conv2d(
                F.pad(delta, [p, p, p, p], mode='reflect'),
                self._gauss_kernel,
                groups=delta.shape[1],
            )

        alpha = torch.exp(self.log_alpha)
        delta = (alpha * delta).clamp(-self.max_residual, self.max_residual)
        return (image_wm + delta).clamp(-1.0, 1.0)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ── NAFNetAdapter ─────────────────────────────────────────────────────────────



# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Lightning training wrapper
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class ControlAEPostG(pl.LightningModule):
    """
    RoSteALS + Post-G FreqJNDAdapter.

    E, G, F, D 모두 freeze.  FreqJNDAdapter만 학습.
    """

    def __init__(
        self,
        first_stage_key: str,
        first_stage_config: dict,
        control_key: str,
        control_config: dict,
        decoder_config: dict,
        loss_config: dict,
        noise_config: str = '__none__',
        scale_factor: float = 1.0,
        # 어댑터 종류: 'freq_jnd' (기본) | 'jpeg_dct' | 'small_ae'
        adapter_type: str = 'freq_jnd',
        # FreqJNDAdapter / JpegDCTAdapter 공통 설정
        adapter_hidden: int = 64,
        adapter_n_blocks: int = 4,
        adapter_cond_dim: int = 128,
        adapter_jnd_ksize: int = 11,
        adapter_jnd_sharpness: float = 20.0,
        # FreqJNDAdapter 전용
        adapter_freq_low: float = 0.02,
        adapter_freq_high: float = 0.20,
        # JpegDCTAdapter 전용
        adapter_jpeg_quality: int = 75,
        adapter_max_residual: float = 0.12,   # 최대 잔차 크기 (YAML에서 덮어씀)
        # SmallAEAdapter / UNetAdapter 전용
        adapter_n_down: int = 1,         # 다운샘플 횟수 (1: H/2 bottleneck) / ViT: 0=no_bn, >=1=bn
        adapter_smooth_delta: bool = False,   # delta Gaussian 스무딩 (블록 아티팩트 제거)
        adapter_smooth_sigma: float = 1.0,    # Gaussian sigma
        # LightViTAdapter 전용
        adapter_vit_patch_size: int = 16,
        adapter_vit_n_heads: int = 4,
        # Robustness-first loss weights
        lambda_preserve: float = 1.0,
        lambda_img: float = 0.1,
        lambda_lpips: float = 0.0,       # LPIPS perceptual loss weight (0=비활성)
        attack_prob: float = 0.9,
        # Texture masking
        use_texture_mask: bool = False,  # Laplacian 기반 텍스처 마스크로 delta 조절
        texture_mask_power: float = 0.5, # 마스크 강도 (0=균등, 1=완전 마스킹)
        # D joint fine-tune
        finetune_decoder: bool = False,
        # Checkpoint
        pretrain_ckpt: str = '__none__',
    ):
        super().__init__()

        self.scale_factor    = scale_factor
        # ViT 전용 파라미터 임시 저장 (adapter 초기화 전에 필요)
        self._vit_patch_size = adapter_vit_patch_size
        self._vit_n_heads    = adapter_vit_n_heads
        self.first_stage_key = first_stage_key
        self.control_key     = control_key
        self.lambda_preserve    = lambda_preserve
        self.lambda_img         = lambda_img
        self.lambda_lpips       = lambda_lpips
        self.attack_prob        = attack_prob
        self.use_texture_mask   = use_texture_mask
        self.texture_mask_power = texture_mask_power
        self.finetune_decoder   = finetune_decoder
        self.adapter_type    = adapter_type

        # ── Sub-models ─────────────────────────────────────────────────
        self.ae      = instantiate_from_config(first_stage_config)
        self.control = instantiate_from_config(control_config)
        self.decoder = instantiate_from_config(decoder_config)
        self.control.copy_encoder_weight(self.ae)

        # ── Freeze E, G, F (항상) ──────────────────────────────────────
        for m in (self.ae, self.control):
            m.eval()
            m.train = disabled_train
            for p in m.parameters():
                p.requires_grad = False

        # ── D: finetune_decoder=True 이면 학습, 아니면 freeze ──────────
        if finetune_decoder:
            self.decoder.train()
        else:
            self.decoder.eval()
            self.decoder.train = disabled_train
            for p in self.decoder.parameters():
                p.requires_grad = False

        # ── Adapter 선택 ───────────────────────────────────────────────
        secret_len = control_config['params']['secret_len']
        if adapter_type == 'jpeg_dct':
            self.adapter = JpegDCTAdapter(
                img_channels  = 3,
                hidden        = adapter_hidden,
                n_blocks      = adapter_n_blocks,
                secret_len    = secret_len,
                cond_dim      = adapter_cond_dim,
                jpeg_quality  = adapter_jpeg_quality,
                jnd_ksize     = adapter_jnd_ksize,
                jnd_sharpness = adapter_jnd_sharpness,
                max_residual  = adapter_max_residual,
            )
        elif adapter_type == 'small_ae':
            self.adapter = SmallAEAdapter(
                img_channels  = 3,
                base_ch       = adapter_hidden,
                n_down        = adapter_n_down,
                n_res_blocks  = adapter_n_blocks,
                secret_len    = secret_len,
                cond_dim      = adapter_cond_dim,
                smooth_delta  = adapter_smooth_delta,
                smooth_sigma  = adapter_smooth_sigma,
            )
        elif adapter_type == 'unet':
            from cldm.arch_adapters import UNetAdapter
            self.adapter = UNetAdapter(
                img_channels = 3,
                base_ch      = adapter_hidden,
                n_down       = adapter_n_down,
                n_res        = adapter_n_blocks,
                secret_len   = secret_len,
                cond_dim     = adapter_cond_dim,
            )
        elif adapter_type == 'light_vit':
            from cldm.arch_adapters import LightViTAdapter
            self.adapter = LightViTAdapter(
                img_channels    = 3,
                patch_size      = getattr(self, '_vit_patch_size', 16),
                dim             = adapter_hidden,
                n_heads         = getattr(self, '_vit_n_heads', 4),
                n_layers        = adapter_n_blocks,
                with_bottleneck = bool(adapter_n_down),
                secret_len      = secret_len,
                cond_dim        = adapter_cond_dim,
            )
        elif adapter_type == 'nafnet':
            self.adapter = NAFNetAdapter(
                img_channels  = 3,
                width         = adapter_hidden,       # 채널 수 (32 권장)
                n_blocks      = adapter_n_blocks,     # NAFBlock 수 (8 권장)
                n_down        = adapter_n_down,       # 0=flat, 1=U-Net
                secret_len    = secret_len,
                cond_dim      = adapter_cond_dim,
                max_residual  = adapter_max_residual,
            )
        else:  # 'freq_jnd' (기본)
            self.adapter = FreqJNDAdapter(
            img_channels  = 3,
            hidden        = adapter_hidden,
            n_blocks      = adapter_n_blocks,
            secret_len    = secret_len,
            cond_dim      = adapter_cond_dim,
            freq_low      = adapter_freq_low,
            freq_high     = adapter_freq_high,
            jnd_ksize     = adapter_jnd_ksize,
            jnd_sharpness = adapter_jnd_sharpness,
            )

        # ── Loss ───────────────────────────────────────────────────────
        self.loss_layer = instantiate_from_config(loss_config)
        # logvar: 어댑터 학습에서 업데이트하지 않음 → freeze
        if hasattr(self.loss_layer, 'logvar'):
            self.loss_layer.logvar.requires_grad_(False)
        # LPIPS: 항상 eval 고정
        if hasattr(self.loss_layer, 'perceptual_loss'):
            self.loss_layer.perceptual_loss.eval()
            for p in self.loss_layer.perceptual_loss.parameters():
                p.requires_grad_(False)

        # ── Noise augmentation ─────────────────────────────────────────
        if noise_config != '__none__':
            self.noise = instantiate_from_config(noise_config)

        # ── Fixed-input warmup ─────────────────────────────────────────
        self.secret_len  = secret_len
        self.fixed_x     = None
        self.fixed_img   = None
        self.fixed_control = None
        self.register_buffer("fixed_input", torch.tensor(True))

        # ── Load pretrained weights ─────────────────────────────────────
        if pretrain_ckpt != '__none__':
            self._load_pretrain(pretrain_ckpt)

        self._print_info()

    # ── train() override: 항상 frozen 모듈은 eval 유지 ────────────────

    def train(self, mode: bool = True):
        """PL이 model.train()을 호출해도 frozen/LPIPS 모듈은 eval 고정."""
        super().train(mode)
        # ae, control은 disabled_train으로 보호됨
        # finetune_decoder=False이면 decoder도 disabled_train으로 보호됨
        # LPIPS는 항상 eval 강제
        if hasattr(self, 'loss_layer') and hasattr(self.loss_layer, 'perceptual_loss'):
            self.loss_layer.perceptual_loss.eval()
        return self

    # ── Helpers ────────────────────────────────────────────────────────

    def _load_pretrain(self, path: str):
        sd = safe_torch_load(path, map_location='cpu')
        if 'state_dict' in sd:
            sd = sd['state_dict']
        skip     = {'fixed_input', 'fixed_x', 'fixed_img', 'fixed_control'}
        # adapter 가중치는 항상 다시 초기화 (새 학습); noise/loss는 config에서
        skip_pfx = ('noise.', 'loss_layer.', 'adapter.')
        model_sd = self.state_dict()
        filtered = {
            k: v for k, v in sd.items()
            if k not in skip
            and not any(k.startswith(p) for p in skip_pfx)
            and k in model_sd
            and model_sd[k].shape == v.shape
        }
        self.load_state_dict(filtered, strict=False)
        print(f"[PostG] Loaded pretrain: {path}  (matched {len(filtered)} keys)")
        self.fixed_input.fill_(0)
        # Robustness-first: step 0부터 attack 적용
        if hasattr(self, 'noise') and hasattr(self.noise, 'step0'):
            self.noise.step0.fill_(1)
            print("[PostG] noise activated from step 0 (robustness-first)")
        # finetune_decoder: D 파라미터가 로드됐으면 requires_grad 재확인
        if self.finetune_decoder:
            for p in self.decoder.parameters():
                p.requires_grad = True
            print("[PostG] decoder unfrozen for joint fine-tune")

    def _print_info(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        dec_train = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
        print("=" * 60)
        print(f"  ControlAEPostG  [{self.adapter_type.upper()}]")
        print("=" * 60)
        print(f"  adapter type      : {self.adapter_type}")
        adap_p = getattr(self.adapter, 'param_count', lambda: sum(p.numel() for p in self.adapter.parameters()))()
        print(f"  adapter params    : {adap_p:,}")
        print(f"  decoder trainable : {dec_train:,}  ({'fine-tune' if self.finetune_decoder else 'frozen'})")
        print(f"  trainable params  : {trainable:,}")
        print(f"  total params      : {total:,}")
        if hasattr(self.adapter, 'freq_low'):
            print(f"  freq band         : [{self.adapter.freq_low}, {self.adapter.freq_high}]")
        if hasattr(self.adapter, 'jpeg_quality'):
            print(f"  jpeg_quality      : {self.adapter.jpeg_quality}")
        if hasattr(self.adapter, 'jnd_ksize'):
            print(f"  jnd ksize         : {self.adapter.jnd_ksize}")
        if hasattr(self.adapter, 'n_down'):
            print(f"  ae n_down         : {self.adapter.n_down}")
        print(f"  lambda_preserve   : {self.lambda_preserve}")
        print(f"  attack_prob       : {self.attack_prob}")
        print("=" * 60)

    # ── Encode / Decode ────────────────────────────────────────────────

    @torch.no_grad()
    def encode_first_stage(self, image: torch.Tensor) -> torch.Tensor:
        posterior = self.ae.encode(image)
        if isinstance(posterior, DiagonalGaussianDistribution):
            z = posterior.sample()
        elif isinstance(posterior, torch.Tensor):
            z = posterior
        else:
            raise NotImplementedError(type(posterior))
        return self.scale_factor * z

    @torch.no_grad()
    def decode_first_stage(self, z: torch.Tensor) -> torch.Tensor:
        return self.ae.decode(z / self.scale_factor)

    # ── Input ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def get_input(self, batch):
        image  = batch[self.first_stage_key]
        secret = batch[self.control_key]
        image  = einops.rearrange(image, "b h w c -> b c h w").contiguous()
        x      = self.encode_first_stage(image)

        if self.fixed_input:
            if self.fixed_x is None:
                print('[PostG] Warmup: fixed input image.')
                self.fixed_x       = x.detach().clone()
                self.fixed_img     = image.detach().clone()
                self.fixed_control = secret.detach().clone()
            return self.fixed_x, self.fixed_img, self.fixed_control

        return x, image, secret

    # ── Forward ────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor, img: torch.Tensor, s: torch.Tensor):
        """Returns (image_out, image_wm, posterior)."""
        with torch.no_grad():
            if self.control.__class__.__name__ == 'SecretEncoder6':
                eps, posterior = self.control(x, s)
            else:
                eps, posterior = self.control(img, s)
            z_wm      = x + eps
            image_wm  = self.decode_first_stage(z_wm).clamp(-1., 1.)

        image_out = self.adapter(image_wm, s)
        return image_out, image_wm, posterior

    # ── Training step ──────────────────────────────────────────────────

    def shared_step(self, batch):
        x, img, s = self.get_input(batch)
        image_out, image_wm, posterior = self(x, img, s)

        # 크기 정렬
        if img.shape[-1] > 256:
            img       = thf.interpolate(img,       (256, 256), mode='bilinear', align_corners=False).detach()
            image_out = thf.interpolate(image_out, (256, 256), mode='bilinear', align_corners=False)
            image_wm  = thf.interpolate(image_wm,  (256, 256), mode='bilinear', align_corners=False)

        image_out = image_out.clamp(-1., 1.)

        # Attack augmentation
        if hasattr(self, 'noise') and self.noise.is_activated():
            image_noised = self.noise(image_out, self.global_step, p=self.attack_prob)
        else:
            image_noised = image_out

        # D: attacked 이미지에서 bit 예측
        # finetune_decoder=True이면 D도 그라디언트 흐름
        if self.finetune_decoder:
            pred = self.decoder(image_noised)
        else:
            pred = self.decoder(image_noised)

        # Loss (ImageSecretLoss: recon + secret + perceptual)
        loss, loss_dict = self.loss_layer(img, image_out, posterior, s, pred, self.global_step)

        # Texture masking: 텍스처가 풍부한 영역에만 delta 집중
        if self.use_texture_mask:
            with torch.no_grad():
                # Laplacian으로 텍스처 맵 계산 (edge/texture 강도)
                gray = image_wm.detach().mean(dim=1, keepdim=True)  # (B,1,H,W)
                lap_k = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]],
                                     dtype=gray.dtype, device=gray.device).view(1,1,3,3)
                lap = thf.conv2d(thf.pad(gray,[1,1,1,1],'reflect'), lap_k).abs()
                lap = lap.expand(-1,3,-1,-1)
                # 정규화: 0~1, power로 강도 조절
                lap_max = lap.flatten(2).max(dim=2)[0][...,None,None].clamp(min=1e-6)
                texture_mask = (lap / lap_max).pow(1.0 - self.texture_mask_power)
                # delta에 마스크 적용 (smooth area → delta 억제)
                delta_orig = image_out - image_wm
                image_out  = (image_wm + delta_orig * texture_mask).clamp(-1., 1.)

        # L_preserve: image_out ≈ image_wm (D 분포 유지)
        if self.lambda_preserve > 0:
            l_preserve = thf.l1_loss(image_out, image_wm.detach())
            loss       = loss + self.lambda_preserve * l_preserve
            loss_dict['l_preserve'] = l_preserve.detach()

        # LPIPS perceptual loss: 시각적 품질 강제
        if self.lambda_lpips > 0:
            if hasattr(self.loss_layer, 'perceptual_loss'):
                lpips_fn = self.loss_layer.perceptual_loss
            else:
                lpips_fn = None
            if lpips_fn is not None:
                # [-1,1] → LPIPS 입력 형식
                l_lpips = lpips_fn(image_out, img.detach()).mean()
                loss    = loss + self.lambda_lpips * l_lpips
                loss_dict['l_lpips'] = l_lpips.detach()

        # 모니터링: clean bit_acc + PSNR
        with torch.no_grad():
            pred_clean    = self.decoder(image_out)
            bit_acc_clean = ((pred_clean > 0).float() == s).float().mean()
            loss_dict['bit_acc_clean'] = bit_acc_clean

            a   = (img.clamp(-1, 1) + 1) / 2
            b   = (image_out.clamp(-1, 1) + 1) / 2
            mse = thf.mse_loss(a, b).clamp(min=1e-10)
            loss_dict['psnr'] = -10 * torch.log10(mse)

            # adapter residual magnitude (JND 마스크 적용 전 대비 후)
            loss_dict['residual_l2'] = (image_out - image_wm).pow(2).mean().sqrt()

        # Fixed-input warmup 종료 조건
        if self.fixed_input and loss_dict['bit_acc'].item() > 0.9:
            print(f'[PostG] bit_acc={loss_dict["bit_acc"]:.3f}, switching to full dataset.')
            self.fixed_input.fill_(0)

        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch)
        self.log_dict({f'train/{k}': v for k, v in loss_dict.items()},
                      prog_bar=True, logger=True, on_step=True, on_epoch=True,
                      sync_dist=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch)
        self.log_dict({f'val/{k}': v for k, v in loss_dict.items()},
                      prog_bar=False, logger=True, on_step=False, on_epoch=True,
                      sync_dist=True)

    def configure_optimizers(self):
        lr = self.learning_rate
        # param_groups 방식: adapter(기본 lr) + D(lr×0.1)
        param_groups = [
            {'params': list(self.adapter.parameters()), 'lr': lr, 'weight_decay': 1e-4},
        ]
        if self.finetune_decoder:
            param_groups.append(
                {'params': list(self.decoder.parameters()), 'lr': lr * 0.1, 'weight_decay': 1e-4}
            )
        opt = torch.optim.AdamW(param_groups)
        n_adapter = sum(p.numel() for p in self.adapter.parameters())
        n_dec     = sum(p.numel() for p in self.decoder.parameters()) if self.finetune_decoder else 0
        print(f'[PostG] AdamW: adapter lr={lr:.2e} ({n_adapter:,}p)'
              + (f', decoder lr={lr*0.1:.2e} ({n_dec:,}p)' if self.finetune_decoder else ''))
        return opt

    # ── Logging ────────────────────────────────────────────────────────

    @torch.no_grad()
    def log_images(self, batch, **kwargs):
        x, img, s = self.get_input(batch)
        image_out, image_wm, _ = self(x, img, s)
        log = {
            'input':   img,
            'wm_base': image_wm,
            'output':  image_out,
            'diff_5x': (image_out - image_wm).abs().clamp(0, 1) * 5,
        }
        if hasattr(self, 'noise') and self.noise.is_activated():
            log['noised'] = self.noise(image_out, self.global_step, p=1.0)
        return log
