"""
추가 아키텍처 어댑터
===================
SmallAE bottleneck 효과가 아키텍처에 무관한지 검증하기 위해
U-Net과 경량 ViT 두 가지 완전히 다른 아키텍처를 구현.

UNetAdapter  : skip connection이 있는 U-Net 구조
LightViTAdapter: 패치 기반 Transformer 구조
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── 공통 primitives ────────────────────────────────────────────────────────────

class _ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1),
            nn.GroupNorm(min(8, ch), ch),
            nn.SiLU(),
            nn.Conv2d(ch, ch, 3, 1, 1),
            nn.GroupNorm(min(8, ch), ch),
        )
        self.silu = nn.SiLU()

    def forward(self, x):
        return self.silu(x + self.net(x))


class _FiLM(nn.Module):
    """Feature-wise Linear Modulation (zero-init → identity start)"""
    def __init__(self, cond_dim: int, ch: int):
        super().__init__()
        self.proj = nn.Linear(cond_dim, ch * 2)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, cond):
        # x: [B, ch, H, W] 또는 [B, N, ch]
        params = self.proj(cond)          # [B, ch*2]
        scale, shift = params.chunk(2, dim=-1)
        if x.dim() == 4:
            scale = scale[:, :, None, None]
            shift = shift[:, :, None, None]
        return x * (1 + scale) + shift


# ── UNet Adapter ───────────────────────────────────────────────────────────────

class UNetAdapter(nn.Module):
    """
    U-Net 스타일 어댑터.

    SmallAE와의 결정적 차이:
      SmallAE : encoder 출력을 그대로 decoder로 전달
      UNetAdapter: encoder의 중간 feature를 skip connection으로 decoder에 전달
                   → 고해상도 세부 정보 + 저해상도 맥락 동시 활용

    파이프라인 (n_down=1, base_ch=32):
        image_wm [3, 256, 256]
          → enc_in(3→32) + ResBlock×n_res          → skip [32, 256, 256]
          → stride-2 conv(32→64) + ResBlock×n_res  → [64, 128, 128]
          → FiLM(s)
          → upsample(64→32) + cat(skip) → merge(64→32) + ResBlock×n_res → [32, 256, 256]
          → FiLM(s) + conv_out(32→3) zero-init → delta [3, 256, 256]
        image_out = clamp(image_wm + exp(log_alpha) * delta)

    n_down=0: skip connection만 있는 동일 해상도 residual CNN
    """

    def __init__(
        self,
        img_channels: int   = 3,
        base_ch: int        = 32,
        n_down: int         = 1,
        n_res: int          = 2,
        secret_len: int     = 100,
        cond_dim: int       = 128,
        log_alpha_init: float = -2.0,
        max_residual: float = 0.12,
    ):
        super().__init__()
        self.n_down = n_down
        self.max_residual = max_residual

        self.secret_enc = nn.Sequential(
            nn.Linear(secret_len, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        # ── Encoder ────────────────────────────────────────────────────────
        self.enc_in  = nn.Conv2d(img_channels, base_ch, 3, 1, 1)
        self.enc_res = nn.Sequential(*[_ResBlock(base_ch) for _ in range(n_res)])

        if n_down > 0:
            self.enc_down     = nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1)
            self.enc_down_res = nn.Sequential(*[_ResBlock(base_ch * 2) for _ in range(n_res)])
            bn_ch = base_ch * 2
        else:
            bn_ch = base_ch

        self.film_bn = _FiLM(cond_dim, bn_ch)

        # ── Decoder ────────────────────────────────────────────────────────
        if n_down > 0:
            self.dec_up_conv = nn.Conv2d(bn_ch, base_ch, 3, 1, 1)   # after bilinear upsample
            self.dec_merge   = nn.Conv2d(base_ch * 2, base_ch, 1)    # merge skip (concat→conv)
            self.dec_res     = nn.Sequential(*[_ResBlock(base_ch) for _ in range(n_res)])

        self.film_out = _FiLM(cond_dim, base_ch)
        self.conv_out = nn.Conv2d(base_ch, img_channels, 3, 1, 1)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

        self.log_alpha = nn.Parameter(torch.tensor(float(log_alpha_init)))

    def forward(self, image_wm: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        cond   = self.secret_enc(s)

        h      = F.silu(self.enc_in(image_wm))
        h_skip = self.enc_res(h)        # skip: [B, base_ch, H, W]

        if self.n_down > 0:
            h = F.silu(self.enc_down(h_skip))
            h = self.enc_down_res(h)    # [B, base_ch*2, H/2, W/2]
        else:
            h = h_skip

        h = self.film_bn(h, cond)

        if self.n_down > 0:
            h = F.interpolate(h, scale_factor=2, mode='bilinear', align_corners=False)
            h = F.silu(self.dec_up_conv(h))                     # [B, base_ch, H, W]
            h = torch.cat([h, h_skip], dim=1)                    # skip connection ← 핵심 차이
            h = self.dec_merge(h)
            h = self.dec_res(h)

        h     = self.film_out(h, cond)
        delta = self.conv_out(h)
        alpha = torch.exp(self.log_alpha)
        delta = (alpha * delta).clamp(-self.max_residual, self.max_residual)
        return (image_wm + delta).clamp(-1., 1.)


# ── Lightweight ViT Adapter ────────────────────────────────────────────────────

class _TransformerBlock(nn.Module):
    """경량 Transformer 블록 (Multi-Head Self-Attention + FFN)"""

    def __init__(self, dim: int, n_heads: int, mlp_ratio: float = 2.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim    = int(dim * mlp_ratio)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, h, h)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class LightViTAdapter(nn.Module):
    """
    패치 기반 경량 ViT 어댑터.

    SmallAE/UNet과의 결정적 차이:
      Conv 기반: 국소 수용장 (local receptive field)
      ViT 기반 : Multi-Head Self-Attention으로 전역 컨텍스트 캡처

    파이프라인 (patch_size=16, dim=64, with_bottleneck=True):
        image_wm [3, 256, 256]
          → patch_embed(stride=16) → [dim, 16, 16] (= 256 tokens)
          → compress(stride=2)     → [dim, 8, 8]   (= 64 tokens) ← bottleneck
          → FiLM(s)
          → TransformerBlock × n_layers (global attention on 64 tokens)
          → expand(upsample×2)     → [dim, 16, 16]
          → patch_unembed(stride=16)→ delta [3, 256, 256]

    with_bottleneck=False:
        patch_embed → [dim, 16, 16] → FiLM → Transformer (256 tokens) → patch_unembed
        (공간 압축 없이 전체 해상도에서 처리)
    """

    def __init__(
        self,
        img_channels: int     = 3,
        patch_size: int       = 16,
        dim: int              = 64,
        n_heads: int          = 4,
        n_layers: int         = 3,
        with_bottleneck: bool = True,
        secret_len: int       = 100,
        cond_dim: int         = 128,
        log_alpha_init: float = -2.0,
        max_residual: float   = 0.12,
    ):
        super().__init__()
        self.patch_size      = patch_size
        self.with_bottleneck = with_bottleneck
        self.dim             = dim
        self.max_residual    = max_residual

        self.secret_enc = nn.Sequential(
            nn.Linear(secret_len, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        # Patch embedding: (3, 256, 256) → (dim, 16, 16) with patch_size=16
        self.patch_embed = nn.Conv2d(img_channels, dim, patch_size, stride=patch_size)

        if with_bottleneck:
            # (dim, 16, 16) → (dim, 8, 8)  ← spatial compression
            self.compress = nn.Conv2d(dim, dim, 2, stride=2)
            # (dim, 8, 8)  → (dim, 16, 16)
            self.expand = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(dim, dim, 3, 1, 1),
            )

        self.film = _FiLM(cond_dim, dim)

        # Transformer: 64 tokens (bottleneck) or 256 tokens (no bottleneck)
        self.transformer = nn.Sequential(
            *[_TransformerBlock(dim, n_heads) for _ in range(n_layers)]
        )

        # Patch reconstruction: (dim, 16, 16) → (3, 256, 256)
        self.patch_unembed = nn.ConvTranspose2d(dim, img_channels, patch_size, stride=patch_size)
        nn.init.zeros_(self.patch_unembed.weight)
        nn.init.zeros_(self.patch_unembed.bias)

        self.log_alpha = nn.Parameter(torch.tensor(float(log_alpha_init)))

    def forward(self, image_wm: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        cond = self.secret_enc(s)

        # Patch embedding
        h = self.patch_embed(image_wm)          # [B, dim, H/p, W/p]

        if self.with_bottleneck:
            h = self.compress(h)                # [B, dim, H/(2p), W/(2p)]

        # FiLM conditioning
        h = self.film(h, cond)

        # Transformer (sequence view)
        B, D, Hh, Wh = h.shape
        h_seq = h.flatten(2).transpose(1, 2)    # [B, N, dim]
        for blk in self.transformer:
            h_seq = blk(h_seq)
        h = h_seq.transpose(1, 2).reshape(B, D, Hh, Wh)

        if self.with_bottleneck:
            h = self.expand(h)                  # [B, dim, H/p, W/p]

        # Patch reconstruction → delta
        delta = self.patch_unembed(h)           # [B, 3, H, W]
        alpha = torch.exp(self.log_alpha)
        delta = (alpha * delta).clamp(-self.max_residual, self.max_residual)
        return (image_wm + delta).clamp(-1., 1.)
