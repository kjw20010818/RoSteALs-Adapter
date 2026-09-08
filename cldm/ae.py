from cldm.compat import safe_torch_load
import numpy as np
import einops
import torch
import torch as th
import torch.nn as nn
from torch.nn import functional as thf
import pytorch_lightning as pl
import torchvision
from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)
from contextlib import contextmanager, nullcontext
from einops import rearrange, repeat
from torchvision.utils import make_grid
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import log_txt_as_img, exists, instantiate_from_config, default
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.modules.ema import LitEma
from ldm.modules.distributions.distributions import normal_kl, DiagonalGaussianDistribution
from ldm.modules.diffusionmodules.model import Encoder
import lpips
from kornia import color
from cldm.adapters import build_adapter, count_params, LoRADecoderAdapter, is_gated_adapter

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

class View(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class SecretEncoder3(nn.Module):
    def __init__(self, secret_len, base_res=16, resolution=64) -> None:
        super().__init__()
        log_resolution = int(np.log2(resolution))
        log_base = int(np.log2(base_res))
        self.secret_len = secret_len
        self.secret_scaler = nn.Sequential(
            nn.Linear(secret_len, base_res*base_res*3),
            nn.SiLU(),
            View(-1, 3, base_res, base_res),
            nn.Upsample(scale_factor=(2**(log_resolution-log_base), 2**(log_resolution-log_base))),  # chx16x16 -> chx256x256
            zero_module(conv_nd(2, 3, 3, 3, padding=1))
        )  # secret len -> ch x res x res
    
    def copy_encoder_weight(self, ae_model):
        # misses, ignores = self.load_state_dict(ae_state_dict, strict=False)
        return None

    def encode(self, x):
        x = self.secret_scaler(x)
        return x
    
    def forward(self, x, c):
        # x: [B, C, H, W], c: [B, secret_len]
        c = self.encode(c)
        return c, None


class SecretEncoder4(nn.Module):
    """same as SecretEncoder3 but with ch as input"""
    def __init__(self, secret_len, ch=3, base_res=16, resolution=64) -> None:
        super().__init__()
        log_resolution = int(np.log2(resolution))
        log_base = int(np.log2(base_res))
        self.secret_len = secret_len
        self.secret_scaler = nn.Sequential(
            nn.Linear(secret_len, base_res*base_res*ch),
            nn.SiLU(),
            View(-1, ch, base_res, base_res),
            nn.Upsample(scale_factor=(2**(log_resolution-log_base), 2**(log_resolution-log_base))),  # chx16x16 -> chx256x256
            zero_module(conv_nd(2, ch, ch, 3, padding=1))
        )  # secret len -> ch x res x res
    
    def copy_encoder_weight(self, ae_model):
        # misses, ignores = self.load_state_dict(ae_state_dict, strict=False)
        return None

    def encode(self, x):
        x = self.secret_scaler(x)
        return x
    
    def forward(self, x, c):
        # x: [B, C, H, W], c: [B, secret_len]
        c = self.encode(c)
        return c, None
    
class SecretEncoder6(nn.Module):
    """join img emb with secret emb"""
    def __init__(self, secret_len, ch=3, base_res=16, resolution=64, emode='c3') -> None:
        super().__init__()
        assert emode in ['c3', 'c2', 'm3']
        
        if emode == 'c3':  # c3: concat c and x each has ch channels
            secret_ch = ch 
            join_ch = 2*ch
        elif emode == 'c2':  # c2: concat c (2) and x ave (1)
            secret_ch = 2
            join_ch = ch
        elif emode == 'm3':  # m3: multiply c (ch) and x (ch)
            secret_ch = ch
            join_ch = ch       
        
        # m3: multiply c (ch) and x ave (1)
        log_resolution = int(np.log2(resolution))
        log_base = int(np.log2(base_res))
        self.secret_len = secret_len
        self.emode = emode
        self.resolution = resolution
        self.secret_scaler = nn.Sequential(
            nn.Linear(secret_len, base_res*base_res*secret_ch),
            nn.SiLU(),
            View(-1, secret_ch, base_res, base_res),
            nn.Upsample(scale_factor=(2**(log_resolution-log_base), 2**(log_resolution-log_base))),  # chx16x16 -> chx256x256
        )  # secret len -> ch x res x res
        self.join_encoder = nn.Sequential(
            conv_nd(2, join_ch, join_ch, 3, padding=1),
            nn.SiLU(),
            conv_nd(2, join_ch, ch, 3, padding=1),
            nn.SiLU(),
            conv_nd(2, ch, ch, 3, padding=1),
            nn.SiLU()
        )
        self.out_layer = zero_module(conv_nd(2, ch, ch, 3, padding=1))
    
    def copy_encoder_weight(self, ae_model):
        # misses, ignores = self.load_state_dict(ae_state_dict, strict=False)
        return None

    def encode(self, x):
        x = self.secret_scaler(x)
        return x
    
    def forward(self, x, c):
        # x: [B, C, H, W], c: [B, secret_len]
        c = self.encode(c)
        if self.emode == 'c3':
            x = torch.cat([x, c], dim=1)
        elif self.emode == 'c2':
            x = torch.cat([x.mean(dim=1, keepdim=True), c], dim=1)
        elif self.emode == 'm3':
            x = x * c
        dx = self.join_encoder(x)
        dx = self.out_layer(dx)
        return dx, None
        
class SecretEncoder5(nn.Module):
    """same as SecretEncoder3 but with ch as input"""
    def __init__(self, secret_len, ch=3, base_res=16, resolution=64, joint=False) -> None:
        super().__init__()
        log_resolution = int(np.log2(resolution))
        log_base = int(np.log2(base_res))
        self.secret_len = secret_len
        self.joint = joint
        self.resolution = resolution
        self.secret_scaler = nn.Sequential(
            nn.Linear(secret_len, base_res*base_res*ch),
            nn.SiLU(),
            View(-1, ch, base_res, base_res),
            nn.Upsample(scale_factor=(2**(log_resolution-log_base), 2**(log_resolution-log_base))),  # chx16x16 -> chx256x256
        )  # secret len -> ch x res x res
        if joint:
            self.join_encoder = nn.Sequential(
                conv_nd(2, 2*ch, 2*ch, 3, padding=1),
                nn.SiLU(),
                conv_nd(2, 2*ch, ch, 3, padding=1),
                nn.SiLU()
            )
        self.out_layer = zero_module(conv_nd(2, ch, ch, 3, padding=1))
    
    def copy_encoder_weight(self, ae_model):
        # misses, ignores = self.load_state_dict(ae_state_dict, strict=False)
        return None

    def encode(self, x):
        x = self.secret_scaler(x)
        return x
    
    def forward(self, x, c):
        # x: [B, C, H, W], c: [B, secret_len]
        c = self.encode(c)
        if self.joint:
            x = thf.interpolate(x, size=(self.resolution, self.resolution), mode="bilinear", align_corners=False, antialias=True)
            c = self.join_encoder(torch.cat([x, c], dim=1))
        c = self.out_layer(c)
        return c, None


class SecretEncoder2(nn.Module):
    def __init__(self, secret_len, embed_dim, ddconfig, ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 ema_decay=None,
                 learn_logvar=False) -> None:
        super().__init__()
        log_resolution = int(np.log2(ddconfig.resolution))
        self.secret_len = secret_len
        self.learn_logvar = learn_logvar
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.encoder.conv_out = zero_module(self.encoder.conv_out)
        self.embed_dim = embed_dim

        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))

        if monitor is not None:
            self.monitor = monitor

        self.secret_scaler = nn.Sequential(
            nn.Linear(secret_len, 32*32*ddconfig.out_ch),
            nn.SiLU(),
            View(-1, ddconfig.out_ch, 32, 32),
            nn.Upsample(scale_factor=(2**(log_resolution-5), 2**(log_resolution-5))),  # chx16x16 -> chx256x256
            # zero_module(conv_nd(2, ddconfig.out_ch, ddconfig.out_ch, 3, padding=1))
        )  # secret len -> ch x res x res
        # out_resolution = ddconfig.resolution//(len(ddconfig.ch_mult)-1)
        # self.out_layer = zero_module(conv_nd(2, ddconfig.out_ch, ddconfig.out_ch, 3, padding=1))

        self.use_ema = ema_decay is not None
        if self.use_ema:
            self.ema_decay = ema_decay
            assert 0. < ema_decay < 1.
            self.model_ema = LitEma(self, decay=ema_decay)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)


    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = safe_torch_load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        misses, ignores = self.load_state_dict(sd, strict=False)
        print(f"[SecretEncoder] Restored from {path}, misses: {misses}, ignores: {ignores}")

    def copy_encoder_weight(self, ae_model):
        # misses, ignores = self.load_state_dict(ae_state_dict, strict=False)
        return None
        self.encoder.load_state_dict(ae_model.encoder.state_dict())
        self.quant_conv.load_state_dict(ae_model.quant_conv.state_dict())

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")
    
    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self)

    def encode(self, x):
        h = self.encoder(x)
        posterior = h
        return posterior
    
    def forward(self, x, c):
        # x: [B, C, H, W], c: [B, secret_len]
        c = self.secret_scaler(c)
        x = torch.cat([x, c], dim=1)
        z = self.encode(x)
        # z = self.out_layer(z)
        return z, None

class SecretEncoder(nn.Module):
    def __init__(self, secret_len, embed_dim, ddconfig, ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 ema_decay=None,
                 learn_logvar=False) -> None:
        super().__init__()
        log_resolution = int(np.log2(ddconfig.resolution))
        self.secret_len = secret_len
        self.learn_logvar = learn_logvar
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.embed_dim = embed_dim

        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))

        if monitor is not None:
            self.monitor = monitor

        self.use_ema = ema_decay is not None
        if self.use_ema:
            self.ema_decay = ema_decay
            assert 0. < ema_decay < 1.
            self.model_ema = LitEma(self, decay=ema_decay)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        self.secret_scaler = nn.Sequential(
            nn.Linear(secret_len, 32*32*ddconfig.out_ch),
            nn.SiLU(),
            View(-1, ddconfig.out_ch, 32, 32),
            nn.Upsample(scale_factor=(2**(log_resolution-5), 2**(log_resolution-5))),  # chx16x16 -> chx256x256
            zero_module(conv_nd(2, ddconfig.out_ch, ddconfig.out_ch, 3, padding=1))
        )  # secret len -> ch x res x res
        # out_resolution = ddconfig.resolution//(len(ddconfig.ch_mult)-1)
        self.out_layer = zero_module(conv_nd(2, ddconfig.out_ch, ddconfig.out_ch, 3, padding=1))

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = safe_torch_load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        misses, ignores = self.load_state_dict(sd, strict=False)
        print(f"[SecretEncoder] Restored from {path}, misses: {misses}, ignores: {ignores}")

    def copy_encoder_weight(self, ae_model):
        # misses, ignores = self.load_state_dict(ae_state_dict, strict=False)
        self.encoder.load_state_dict(ae_model.encoder.state_dict())
        self.quant_conv.load_state_dict(ae_model.quant_conv.state_dict())

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")
    
    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self)

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior
    
    def forward(self, x, c):
        # x: [B, C, H, W], c: [B, secret_len]
        c = self.secret_scaler(c)
        x = x + c
        posterior = self.encode(x)
        z = posterior.sample()
        z = self.out_layer(z)
        return z, posterior


class ControlAE(pl.LightningModule):
    def __init__(self,
                 first_stage_key,
                 first_stage_config,
                 control_key,
                 control_config,
                 decoder_config,
                 loss_config,
                 noise_config='__none__',
                 use_ema=False,
                 secret_warmup=False,
                 scale_factor=1.,
                 ckpt_path="__none__",
                 # ── Adapter settings ────────────────────────────
                 adapter_type='none',
                 adapter_dim=16,
                 adapter_depth=1,
                 adapter_rank=4,
                 adapter_scale=1.0,
                 window_size=8,
                 lambda_image=1.0,
                 lambda_perceptual=0.1,
                 lambda_adapter=0.01,
                 use_lpips=True,
                 freeze_FD=False,
                 freeze_F=False,
                 # ── Adapter-only baseline-preserving loss ────────
                 lambda_base=1.0,
                 lambda_base_perc=0.1,
                 lambda_logit=0.01,
                 # ── Latent-space content-adaptive JND ────────────
                 latent_jnd: bool = False,
                 latent_jnd_ksize: int = 11,
                 latent_jnd_sharpness: float = 10.0,
                 latent_jnd_min: float = 0.2,
                 # ── Fine-tune from checkpoint ─────────────────────
                 pretrain_ckpt: str = '__none__',
                 ):
        super().__init__()
        self.scale_factor = scale_factor
        self.control_key = control_key
        self.first_stage_key = first_stage_key
        self.ae = instantiate_from_config(first_stage_config)
        self.control = instantiate_from_config(control_config)
        self.decoder = instantiate_from_config(decoder_config)
        if noise_config != '__none__':
            print('Using noise')
            self.noise = instantiate_from_config(noise_config)
        # copy weights from first stage
        self.control.copy_encoder_weight(self.ae)
        # freeze first stage
        self.ae.eval()
        self.ae.train = disabled_train
        for p in self.ae.parameters():
            p.requires_grad = False

        # ── Build adapter (after freezing ae so LoRA params stay trainable) ──
        self.adapter_type = adapter_type
        self.lambda_image = lambda_image
        self.lambda_perceptual = lambda_perceptual
        self.lambda_adapter = lambda_adapter
        self.use_lpips = use_lpips
        self.lambda_base = lambda_base
        self.lambda_base_perc = lambda_base_perc
        self.lambda_logit = lambda_logit
        self._is_gated_adapter = is_gated_adapter(adapter_type)

        latent_channels = first_stage_config.params.get('embed_dim', 3)
        if adapter_type == 'lora':
            # LoRA is injected directly into the VQ decoder
            self.adapter = build_adapter(
                adapter_type='lora',
                decoder_module=self.ae.decoder,
                adapter_rank=adapter_rank,
                adapter_scale=adapter_scale,
            )
        elif adapter_type != 'none':
            self.adapter = build_adapter(
                adapter_type=adapter_type,
                in_channels=latent_channels,
                adapter_dim=adapter_dim,
                adapter_depth=adapter_depth,
                adapter_rank=adapter_rank,
                adapter_scale=adapter_scale,
                window_size=window_size,
            )
        else:
            self.adapter = None

        self.loss_layer = instantiate_from_config(loss_config)

        # ── Optionally freeze F / D ────────────────────────────────────────
        self.freeze_FD = freeze_FD
        self.freeze_F  = freeze_F
        if freeze_FD:
            for p in self.control.parameters():
                p.requires_grad = False
            for p in self.decoder.parameters():
                p.requires_grad = False
            # eval() 모드로 설정 → BN running_mean/var가 forward 중 업데이트되지 않음
            self.control.eval()
            self.decoder.eval()
            print("[Init] freeze_FD=True: F (control) and D (decoder) are frozen (eval mode).")
        elif freeze_F:
            for p in self.control.parameters():
                p.requires_grad = False
            # F도 eval() 모드 → BN statistics 고정
            self.control.eval()
            print("[Init] freeze_F=True: F (control) is frozen (eval mode). D (decoder) will be trained.")

        # ── Latent-space content-adaptive JND ────────────────────────────────
        self.latent_jnd          = latent_jnd
        self.latent_jnd_ksize    = latent_jnd_ksize
        self.latent_jnd_sharpness = latent_jnd_sharpness
        self.latent_jnd_min      = latent_jnd_min
        if latent_jnd:
            print(f"[LatentJND] enabled  ksize={latent_jnd_ksize}  "
                  f"sharpness={latent_jnd_sharpness}  min={latent_jnd_min}")

        # ── Baseline checkpoint 로드 (fine-tune용) ────────────────────────────
        if pretrain_ckpt and pretrain_ckpt != '__none__':
            import os
            if os.path.exists(pretrain_ckpt):
                sd = safe_torch_load(pretrain_ckpt, map_location='cpu')
                state = sd.get('state_dict', sd)
                miss, unexp = self.load_state_dict(state, strict=False)
                print(f"[PretrainCkpt] Loaded {pretrain_ckpt}")
                print(f"  missing={len(miss)}  unexpected={len(unexp)}")
            else:
                print(f"[PretrainCkpt] WARNING: not found → {pretrain_ckpt}")

        # early training phase
        # self.fixed_input = True
        self.fixed_x = None
        self.fixed_img = None
        self.fixed_input_recon = None
        self.fixed_control = None
        self.register_buffer("fixed_input", torch.tensor(True))

        # secret warmup
        self.secret_warmup = secret_warmup
        self.secret_baselen = 2
        self.secret_len = control_config.params.secret_len
        if self.secret_warmup:
            assert self.secret_len == 2**(int(np.log2(self.secret_len)))

        self.use_ema = use_ema
        if self.use_ema:
            print('Using EMA')
            self.control_ema = LitEma(self.control)
            self.decoder_ema = LitEma(self.decoder)
            print(f"Keeping EMAs of {len(list(self.control_ema.buffers()) + list(self.decoder_ema.buffers()))}.")

        if ckpt_path != '__none__':
            self.init_from_ckpt(ckpt_path, ignore_keys=[])
            # 원본 체크포인트에 fixed_input / noise.step0 버퍼가 없을 경우
            # "학습 완료" 상태로 강제 설정 (curriculum 재시작 방지)
            if not self.fixed_input.item():
                pass  # 체크포인트에서 False로 로드된 경우 유지
            else:
                # 원본 체크포인트에 버퍼가 없어 True로 남은 경우 → False로 설정
                self.fixed_input.fill_(0)
                print("[Init] fixed_input set to False (pretrain_ckpt loaded)")
            if hasattr(self, 'noise') and hasattr(self.noise, 'step0'):
                if not self.noise.is_activated():
                    self.noise.step0.fill_(1)
                    print("[Init] noise activated (pretrain_ckpt loaded)")

        self._print_model_info()

    # ── Info helpers ────────────────────────────────────────────────

    def _print_model_info(self):
        secret_len = getattr(self.control, 'secret_len', '?')
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters()
                               if p.requires_grad)
        frozen_params = total_params - trainable_params

        print("=" * 60)
        print("  RoSteALS + Adapter Model Info")
        print("=" * 60)
        print(f"  adapter_type      : {self.adapter_type}")
        print(f"  trainable params  : {trainable_params:,}")
        print(f"  frozen params     : {frozen_params:,}")
        print(f"  total params      : {total_params:,}")
        print(f"  secret_len        : {secret_len}")
        print(f"  lambda_image      : {self.lambda_image}")
        print(f"  lambda_perceptual : {self.lambda_perceptual}")
        print(f"  lambda_adapter    : {self.lambda_adapter}")
        print(f"  use_lpips         : {self.use_lpips}")
        print(f"  freeze_FD         : {self.freeze_FD}")
        print(f"  freeze_F          : {self.freeze_F}")
        print(f"  lambda_base       : {self.lambda_base}")
        print(f"  lambda_base_perc  : {self.lambda_base_perc}")
        print(f"  lambda_logit      : {self.lambda_logit}")
        print(f"  gated_adapter     : {self._is_gated_adapter}")

        # Verify ae is frozen
        ae_trainable = sum(p.numel() for p in self.ae.parameters()
                           if p.requires_grad)
        if self.adapter_type == 'lora':
            lora_in_ae = sum(
                p.numel() for name, p in self.ae.named_parameters()
                if p.requires_grad
            )
            print(f"  LoRA params in ae : {lora_in_ae:,}  (rest frozen)")
        else:
            assert ae_trainable == 0, (
                f"BUG: ae has {ae_trainable} trainable params – should be 0!"
            )
            print(f"  ae frozen         : OK (0 trainable params)")
        print("=" * 60)

    def _verify_freeze(self):
        """Call this to assert ae (non-LoRA parts) are correctly frozen."""
        if self.adapter_type == 'lora':
            # Only LoRA params should be trainable inside ae
            for name, p in self.ae.named_parameters():
                if p.requires_grad:
                    # Should only be lora_A / lora_B weights
                    assert 'lora_' in name, (
                        f"Unexpected trainable param in ae: {name}"
                    )
        else:
            for name, p in self.ae.named_parameters():
                assert not p.requires_grad, (
                    f"ae param '{name}' should be frozen!"
                )

    def get_warmup_secret(self, old_secret):
        # old_secret: [B, secret_len]
        # new_secret: [B, secret_len]
        if self.secret_warmup:
            bsz = old_secret.shape[0]
            nrepeats = self.secret_len // self.secret_baselen
            new_secret  = torch.zeros((bsz, self.secret_baselen), dtype=torch.float).random_(0, 2).repeat_interleave(nrepeats, dim=1)
            return new_secret.to(old_secret.device)
        else:
            return old_secret
        
    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = safe_torch_load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.control_ema.store(self.control.parameters())
            self.decoder_ema.store(self.decoder.parameters())
            self.control_ema.copy_to(self.control)
            self.decoder_ema.copy_to(self.decoder)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.control_ema.restore(self.control.parameters())
                self.decoder_ema.restore(self.decoder.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.control_ema(self.control)
            self.decoder_ema(self.decoder)

    def compute_loss(self, pred, target):
        # return thf.mse_loss(pred, target, reduction="none").mean(dim=(1, 2, 3))
        lpips_loss = self.lpips_loss(pred, target).mean(dim=[1,2,3])
        pred_yuv = color.rgb_to_yuv((pred + 1) / 2)
        target_yuv = color.rgb_to_yuv((target + 1) / 2)
        yuv_loss = torch.mean((pred_yuv - target_yuv)**2, dim=[2,3])
        yuv_loss = 1.5*torch.mm(yuv_loss, self.yuv_scales).squeeze(1)
        return lpips_loss + yuv_loss

    def _latent_jnd_mask(self, image: torch.Tensor) -> torch.Tensor:
        """
        이미지 픽셀 공간(256×256)에서 local variance(텍스처 강도)를 계산하고
        latent 해상도(64×64)로 다운샘플해 content-adaptive 가중치 맵을 반환.

        반환값: [B, 1, 64, 64], 텍스처 강할수록 크고 평탄할수록 latent_jnd_min에 가까움.
        에너지 보존을 위해 배치별 평균이 1.0이 되도록 정규화.
        """
        k = self.latent_jnd_ksize
        p = k // 2
        # 그레이스케일로 변환 후 국소 분산 계산
        gray = image.mean(dim=1, keepdim=True)          # [B,1,H,W]
        gray_pad = torch.nn.functional.pad(gray, (p, p, p, p), mode='reflect')
        mean_l = torch.nn.functional.avg_pool2d(gray_pad, k, stride=1)
        sq_l   = torch.nn.functional.avg_pool2d(gray_pad**2, k, stride=1)
        var    = (sq_l - mean_l**2).clamp(min=0)        # [B,1,H,W]

        # sigmoid → [0,1], 평탄=0, 텍스처=1 방향
        mask_img = torch.sigmoid(self.latent_jnd_sharpness * var)  # [B,1,256,256]

        # latent 해상도로 다운샘플 (64×64)
        mask_lat = torch.nn.functional.interpolate(
            mask_img, size=(64, 64), mode='bilinear', align_corners=False)

        # 최솟값 floor → 평탄 영역도 최소 latent_jnd_min만큼은 신호 유지
        mask_lat = mask_lat.clamp(min=self.latent_jnd_min)

        # 배치별 에너지 정규화: 평균이 1.0 → 총 워터마크 에너지 보존
        norm = mask_lat.mean(dim=[1, 2, 3], keepdim=True).clamp(min=1e-6)
        mask_lat = mask_lat / norm                       # [B,1,64,64]
        return mask_lat

    def forward(self, x, image, c):
        if self.control.__class__.__name__ == 'SecretEncoder6':
            eps, posterior = self.control(x, c)
        else:
            eps, posterior = self.control(image, c)

        # ── Latent-space content-adaptive JND: 텍스처 영역에 신호 집중 ──────
        if self.latent_jnd and image is not None:
            jnd_mask = self._latent_jnd_mask(image)     # [B,1,64,64]
            eps = eps * jnd_mask                        # 채널 broadcast

        if self.adapter is not None and not isinstance(self.adapter, LoRADecoderAdapter):
            if self._is_gated_adapter:
                # Watermark residual gated: adapter operates on r=F(s), not z_wm
                # r' = r * (1 + alpha * tanh(A(r)))
                # z'_wm = z + r'
                eps = self.adapter(eps)   # adapter modifies residual in-place
                z_wm = x + eps
            else:
                # Additive residual adapters: operate on the full z_wm
                z_wm = x + eps
                z_wm = self.adapter(z_wm)
        else:
            z_wm = x + eps
        # LoRA: no-op here; the update is baked inside ae.decoder
        return z_wm, posterior

    @torch.no_grad()
    def get_input(self, batch, return_first_stage=False, bs=None):
        image = batch[self.first_stage_key]
        control = batch[self.control_key]
        control = self.get_warmup_secret(control)
        if bs is not None:
            image = image[:bs]
            control = control[:bs]
        else:
            bs = image.shape[0]
        # encode image 1st stage
        image = einops.rearrange(image, "b h w c -> b c h w").contiguous()
        x = self.encode_first_stage(image).detach()
        image_rec = self.decode_first_stage(x).detach()
        
        # check if using fixed input (early training phase)
        # if self.training and self.fixed_input:
        if self.fixed_input:
            if self.fixed_x is None:  # first iteration
                print('[TRAINING] Warmup - using fixed input image for now!')
                self.fixed_x = x.detach().clone()[:bs]
                self.fixed_img = image.detach().clone()[:bs]
                self.fixed_input_recon = image_rec.detach().clone()[:bs]
                self.fixed_control = control.detach().clone()[:bs]  # use for log_images with fixed_input option only
            x, image, image_rec = self.fixed_x, self.fixed_img, self.fixed_input_recon
        
        out = [x, control]
        if return_first_stage:
            out.extend([image, image_rec])
        return out

    def decode_first_stage(self, z):
        z = 1./self.scale_factor * z
        image_rec = self.ae.decode(z)
        return image_rec
    
    def encode_first_stage(self, image):
        encoder_posterior = self.ae.encode(image)
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        return self.scale_factor * z

    def shared_step(self, batch):
        x, c, img, _ = self.get_input(batch, return_first_stage=True)

        # Forward: applies secret encoder + optional pre-decoder adapter
        z_wm, posterior = self(x, img, c)

        # Compute baseline latent and output (needed for adapter regularization
        # and baseline-preserving loss in adapter-only experiments)
        z_before_adapter = None   # z_wm without adapter (additive case)
        eps_ref = None            # F(s) without adapter (gated case)
        x_base = None             # G(z_wm) without adapter

        if self.adapter is not None and not isinstance(self.adapter, LoRADecoderAdapter):
            with torch.no_grad():
                if self.control.__class__.__name__ == 'SecretEncoder6':
                    eps_ref, _ = self.control(x, c)
                else:
                    eps_ref, _ = self.control(img, c)

            if self._is_gated_adapter:
                # For gated adapter: baseline is z + F(s) (before residual boost)
                z_before_adapter = (x + eps_ref).detach()
                # x_base: decode the unmodified z_wm (adapter only changed eps)
                if self.freeze_FD:
                    with torch.no_grad():
                        x_base = self.decode_first_stage(z_before_adapter).detach()
            else:
                z_before_adapter = (x + eps_ref).detach()
                if self.freeze_FD:
                    with torch.no_grad():
                        x_base = self.decode_first_stage(z_before_adapter).detach()

        image_rec = self.decode_first_stage(z_wm)
        # resize
        if img.shape[-1] > 256:
            img = thf.interpolate(img, size=(256, 256), mode='bilinear', align_corners=False).detach()
            image_rec = thf.interpolate(image_rec, size=(256, 256), mode='bilinear', align_corners=False)
        # Clamp to [-1,1] before noise to prevent uint8 wrap-around in ImagenetCTransform
        image_rec = torch.clamp(image_rec, -1., 1.)
        if hasattr(self, 'noise') and self.noise.is_activated():
            image_rec_noised = self.noise(image_rec, self.global_step, p=0.9)
        else:
            image_rec_noised = image_rec
        pred = self.decoder(image_rec_noised)

        loss, loss_dict = self.loss_layer(img, image_rec, posterior, c, pred, self.global_step)

        # ── L_adapter regularization ──────────────────────────────
        if self.adapter is not None and self.lambda_adapter > 0:
            if isinstance(self.adapter, LoRADecoderAdapter):
                adapter_loss = self.adapter.adapter_norm()
            else:
                # ||adapter(z) - z||² or ||r' - r||² depending on adapter type
                assert z_before_adapter is not None
                adapter_loss = (z_wm - z_before_adapter).pow(2).mean()
            loss = loss + self.lambda_adapter * adapter_loss
            loss_dict['adapter_loss'] = adapter_loss.detach()

        # ── Baseline-preserving loss (adapter-only experiments) ───
        # Only active when F,D are frozen (adapter-only or adapter+D experiments)
        if self.freeze_FD and x_base is not None:
            x_ada = image_rec  # already clamped; gradients flow through x_ada → adapter
            # Align x_base size with x_ada in case of resize
            if x_base.shape != x_ada.shape:
                x_base = thf.interpolate(x_base, size=x_ada.shape[-2:],
                                         mode='bilinear', align_corners=False)

            # L_base: pixel-space L1 between adapter output and baseline
            l_base = thf.l1_loss(x_ada, x_base)
            loss = loss + self.lambda_base * l_base
            loss_dict['l_base'] = l_base.detach()

            # L_base_lpips: perceptual distillation
            # Note: gradients must flow through x_ada, so no torch.no_grad() here
            if self.lambda_base_perc > 0 and self.use_lpips:
                lpips_base = self.loss_layer.perceptual_loss(
                    x_ada.contiguous(), x_base.detach().contiguous()
                ).mean()
                loss = loss + self.lambda_base_perc * lpips_base
                loss_dict['l_base_lpips'] = lpips_base.detach()

            # L_logit: logit distillation — keep D(x_ada) close to D(x_base)
            # D is frozen (eval mode), but gradients still flow through x_ada → D → logit_ada
            if self.lambda_logit > 0:
                with torch.no_grad():
                    logit_base = self.decoder(x_base.detach())
                logit_ada = self.decoder(x_ada)
                l_logit = thf.mse_loss(logit_ada, logit_base.detach())
                loss = loss + self.lambda_logit * l_logit
                loss_dict['l_logit'] = l_logit.detach()

        bit_acc = loss_dict["bit_acc"]
        bit_acc_ = bit_acc.item()

        # PSNR (dB) — images in [-1,1], convert to [0,1] first
        with torch.no_grad():
            img_01 = (img.detach().clamp(-1, 1) + 1) / 2
            rec_01 = (image_rec.detach().clamp(-1, 1) + 1) / 2
            mse = thf.mse_loss(img_01, rec_01).clamp(min=1e-10)
            psnr = -10 * torch.log10(mse)   # log10(1/x) = -log10(x), device 불일치 방지
            loss_dict['psnr'] = psnr

        if (bit_acc_ > 0.98) and (not self.fixed_input) and hasattr(self, 'noise') and self.noise.is_activated():
            self.loss_layer.activate_ramp(self.global_step)

        if (bit_acc_ > 0.95) and (not self.fixed_input):
            if hasattr(self, 'noise') and (not self.noise.is_activated()):
                self.noise.activate(self.global_step)

        if (bit_acc_ > 0.9) and self.fixed_input:
            print(f'[TRAINING] High bit acc ({bit_acc_}) achieved, switch to full image dataset training.')
            self.fixed_input.fill_(0)
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch)

        # Guard against NaN loss (can arise from extreme imagenet_c corruptions)
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"[WARNING] NaN/Inf loss at step {self.global_step}, skipping batch.")
            return None

        # prog_bar에는 핵심 지표만 표시 (loss, bit_acc, psnr)
        prog_bar_keys = {'loss', 'bit_acc', 'psnr'}
        for key, val in loss_dict.items():
            self.log(f"train/{key}", val,
                     prog_bar=(key in prog_bar_keys),
                     logger=True, on_step=True, on_epoch=True)

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _, loss_dict_no_ema = self.shared_step(batch)
        loss_dict_no_ema = {f"val/{key}": val for key, val in loss_dict_no_ema.items() if key != 'img_lw'}
        with self.ema_scope():
            _, loss_dict_ema = self.shared_step(batch)
            loss_dict_ema = {'val/' + key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}
        self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
    
    @torch.no_grad()
    def log_images(self, batch, fixed_input=False, **kwargs):
        log = dict()
        if fixed_input and self.fixed_img is not None:
            x, c, img, img_recon = self.fixed_x, self.fixed_control, self.fixed_img, self.fixed_input_recon
        else:
            x, c, img, img_recon = self.get_input(batch, return_first_stage=True)
        x, _ = self(x, img, c)
        image_out = self.decode_first_stage(x)
        if hasattr(self, 'noise') and self.noise.is_activated():
            img_noise = self.noise(image_out, self.global_step, p=1.0)
            log['noised'] = img_noise
        log['input'] = img
        log['output'] = image_out
        log['recon'] = img_recon
        return log
    
    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """체크포인트 로드 시 optimizer 그룹 불일치 자동 처리.
        freeze_FD 변경 등으로 optimizer 구조가 달라졌을 때 가중치만 복원한다."""
        opt_states = checkpoint.get('optimizer_states', [])
        if not opt_states:
            return
        ckpt_n_groups = len(opt_states[0].get('param_groups', []))
        # 현재 설정 기준 예상 그룹 수: freeze_FD면 1, 아니면 1 (params 묶음 하나)
        # 실제 불일치는 그룹 내 파라미터 수 차이이므로 state key 수로 비교
        ckpt_n_params = sum(
            len(g.get('params', [])) for g in opt_states[0].get('param_groups', [])
        )
        # 현재 trainable 파라미터 수
        current_n_params = sum(1 for p in self.parameters() if p.requires_grad)
        if ckpt_n_params != current_n_params:
            print(f"[Checkpoint] Optimizer 파라미터 수 불일치 "
                  f"(ckpt={ckpt_n_params}, model={current_n_params}). "
                  f"가중치만 복원하고 optimizer는 초기화합니다.")
            checkpoint['optimizer_states'] = []

    def configure_optimizers(self):
        lr = self.learning_rate

        if self.freeze_FD:
            # Adapter만 학습 (F+D 동결)
            if self.adapter is None:
                raise ValueError("freeze_FD=True but no adapter is set.")
            if isinstance(self.adapter, LoRADecoderAdapter):
                params = self.adapter.trainable_params()
            else:
                params = list(self.adapter.parameters())

        elif self.freeze_F:
            # D + Adapter 학습 (F만 동결)
            params = list(self.decoder.parameters())
            if self.adapter is not None:
                if isinstance(self.adapter, LoRADecoderAdapter):
                    params = params + self.adapter.trainable_params()
                else:
                    params = params + list(self.adapter.parameters())

        else:
            # F + D + Adapter 모두 학습
            params = list(self.control.parameters()) + list(self.decoder.parameters())
            if self.adapter is not None:
                if isinstance(self.adapter, LoRADecoderAdapter):
                    params = params + self.adapter.trainable_params()
                else:
                    params = params + list(self.adapter.parameters())

        optimizer = torch.optim.AdamW(params, lr=lr)
        return optimizer
    




