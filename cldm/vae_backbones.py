"""
vae_backbones.py — Diffusers VAE wrappers for RoSteALS
========================================================
FluxVAEInterface   : FLUX.1 VAE (f=8, 16채널)
QwenImageVAEInterface : Qwen-Image VAE (f=8, 16채널, 3D causal)
"""
from typing import Optional
import torch
import torch.nn as nn
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution


class _DiffusersVAEBase(nn.Module):
    """공통 유틸: posterior → latent, fallback 로딩."""

    def __init__(self, sample_posterior: bool = False):
        super().__init__()
        self.sample_posterior = sample_posterior

    def _latent_from_posterior(self, posterior):
        if isinstance(posterior, DiagonalGaussianDistribution):
            return posterior.sample() if self.sample_posterior else posterior.mode()
        return posterior  # 이미 tensor

    @staticmethod
    def _load_with_fallback(cls, sources, subfolder, **kwargs):
        errs = []
        for src in sources:
            # 서브폴더 포함 시도
            if subfolder:
                try:
                    m = cls.from_pretrained(src, subfolder=subfolder, **kwargs)
                    print(f"[VAE] loaded {cls.__name__} from '{src}' (subfolder={subfolder})")
                    return m
                except Exception as e:
                    errs.append(f"  {src}/{subfolder}: {e}")
            # 서브폴더 없이 시도
            try:
                m = cls.from_pretrained(src, **kwargs)
                print(f"[VAE] loaded {cls.__name__} from '{src}'")
                return m
            except Exception as e:
                errs.append(f"  {src}: {e}")
        raise RuntimeError(
            f"{cls.__name__} 로드 실패. 시도한 소스:\n" + "\n".join(errs) +
            "\ngated 레포라면 `huggingface-cli login` 후 재시도하거나 "
            "YAML 의 pretrained_path 에 로컬 경로를 지정하세요."
        )


class FluxVAEInterface(_DiffusersVAEBase):
    """FLUX.1 VAE (diffusers AutoencoderKL, f=8, 16채널).

    latent 규약:  z = (raw / scale) - shift
    """

    DEFAULT_SOURCES = [
        "black-forest-labs/FLUX.1-schnell",
        "black-forest-labs/FLUX.1-dev",
    ]

    def __init__(
        self,
        pretrained_path: Optional[str] = None,
        subfolder: str = "vae",
        embed_dim: int = 16,
        sample_posterior: bool = False,
    ):
        super().__init__(sample_posterior=sample_posterior)
        from diffusers import AutoencoderKL

        sources = [pretrained_path] if pretrained_path else self.DEFAULT_SOURCES
        self.vae = self._load_with_fallback(
            AutoencoderKL, sources, subfolder, torch_dtype=torch.float32
        )

        cfg = self.vae.config
        self.embed_dim   = cfg.latent_channels          # 16
        self.scale_ratio = 2 ** (len(cfg.block_out_channels) - 1)  # 8

        # FLUX shift/scale normalization
        self._shift = getattr(cfg, "shift_factor", 0.1159)
        self._scale = getattr(cfg, "scaling_factor", 0.3611)

        if self.embed_dim != embed_dim:
            print(f"[VAE] warning: YAML embed_dim={embed_dim} 이지만 "
                  f"실제 latent_channels={self.embed_dim}")
        print(f"[VAE] FLUX.1  f={self.scale_ratio}  ch={self.embed_dim}  "
              f"shift={self._shift:.4f}  scale={self._scale:.4f}")

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        """image: [B,3,H,W] in [-1,1]  →  z: [B,16,H/8,W/8]"""
        posterior = self.vae.encode(image).latent_dist
        # diffusers DiagonalGaussianDistribution: mode()는 메서드, sample()도 메서드
        raw = posterior.sample() if self.sample_posterior else posterior.mode()
        return (raw / self._scale) - self._shift

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z: [B,16,H/8,W/8]  →  image: [B,3,H,W] in [-1,1]"""
        raw = (z + self._shift) * self._scale
        return self.vae.decode(raw).sample

    # ControlAE 호환 인터페이스
    def copy_encoder_weight(self, *args, **kwargs):
        pass  # VQ-GAN 전용 메서드 — no-op

    def get_last_layer(self):
        return self.vae.decoder.conv_out.weight


class QwenImageVAEInterface(_DiffusersVAEBase):
    """Qwen-Image VAE (diffusers AutoencoderKLQwenImage, f=8, 16채널).

    3D(video) VAE 이므로 [B,C,H,W] 를 [B,C,1,H,W] 로 감싸 단일 프레임으로 처리한다.
    latent 규약:  z = (raw - latents_mean) / latents_std   (채널별)
    """

    DEFAULT_SOURCES = ["Qwen/Qwen-Image"]

    def __init__(
        self,
        pretrained_path: Optional[str] = None,
        subfolder: str = "vae",
        embed_dim: int = 16,
        sample_posterior: bool = False,
        use_slicing: bool = False,
        use_tiling: bool = False,
    ):
        super().__init__(sample_posterior=sample_posterior)
        from diffusers import AutoencoderKLQwenImage

        sources = [pretrained_path] if pretrained_path else self.DEFAULT_SOURCES
        self.vae = self._load_with_fallback(
            AutoencoderKLQwenImage, sources, subfolder, torch_dtype=torch.float32
        )

        if use_slicing:
            self.vae.enable_slicing()
        if use_tiling:
            self.vae.enable_tiling()

        cfg = self.vae.config
        self.embed_dim   = cfg.z_dim
        self.scale_ratio = 2 ** (len(cfg.dim_mult) - 1)

        mean = torch.tensor(cfg.latents_mean, dtype=torch.float32).view(1, cfg.z_dim, 1, 1, 1)
        std  = torch.tensor(cfg.latents_std,  dtype=torch.float32).view(1, cfg.z_dim, 1, 1, 1)
        self.register_buffer("latents_mean", mean, persistent=False)
        self.register_buffer("latents_std",  std,  persistent=False)

        if self.embed_dim != embed_dim:
            print(f"[VAE] warning: YAML embed_dim={embed_dim} 이지만 "
                  f"실제 z_dim={self.embed_dim}")
        print(f"[VAE] Qwen-Image  f={self.scale_ratio}  ch={self.embed_dim}  "
              f"per-channel mean/std normalisation")

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        posterior = self.vae.encode(image.unsqueeze(2)).latent_dist
        z = self._latent_from_posterior(posterior)
        z = (z - self.latents_mean) / self.latents_std
        return z.squeeze(2)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        raw = z.unsqueeze(2) * self.latents_std + self.latents_mean
        return self.vae.decode(raw).sample[:, :, 0]

    def copy_encoder_weight(self, *args, **kwargs):
        pass

    def get_last_layer(self):
        return None
