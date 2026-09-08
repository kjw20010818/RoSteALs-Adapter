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


# ── PixelSeal JND (휘도 + 대비 마스킹) ───────────────────────────────────────

class PixelSealJND(nn.Module):
    """
    PixelSeal/VideoSeal 기반 JND 마스크 (휘도 마스킹 + Sobel 대비 마스킹).

    heatmap_raw: 픽셀별 JND 허용치 (0~1, 작을수록 평탄/민감)
    mask_sigmoid: 공간 가중치 마스크 [min_val, 1]
                  min_val=0 → 완전 억제, min_val=1 → 억제 없음
    """
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]]).view(1,1,3,3).repeat(3,1,1,1)
        ky = torch.tensor([[1.,2.,1.],[0.,0.,0.],[-1.,-2.,-1.]]).view(1,1,3,3).repeat(3,1,1,1)
        klum = torch.tensor([[1.,1.,1.,1.,1.],[1.,2.,2.,2.,1.],[1.,2.,0.,2.,1.],
                              [1.,2.,2.,2.,1.],[1.,1.,1.,1.,1.]]).view(1,1,5,5).repeat(3,1,1,1)
        self.cx   = nn.Conv2d(3,3,3,1,1,bias=False,groups=3); self.cx.weight   = nn.Parameter(kx,   False)
        self.cy   = nn.Conv2d(3,3,3,1,1,bias=False,groups=3); self.cy.weight   = nn.Parameter(ky,   False)
        self.clum = nn.Conv2d(3,3,5,1,2,bias=False,groups=3); self.clum.weight = nn.Parameter(klum, False)

    @torch.no_grad()
    def heatmap_raw(self, imgs):
        """imgs: [0,1] → heatmap in [0, ~0.12]"""
        x  = imgs * 255.0
        la = self.clum(x) / 32.0
        m  = la <= 127
        la[m]  = 17.0 * (1.0 - torch.sqrt(la[m]  / 127.0 + 1e-5))
        la[~m] = 3.0 / 128.0 * (la[~m] - 127.0) + 3.0
        gx = self.cx(x); gy = self.cy(x)
        cm = 0.117 * (16.0 * (gx**2 + gy**2)**1.2 / ((gx**2 + gy**2) + 26.0**2))
        return torch.clamp_min(la + cm - 0.3 * torch.minimum(la, cm), 0.0) / 255.0

    @torch.no_grad()
    def get_mask(self, imgs_01, sharpness=50.0, min_val=0.0):
        """imgs_01: [0,1] → spatial mask in [min_val, 1]"""
        h = self.heatmap_raw(imgs_01)
        base = torch.sigmoid(sharpness * (h - h.mean()))
        return base * (1.0 - min_val) + min_val


# ── FiLM Block ────────────────────────────────────────────────────────────────

class FiLMBlock(nn.Module):
    """Conv residual block with FiLM conditioning."""

    def __init__(self, channels: int, cond_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.film  = nn.Linear(cond_dim, channels * 2)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        film = self.film(cond)[:, :, None, None]
        scale, shift = film.chunk(2, dim=1)
        h = self.norm1(x)
        h = (1 + scale) * h + shift
        h = F.silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        return x + h


# ── FreqJNDAdapter ────────────────────────────────────────────────────────────

class FreqJNDAdapter(nn.Module):
    """
    FFT 밴드패스 + JND 마스크를 활용한 주파수-인지적 워터마크 어댑터.

    Parameters
    ----------
    img_channels  : RGB 채널 수 (3)
    hidden        : FiLM CNN 내부 채널 (default 64)
    n_blocks      : FiLM residual block 수 (default 4)
    secret_len    : 워터마크 비트 수 (default 100)
    cond_dim      : FiLM 조건 벡터 차원 (default 128)
    freq_low      : 정규화 주파수 하한 (0~0.5, default 0.02)  ─┐ FFT
    freq_high     : 정규화 주파수 상한 (0~0.5, default 0.20)  ─┘ 밴드패스
    jnd_ksize     : JND 국소 분산 계산 커널 크기 (default 11)
    jnd_sharpness : JND sigmoid steepness (default 20.0)
    """

    def __init__(
        self,
        img_channels: int = 3,
        hidden: int = 64,
        n_blocks: int = 4,
        secret_len: int = 100,
        cond_dim: int = 128,
        freq_low: float = 0.02,
        freq_high: float = 0.20,
        jnd_ksize: int = 11,
        jnd_sharpness: float = 20.0,
    ):
        super().__init__()

        self.freq_low  = freq_low
        self.freq_high = freq_high
        self.jnd_ksize = jnd_ksize
        self.jnd_sharpness = jnd_sharpness

        # Secret → conditioning vector
        self.secret_enc = nn.Sequential(
            nn.Linear(secret_len, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        # FiLM residual CNN
        self.conv_in = nn.Conv2d(img_channels, hidden, 3, padding=1)
        self.blocks  = nn.ModuleList([FiLMBlock(hidden, cond_dim) for _ in range(n_blocks)])
        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, img_channels, 3, padding=1),
        )
        # zero-init → identity 출발
        nn.init.zeros_(self.conv_out[-1].weight)
        nn.init.zeros_(self.conv_out[-1].bias)

        # Learnable output scale (exp(-4) ≈ 0.018)
        self.log_alpha = nn.Parameter(torch.full((1,), -4.0))
        # 최대 잔차 크기 ([-1,1] 기준): 0.05 → PSNR 손실 상한 ≈ 1~2 dB
        self.max_residual: float = 0.05

    # ── JND Mask ──────────────────────────────────────────────────────────────

    def _jnd_mask(self, x: torch.Tensor) -> torch.Tensor:
        """
        국소 분산(local variance)으로 JND 마스크 계산.

        텍스처 풍부 영역(높은 분산) → 큰 마스크 값 → residual 허용량 증가.
        평탄 영역(낮은 분산) → 작은 마스크 값 → 시각 아티팩트 억제.

        Returns: [B, 1, H, W] in (0, 1)
        """
        k = self.jnd_ksize
        padding = k // 2
        # 그레이스케일로 변환 (채널 평균)
        gray = x.mean(dim=1, keepdim=True)                  # [B, 1, H, W]
        mu   = F.avg_pool2d(gray, k, stride=1, padding=padding)
        var  = F.avg_pool2d((gray - mu) ** 2, k, stride=1, padding=padding)
        mask = torch.sigmoid(self.jnd_sharpness * var)       # (0, 1)
        return mask

    # ── FFT Band-pass ─────────────────────────────────────────────────────────

    def _fft_bandpass(self, residual: torch.Tensor) -> torch.Tensor:
        """
        2D FFT → 중간 주파수 대역만 통과 → IFFT.

        freq 범위: [freq_low, freq_high] (정규화, 0~0.5).
          - 저주파(< freq_low) : 주요 색상 정보, 거의 모든 공격에서 살아남지만
                                  D가 이미 읽고 있는 신호와 겹쳐 간섭 가능.
          - 중주파(in range)   : JPEG Q≥50, blur σ≤2 에서 보존됨 → 타깃 대역.
          - 고주파(> freq_high): JPEG / 리사이즈 / blur 에 취약 → 제외.
        """
        H, W = residual.shape[-2:]
        rfft = torch.fft.rfft2(residual, norm='ortho')       # complex [B,C,H,W//2+1]

        # 정규화 주파수 그리드
        # rfft2: 첫 차원은 양/음 주파수 모두 포함(fftfreq), 마지막만 절반(rfftfreq)
        fh = torch.fft.fftfreq(H,  device=residual.device)   # [H],      ∈ (-0.5, 0.5)
        fw = torch.fft.rfftfreq(W, device=residual.device)   # [W//2+1], ∈ [0, 0.5]
        # |freq| = sqrt(fh^2 + fw^2) → [H, W//2+1]
        freq = (fh[:, None] ** 2 + fw[None, :] ** 2).sqrt()

        # soft band-pass (steep sigmoid edges)
        slope = 100.0
        band = (torch.sigmoid(slope * (freq - self.freq_low)) *
                torch.sigmoid(slope * (self.freq_high - freq)))  # ∈ (0,1)

        filtered = rfft * band[None, None]                    # broadcast
        out = torch.fft.irfft2(filtered, s=(H, W), norm='ortho').real
        return out

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, image_wm: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        image_wm : [B, 3, H, W]  G 출력 워터마크 이미지 in [-1, 1]
        s        : [B, secret_len]  워터마크 비트 (float 0/1)

        Returns
        -------
        image_out : [B, 3, H, W] in [-1, 1]
        """
        # 1. FiLM CNN → spatial residual
        cond = self.secret_enc(s)                             # [B, cond_dim]
        h = self.conv_in(image_wm)
        for blk in self.blocks:
            h = blk(h, cond)
        residual = self.conv_out(h)                           # [B, 3, H, W]

        # 2. FFT 밴드패스: 중주파 성분만 유지
        residual = self._fft_bandpass(residual)               # [B, 3, H, W]

        # 3. JND 마스크: 텍스처 영역에 집중
        jnd = self._jnd_mask(image_wm)                        # [B, 1, H, W]

        # 4. 출력 합성
        alpha = torch.exp(self.log_alpha)
        delta = alpha * jnd * residual
        # 잔차 크기 하드클램프: PSNR 과도 하락 방지
        delta = delta.clamp(-self.max_residual, self.max_residual)
        return (image_wm + delta).clamp(-1.0, 1.0)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ── JpegDCTAdapter ────────────────────────────────────────────────────────────

import math as _math

class JpegDCTAdapter(nn.Module):
    """
    JPEG DCT 생존율 기반 주파수 필터를 사용하는 Post-G 어댑터.

    FFT 글로벌 밴드패스 대신, JPEG이 실제로 처리하는 8×8 DCT 블록 단위로
    필터링하여 JPEG q≥50 구간의 강건성을 집중 개선.

    원리:
      - JPEG 양자화 테이블에서 각 DCT 계수의 생존 가중치를 계산
      - 생존율 높은 계수(저주파 DC·AC) → 워터마크 에너지 집중
      - 생존율 낮은 계수(고주파 AC) → 에너지 제거
      - 이로 인해 JPEG q=75 수준에서도 워터마크가 안정적으로 보존

    파이프라인:
      image_wm → FiLM CNN → residual
               → Block-DCT 필터(JPEG 생존 마스크)
               → JND 공간 마스크
               → image_out = image_wm + alpha * jnd * dct_filtered
    """

    # 표준 JPEG 휘도(Y) 양자화 테이블 (quality=50 기준)
    _Q50_LUMA = [
        [16, 11, 10, 16, 24,  40,  51,  61],
        [12, 12, 14, 19, 26,  58,  60,  55],
        [14, 13, 16, 24, 40,  57,  69,  56],
        [14, 17, 22, 29, 51,  87,  80,  62],
        [18, 22, 37, 56, 68, 109, 103,  77],
        [24, 35, 55, 64, 81, 104, 113,  92],
        [49, 64, 78, 87,103, 121, 120, 101],
        [72, 92, 95, 98,112, 100, 103,  99],
    ]

    def __init__(
        self,
        img_channels: int = 3,
        hidden: int = 64,
        n_blocks: int = 4,
        secret_len: int = 100,
        cond_dim: int = 128,
        jpeg_quality: int = 75,     # 타깃 JPEG 품질 (생존 마스크 기준)
        jnd_ksize: int = 11,
        jnd_sharpness: float = 20.0,
        max_residual: float = 0.12,  # 최대 잔차 크기 (YAML에서 조절)
        overlap_stride: int = 8,     # DCT 블록 오버랩 (8=비겹침, 4=오버랩, 2=최대)
    ):
        super().__init__()

        self.jpeg_quality   = jpeg_quality
        self.jnd_ksize      = jnd_ksize
        self.jnd_sharpness  = jnd_sharpness
        self.overlap_stride = overlap_stride

        # Secret → conditioning
        self.secret_enc = nn.Sequential(
            nn.Linear(secret_len, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        # FiLM residual CNN
        self.conv_in = nn.Conv2d(img_channels, hidden, 3, padding=1)
        self.blocks  = nn.ModuleList([FiLMBlock(hidden, cond_dim) for _ in range(n_blocks)])
        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, img_channels, 3, padding=1),
        )
        nn.init.zeros_(self.conv_out[-1].weight)
        nn.init.zeros_(self.conv_out[-1].bias)

        # Learnable scale
        self.log_alpha = nn.Parameter(torch.full((1,), -4.0))
        self.max_residual: float = max_residual

        # 8×8 직교 DCT-II 행렬 (고정, 학습 안 함)
        dct_mat = self._make_dct_mat(8)
        self.register_buffer('dct_mat', dct_mat)                   # [8, 8]

        # JPEG q=jpeg_quality 생존 마스크 (8×8, 정규화 [0,1])
        survival = self._make_survival_mask(jpeg_quality)
        self.register_buffer('jpeg_survival', survival)             # [8, 8]

    # ── 정적 유틸 ──────────────────────────────────────────────────────────────

    @staticmethod
    def _make_dct_mat(n: int = 8) -> torch.Tensor:
        """직교 DCT-II 행렬 (n×n)."""
        mat = torch.zeros(n, n)
        for k in range(n):
            for m in range(n):
                w = _math.sqrt(1.0 / n) if k == 0 else _math.sqrt(2.0 / n)
                mat[k, m] = w * _math.cos(_math.pi * k * (2 * m + 1) / (2 * n))
        return mat

    @classmethod
    def _make_survival_mask(cls, quality: int = 75) -> torch.Tensor:
        """
        JPEG quality → 8×8 생존 가중치 마스크.

        각 DCT 계수의 양자화 스텝이 클수록 → 공격에 취약 → 낮은 가중치.
        마스크 = 1 / q_step, 최대값으로 정규화 → [0, 1].
        """
        # quality → scale factor (IJG 표준 공식)
        if quality <= 0:
            quality = 1
        if quality < 50:
            scale = 5000.0 / quality
        else:
            scale = 200.0 - 2.0 * quality

        q50 = torch.tensor(cls._Q50_LUMA, dtype=torch.float32)
        q   = torch.clamp(torch.floor((q50 * scale + 50.0) / 100.0), 1.0, 255.0)

        # 생존 가중치: 양자화 스텝 작을수록(정밀도 높을수록) 가중치 큼
        survival = 1.0 / q
        survival = survival / survival.max()   # 정규화 → [0, 1]
        return survival

    # ── JND 마스크 ────────────────────────────────────────────────────────────

    def _jnd_mask(self, x: torch.Tensor) -> torch.Tensor:
        k, p = self.jnd_ksize, self.jnd_ksize // 2
        gray = x.mean(dim=1, keepdim=True)
        mu   = F.avg_pool2d(gray, k, stride=1, padding=p)
        var  = F.avg_pool2d((gray - mu) ** 2, k, stride=1, padding=p)
        return torch.sigmoid(self.jnd_sharpness * var)   # [B,1,H,W]

    # ── Block DCT 필터 ────────────────────────────────────────────────────────

    def _block_dct_filter(self, residual: torch.Tensor,
                          overlap_stride: int = 8) -> torch.Tensor:
        """
        8×8 블록 DCT → JPEG 생존 마스크 적용 → 역 DCT.

        overlap_stride=8 : 기존 비겹침 방식 (학습 시 기본값)
        overlap_stride=4 : 오버랩 OLA 방식 → 블록 경계 아티팩트 억제
        overlap_stride=2 : 최대 오버랩 (부드럽지만 느림)
        """
        B, C, H, W = residual.shape
        stride = overlap_stride

        # 8의 배수로 패딩
        pH = (8 - H % 8) % 8
        pW = (8 - W % 8) % 8
        x  = F.pad(residual, (0, pW, 0, pH), mode='reflect') if (pH or pW) else residual
        _, _, H2, W2 = x.shape

        bc = B * C
        xf = x.reshape(bc, 1, H2, W2)

        # unfold → 겹치는 8×8 블록 추출
        blocks = F.unfold(xf, kernel_size=8, stride=stride)     # [B*C, 64, n_blocks]
        n_blocks = blocks.shape[-1]
        blocks   = blocks.permute(0, 2, 1).reshape(-1, 8, 8)    # [N, 8, 8]

        # 2D DCT: D @ X @ D^T
        D  = self.dct_mat
        Dt = D.T
        dct_blk = torch.matmul(D, torch.matmul(blocks, Dt))

        # JPEG 생존 마스크 적용
        dct_blk = dct_blk * self.jpeg_survival.unsqueeze(0)

        # 역 DCT: D^T @ X @ D
        idct_blk = torch.matmul(Dt, torch.matmul(dct_blk, D))

        # 블록 재조합 (OLA: fold가 합산, 카운트로 나눠 평균)
        idct_blk = idct_blk.reshape(bc, n_blocks, 64).permute(0, 2, 1)
        out_sum  = F.fold(idct_blk, output_size=(H2, W2), kernel_size=8, stride=stride)

        if stride < 8:
            # 겹침 횟수로 나눠 평균 (경계 아티팩트 제거)
            ones    = torch.ones_like(xf)
            cnt     = F.fold(
                F.unfold(ones, kernel_size=8, stride=stride),
                output_size=(H2, W2), kernel_size=8, stride=stride
            ).clamp(min=1)
            out_sum = out_sum / cnt

        out = out_sum.reshape(B, C, H2, W2)[:, :, :H, :W]
        return out

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, image_wm: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        # 1. FiLM CNN → spatial residual
        cond     = self.secret_enc(s)
        h        = self.conv_in(image_wm)
        for blk in self.blocks:
            h = blk(h, cond)
        residual = self.conv_out(h)

        # 2. Block-DCT 필터: JPEG 생존 주파수만 통과
        residual = self._block_dct_filter(residual, overlap_stride=self.overlap_stride)

        # 3. JND 공간 마스크
        jnd = self._jnd_mask(image_wm)

        # 4. 합성 + 하드클램프
        alpha = torch.exp(self.log_alpha)
        delta = alpha * jnd * residual
        delta = delta.clamp(-self.max_residual, self.max_residual)
        return (image_wm + delta).clamp(-1.0, 1.0)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ── SmallAEAdapter ────────────────────────────────────────────────────────────

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


class _BlurPool(nn.Module):
    """
    Anti-aliased strided downsampling (Richards et al., 2019).

    3×3 binomial blur (고정 가중치, 학습 안 함) → stride-2 avg.
    stride-2 Conv의 aliasing을 차단하여 격자 경계 아티팩트를 원천 억제.

    channels : 입력 채널 수 (depthwise 적용)
    """
    def __init__(self, channels: int):
        super().__init__()
        # 3×3 binomial filter: [1,2,1]^T × [1,2,1] / 16
        k = torch.tensor([[1., 2., 1.],
                           [2., 4., 2.],
                           [1., 2., 1.]]) / 16.0
        k = k[None, None].repeat(channels, 1, 1, 1)   # (C, 1, 3, 3)
        self.register_buffer('kernel', k)
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # depthwise blur → stride-2 slicing (aliasing 없는 다운샘플)
        x = F.conv2d(
            F.pad(x, [1, 1, 1, 1], mode='reflect'),
            self.kernel,
            stride=1,
            groups=self.channels,
        )
        return x[:, :, ::2, ::2]


class SmallAEAdapter(nn.Module):
    """
    Small Autoencoder Adapter (Post-G) — v14: LCTC upsample.

    변경 사항 (v14):
      LCTC: bilinear upsample + 3×3 conv → 7×7 ConvTranspose2d
           → 업샘플 spectral artifact(물결) 원천 차단
           → ECCV 2024 "Improving Feature Stability during Upsampling" 기반

    파이프라인 (n_down=1, base_ch=32):
        image_wm [3, 256, 256]
          → conv_in(3→32) + ResBlock×2                         → [32, 256, 256]
          → BlurPool(32) + Conv1×1(32→64) + ResBlock×2        → [64, 128, 128]  ← bottleneck
          → FiLM(s)
          → ConvTranspose2d(64→32, 7×7, stride=2) + ResBlock×2 → [32, 256, 256]
          → FiLM(s) + conv_out(32→3) zero-init                 → delta [3, 256, 256]
          → JND mask × delta → clamp → image_out
    """

    def __init__(
        self,
        img_channels: int     = 3,
        base_ch: int          = 32,
        n_down: int           = 1,
        n_res_blocks: int     = 2,
        secret_len: int       = 100,
        cond_dim: int         = 128,
        log_alpha_init: float = -2.0,
        max_residual: float   = 0.12,
        smooth_delta: bool    = True,
        smooth_sigma: float   = 1.0,
        # Anti-aliased downsampling
        use_blurpool: bool    = True,
        # JND mask
        use_jnd: bool         = True,
        jnd_ksize: int        = 11,
        jnd_sharpness: float  = 20.0,
        # tanh delta limiting (StableSignature style)
        use_tanh: bool        = False,
        # skip connection (encoder → decoder)
        use_skip: bool        = False,
        # LCTC: Large Context Transposed Convolution (ECCV 2024)
        use_lctc: bool        = False,
        lctc_kernel: int      = 7,    # 7 or 11 권장
    ):
        super().__init__()
        self.n_down        = n_down
        self.smooth_delta  = smooth_delta
        self.use_blurpool  = use_blurpool
        self.use_jnd       = use_jnd
        self.use_tanh      = use_tanh
        self.use_skip      = use_skip
        self.use_lctc      = use_lctc
        self.lctc_kernel   = lctc_kernel
        self.jnd_ksize     = jnd_ksize
        self.jnd_sharpness = jnd_sharpness

        # Gaussian smoothing kernel (delta 후처리)
        if smooth_delta:
            ksize = 5
            coords = torch.arange(ksize, dtype=torch.float32) - ksize // 2
            g = torch.exp(-coords**2 / (2 * smooth_sigma**2))
            g = g / g.sum()
            kernel = (g[:, None] * g[None, :])
            kernel = kernel[None, None].repeat(img_channels, 1, 1, 1)
            self.register_buffer('_gauss_kernel', kernel)
            self._gauss_pad = ksize // 2

        # Secret → conditioning
        self.secret_enc = nn.Sequential(
            nn.Linear(secret_len, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        # ── Encoder ────────────────────────────────────────────────────────
        self.enc_in     = nn.Conv2d(img_channels, base_ch, 3, 1, 1)
        self.enc_in_res = nn.Sequential(*[_AEResBlock(base_ch) for _ in range(n_res_blocks)])

        # Downsample stages: A-1 활성화 시 BlurPool, 비활성화 시 stride-2 Conv
        self.enc_down = nn.ModuleList()
        ch = base_ch
        for i in range(n_down):
            ch_out = base_ch * (2 ** (i + 1))
            if use_blurpool:
                # A-1: blur → stride-2 slicing → 1×1 채널 확장 (aliasing 없음)
                blk = nn.Sequential(
                    _BlurPool(ch),
                    nn.Conv2d(ch, ch_out, 1),
                    *[_AEResBlock(ch_out) for _ in range(n_res_blocks)],
                )
            else:
                # 기존 방식: stride-2 Conv (aliasing 발생 가능)
                blk = nn.Sequential(
                    nn.Conv2d(ch, ch_out, 3, stride=2, padding=1),
                    *[_AEResBlock(ch_out) for _ in range(n_res_blocks)],
                )
            self.enc_down.append(blk)
            ch = ch_out

        self.film_bn = _AEFiLM(cond_dim, ch)

        # ── Decoder ────────────────────────────────────────────────────────
        self.dec_up = nn.ModuleList()
        # skip_proj: decoder 업샘플 후 skip concat → 채널 절반으로 (use_skip=True 일 때)
        self.skip_proj = nn.ModuleList() if use_skip else None
        for i in reversed(range(n_down)):
            ch_out = base_ch * (2 ** i)
            if use_lctc:
                # LCTC: Large Context Transposed Convolution (ECCV 2024)
                # 7×7 (or 11×11) ConvTranspose2d → spectral artifact 없는 업샘플
                k   = lctc_kernel
                pad = k // 2
                blk = nn.Sequential(
                    nn.ConvTranspose2d(ch, ch_out, kernel_size=k, stride=2,
                                      padding=pad, output_padding=1),
                    *[_AEResBlock(ch_out) for _ in range(n_res_blocks)],
                )
            else:
                # 기존 bilinear + 3×3 conv (물결 artifact 발생 가능)
                blk = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    nn.Conv2d(ch, ch_out, 3, 1, 1),
                    *[_AEResBlock(ch_out) for _ in range(n_res_blocks)],
                )
            self.dec_up.append(blk)
            if use_skip:
                # cat([dec ch_out, enc skip ch_out]) → ch_out via 1×1
                self.skip_proj.append(nn.Conv2d(ch_out * 2, ch_out, 1))
            ch = ch_out

        self.film_dec = _AEFiLM(cond_dim, ch)

        self.conv_out = nn.Conv2d(ch, img_channels, 3, 1, 1)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

        self.log_alpha    = nn.Parameter(torch.tensor(float(log_alpha_init)))
        self.max_residual = max_residual

    # ── A-2: JND mask ──────────────────────────────────────────────────────

    def _jnd_mask(self, x: torch.Tensor) -> torch.Tensor:
        """
        국소 분산(local variance) 기반 JND 마스크.

        평탄 영역(낮은 분산) → 마스크 값 ≈ 0 → delta 억제 → 격자 비가시화
        텍스처 영역(높은 분산) → 마스크 값 ≈ 1 → delta 통과

        Returns: [B, 1, H, W] in (0, 1)
        """
        k = self.jnd_ksize
        p = k // 2
        gray = x.mean(dim=1, keepdim=True)                           # [B,1,H,W]
        mu   = F.avg_pool2d(gray, k, stride=1, padding=p)
        var  = F.avg_pool2d((gray - mu) ** 2, k, stride=1, padding=p)
        return torch.sigmoid(self.jnd_sharpness * var)               # (0, 1)

    def forward(self, image_wm: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        cond = self.secret_enc(s)

        # Encode (A-1 BlurPool 포함)
        h = F.silu(self.enc_in(image_wm))
        h = self.enc_in_res(h)

        # A-4: skip connection — 다운샘플 전 feature 저장
        skips = []
        if self.use_skip:
            skips.append(h)   # [B, base_ch, 256, 256]

        for blk in self.enc_down:
            h = blk(h)

        # Bottleneck FiLM
        h = self.film_bn(h, cond)

        # Decode (+ skip concat)
        for idx, blk in enumerate(self.dec_up):
            h = blk(h)
            if self.use_skip and self.skip_proj is not None and idx < len(self.skip_proj):
                sk = skips[len(skips) - 1 - idx]          # 역순 skip
                h  = self.skip_proj[idx](torch.cat([h, sk], dim=1))

        h     = self.film_dec(h, cond)
        delta = self.conv_out(h)

        # Gaussian smoothing (기존 아티팩트 완화)
        if self.smooth_delta:
            p = self._gauss_pad
            delta = F.conv2d(
                F.pad(delta, [p, p, p, p], mode='reflect'),
                self._gauss_kernel,
                groups=delta.shape[1],
            )

        # A-2: JND 마스크 — 평탄 영역의 delta 억제
        # A-3: tanh vs clamp — tanh는 경계에서 부드럽게 수렴(StableSignature style)
        alpha = torch.exp(self.log_alpha)
        if self.use_jnd:
            jnd = self._jnd_mask(image_wm)
            raw = alpha * jnd * delta
        else:
            raw = alpha * delta

        if self.use_tanh:
            delta = self.max_residual * torch.tanh(raw / self.max_residual)
        else:
            delta = raw.clamp(-self.max_residual, self.max_residual)

        return (image_wm + delta).clamp(-1.0, 1.0)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ── NAFNetAdapter ─────────────────────────────────────────────────────────────

class _SimpleGate(nn.Module):
    """NAFNet 핵심 활성화: 채널을 반으로 나눠 element-wise 곱."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class _NAFBlock(nn.Module):
    """
    NAFNet Basic Block (ECCV 2022).
    - Depthwise separable conv  → checkerboard 아티팩트 없음
    - SimpleGate               → 비선형성 (GELU/ReLU 없음)
    - Simplified Channel Attention (SCA)
    - FiLM 조건 (secret 주입)
    """
    def __init__(self, ch: int, cond_dim: int, dw_expand: int = 2, ffn_expand: int = 2):
        super().__init__()
        dw_ch  = ch * dw_expand
        ffn_ch = ch * ffn_expand

        self.norm1 = nn.LayerNorm(ch)
        self.norm2 = nn.LayerNorm(ch)

        # Spatial mixing
        self.conv1   = nn.Conv2d(ch, dw_ch, 1)                       # 1×1 expand
        self.conv2   = nn.Conv2d(dw_ch, dw_ch, 3, 1, 1, groups=dw_ch)  # depthwise 3×3
        self.conv3   = nn.Conv2d(dw_ch // 2, ch, 1)                  # 1×1 project (after gate)
        self.gate    = _SimpleGate()
        self.sca     = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_ch // 2, dw_ch // 2, 1),
        )

        # FFN
        self.conv4   = nn.Conv2d(ch, ffn_ch, 1)
        self.conv5   = nn.Conv2d(ffn_ch // 2, ch, 1)
        self.gate2   = _SimpleGate()

        # FiLM conditioning (secret → scale/shift for norm1 & norm2)
        self.film1   = nn.Linear(cond_dim, ch * 2)
        self.film2   = nn.Linear(cond_dim, ch * 2)

        # Learnable residual scale (NAFNet style)
        self.beta    = nn.Parameter(torch.ones(1, ch, 1, 1) * 1e-3)
        self.gamma   = nn.Parameter(torch.ones(1, ch, 1, 1) * 1e-3)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # ── Spatial branch ────────────────────────────────────────────
        h = x.permute(0, 2, 3, 1)          # BHWC for LayerNorm
        h = self.norm1(h).permute(0, 3, 1, 2)

        # FiLM from secret
        f1 = self.film1(cond)[:, :, None, None]
        sc1, sh1 = f1.chunk(2, dim=1)
        h = (1 + sc1) * h + sh1

        h = self.conv1(h)
        h = self.conv2(h)
        h = self.gate(h)
        h = h * self.sca(h)
        h = self.conv3(h)
        x = x + h * self.beta

        # ── FFN branch ────────────────────────────────────────────────
        h = x.permute(0, 2, 3, 1)
        h = self.norm2(h).permute(0, 3, 1, 2)

        f2 = self.film2(cond)[:, :, None, None]
        sc2, sh2 = f2.chunk(2, dim=1)
        h = (1 + sc2) * h + sh2

        h = self.conv4(h)
        h = self.gate2(h)
        h = self.conv5(h)
        x = x + h * self.gamma
        return x


class NAFNetAdapter(nn.Module):
    """
    NAFNet-tiny 기반 Post-G 워터마크 어댑터.

    특징:
    - Depthwise separable conv → ConvTranspose2d 없음 → 아티팩트 없음
    - SimpleGate 활성화 (GELU/ReLU 없음)
    - FiLM으로 secret 조건 주입
    - 선택적 U-Net 구조 (n_down>0): bilinear 업샘플 사용
    - 출력 zero-init → 초기에 identity 보장

    파라미터 수 (width=32, n_blocks=8, n_down=0): ~390K
    """

    def __init__(
        self,
        img_channels: int  = 3,
        width: int         = 32,    # 채널 수 (NAFNet tiny=32)
        n_blocks: int      = 8,     # NAFBlock 수
        n_down: int        = 0,     # 다운샘플 스테이지 수 (0=flat, 1=U-Net 1단)
        secret_len: int    = 100,
        cond_dim: int      = 128,
        dw_expand: int     = 2,
        ffn_expand: int    = 2,
        max_residual: float = 0.08,
    ):
        super().__init__()
        self.n_down      = n_down
        self.max_residual = max_residual

        # Secret → conditioning vector
        self.secret_enc = nn.Sequential(
            nn.Linear(secret_len, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        # Input projection
        self.conv_in = nn.Conv2d(img_channels, width, 3, 1, 1)

        if n_down == 0:
            # ── Flat (no bottleneck) ─────────────────────────────────
            self.blocks = nn.ModuleList([
                _NAFBlock(width, cond_dim, dw_expand, ffn_expand)
                for _ in range(n_blocks)
            ])
        else:
            # ── U-Net (with bottleneck) ──────────────────────────────
            blk_per = max(n_blocks // (2 * n_down + 1), 1)
            self.enc_blocks = nn.ModuleList()
            self.downs      = nn.ModuleList()
            self.ups        = nn.ModuleList()
            self.dec_blocks = nn.ModuleList()
            self.enc_proj   = nn.ModuleList()

            ch = width
            enc_chs = []
            for i in range(n_down):
                self.enc_blocks.append(nn.ModuleList([
                    _NAFBlock(ch, cond_dim, dw_expand, ffn_expand)
                    for _ in range(blk_per)
                ]))
                enc_chs.append(ch)
                ch_out = ch * 2
                self.downs.append(nn.Conv2d(ch, ch_out, 2, 2))  # stride-2 (encoding only)
                ch = ch_out

            # Bottleneck
            self.mid_blocks = nn.ModuleList([
                _NAFBlock(ch, cond_dim, dw_expand, ffn_expand)
                for _ in range(blk_per)
            ])

            for i in reversed(range(n_down)):
                ch_skip = enc_chs[i]
                self.ups.append(nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    nn.Conv2d(ch, ch_skip, 1),
                ))
                self.enc_proj.append(nn.Conv2d(ch_skip * 2, ch_skip, 1))
                ch = ch_skip
                self.dec_blocks.append(nn.ModuleList([
                    _NAFBlock(ch, cond_dim, dw_expand, ffn_expand)
                    for _ in range(blk_per)
                ]))

        # Output projection (zero-init → identity at start)
        self.conv_out = nn.Conv2d(width, img_channels, 3, 1, 1)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

        self.log_alpha = nn.Parameter(torch.full((1,), -4.0))

    def _run_blocks(self, h, blocks, cond):
        for blk in blocks:
            h = blk(h, cond)
        return h

    def forward(self, image_wm: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        cond = self.secret_enc(s)
        h    = self.conv_in(image_wm)

        if self.n_down == 0:
            h = self._run_blocks(h, self.blocks, cond)
        else:
            # Encoder
            skips = []
            for enc_blks, down in zip(self.enc_blocks, self.downs):
                h = self._run_blocks(h, enc_blks, cond)
                skips.append(h)
                h = down(h)

            # Bottleneck
            h = self._run_blocks(h, self.mid_blocks, cond)

            # Decoder (bilinear upsample → no checkerboard)
            for up, proj, dec_blks, skip in zip(
                    self.ups, self.enc_proj, self.dec_blocks, reversed(skips)):
                h = up(h)
                h = proj(torch.cat([h, skip], dim=1))
                h = self._run_blocks(h, dec_blks, cond)

        delta = self.conv_out(h)
        alpha = torch.exp(self.log_alpha)
        delta = (alpha * delta).clamp(-self.max_residual, self.max_residual)
        return (image_wm + delta).clamp(-1.0, 1.0)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ── ChannelBNAdapter ──────────────────────────────────────────────────────────

class ChannelBNAdapter(nn.Module):
    """
    Channel Bottleneck Adapter (Post-G).

    Spatial Bottleneck(SmallAE)의 격자 아티팩트 문제를 해결:
      SmallAE     : 256→[stride-2]→128→[bilinear]→256  (격자 패턴 발생)
      ChannelBN   : Global Avg Pool → MLP → channel-wise FiLM  (공간 패턴 없음)

    파이프라인 (base_ch=32):
        image_wm [3, 256, 256]
          └─ enc_in       → [32, 256, 256]   # full-res local conv
          └─ ResBlocks    → [32, 256, 256]   # 국소 처리
          └─ GAP          → [32, 1, 1]       # 전역 맥락 압축 (공간 grid 없음)
          └─ MLP(secret)  → [ch*2]           # secret + 전역맥락 → scale/shift
          └─ channel FiLM → [32, 256, 256]   # 채널별 변조 (공간 격자 없음)
          └─ ResBlocks    → [32, 256, 256]   # 정제
          └─ conv_out     → [3, 256, 256]    # delta (zero-init)
        delta → clamp → image_out
    """

    def __init__(
        self,
        base_ch:    int = 32,
        n_res_enc:  int = 3,    # encoder ResBlock 수
        n_res_dec:  int = 3,    # decoder ResBlock 수
        secret_len: int = 100,
        cond_dim:   int = 128,
        max_residual: float = 0.12,
        smooth_delta: bool = True,
        smooth_sigma: float = 1.0,
    ):
        super().__init__()
        self.max_residual = max_residual
        self.smooth_delta = smooth_delta
        self.smooth_sigma = smooth_sigma

        # Secret 인코더
        self.secret_enc = nn.Sequential(
            nn.Linear(secret_len, cond_dim), nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        # Encoder: full-resolution local conv
        self.enc_in  = nn.Conv2d(3, base_ch, 3, 1, 1)
        self.enc_res = nn.Sequential(*[_AEResBlock(base_ch) for _ in range(n_res_enc)])

        # Channel Bottleneck: GAP → MLP(secret+global) → scale/shift
        # MLP 입력: GAP feature(base_ch) + secret(cond_dim)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ch_mlp = nn.Sequential(
            nn.Linear(base_ch + cond_dim, base_ch * 2), nn.SiLU(),
            nn.Linear(base_ch * 2, base_ch * 2),        # → scale, shift
        )
        nn.init.zeros_(self.ch_mlp[-1].weight)
        nn.init.zeros_(self.ch_mlp[-1].bias)

        # Decoder: full-resolution local conv (FiLM 후)
        self.dec_res = nn.Sequential(*[_AEResBlock(base_ch) for _ in range(n_res_dec)])
        self.conv_out = nn.Conv2d(base_ch, 3, 3, 1, 1)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

        self.log_alpha = nn.Parameter(torch.zeros(1) - 2.0)

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        # s: (B, secret_len) or (B, 100)
        if s.dim() == 1:
            s = s.unsqueeze(0)
        cond = self.secret_enc(s)   # (B, cond_dim)

        # Encoder
        h = self.enc_in(x)          # (B, base_ch, H, W)
        h = self.enc_res(h)         # (B, base_ch, H, W) - full resolution 유지

        # Channel Bottleneck (공간 격자 없음)
        g = self.gap(h).flatten(1)              # (B, base_ch)
        mlp_in = torch.cat([g, cond], dim=1)    # (B, base_ch + cond_dim)
        sv = self.ch_mlp(mlp_in)                # (B, base_ch*2)
        scale, shift = sv.chunk(2, dim=1)       # (B, base_ch) each
        h = (1.0 + scale[:, :, None, None]) * h + shift[:, :, None, None]  # channel FiLM

        # Decoder
        h = self.dec_res(h)         # (B, base_ch, H, W)
        delta = self.conv_out(h)    # (B, 3, H, W)

        alpha = torch.exp(self.log_alpha)
        delta = (alpha * delta).clamp(-self.max_residual, self.max_residual)

        if self.smooth_delta and self.smooth_sigma > 0:
            from kornia.filters import gaussian_blur2d
            ks = int(self.smooth_sigma * 4) | 1  # 홀수 커널
            ks = max(ks, 3)
            delta = gaussian_blur2d(delta, (ks, ks), (self.smooth_sigma, self.smooth_sigma))

        out = (x + delta).clamp(-1.0, 1.0)
        return out

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ── ControlAEPostG (LightningModule) ─────────────────────────────────────────

# ── PatchGAN Discriminator ────────────────────────────────────────────────────
class PatchDiscriminator(nn.Module):
    """70×70 PatchGAN discriminator (pix2pix style).
    LSGAN 방식으로 학습 — 안정적이고 아티팩트 억제에 효과적.
    입력: RGB [-1,1] 이미지
    출력: (B, 1, H', W') — patch별 real/fake 점수
    """
    def __init__(self, in_ch: int = 3, base_ch: int = 64, n_layers: int = 3):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, base_ch, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        ch = base_ch
        for i in range(1, n_layers):
            ch_next = min(ch * 2, 512)
            layers += [
                nn.Conv2d(ch, ch_next, 4, 2, 1),
                nn.InstanceNorm2d(ch_next),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            ch = ch_next
        ch_next = min(ch * 2, 512)
        layers += [
            nn.Conv2d(ch, ch_next, 4, 1, 1),
            nn.InstanceNorm2d(ch_next),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch_next, 1, 4, 1, 1),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """분리 가능 가우시안 블러. CSF 대역 분해용."""
    k  = int(2 * round(3 * sigma) + 1)
    ax = torch.arange(k, device=x.device, dtype=x.dtype) - k // 2
    g  = torch.exp(-ax.pow(2) / (2 * sigma ** 2))
    g  = g / g.sum()
    C  = x.shape[1]
    x  = thf.conv2d(thf.pad(x, (k // 2, k // 2, 0, 0), mode='reflect'),
                    g.view(1, 1, 1, k).expand(C, 1, 1, k), groups=C)
    return thf.conv2d(thf.pad(x, (0, 0, k // 2, k // 2), mode='reflect'),
                      g.view(1, 1, k, 1).expand(C, 1, k, 1), groups=C)


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
        # SmallAEAdapter 전용: BlurPool, JND 마스크, tanh, skip, LCTC
        adapter_use_blurpool: bool = False,       # anti-aliased downsampling
        adapter_use_jnd: bool = False,            # JND 마스크 (평탄 영역 delta 억제)
        adapter_jnd_ksize_ae: int = 11,           # SmallAE JND 커널 크기
        adapter_jnd_sharpness_ae: float = 20.0,   # SmallAE JND sigmoid steepness
        adapter_use_tanh: bool = False,           # tanh delta limiting (StableSignature style)
        adapter_use_skip: bool = False,           # skip connection (encoder → decoder)
        adapter_use_lctc: bool = False,           # LCTC: Large Context Transposed Conv (ECCV 2024)
        adapter_lctc_kernel: int = 7,             # LCTC 커널 크기 (7 or 11)
        # Discriminator lr ratio (lambda_adv>0 일 때, default: G lr의 0.5배)
        disc_lr_ratio: float = 0.5,
        # LightViTAdapter 전용
        adapter_vit_patch_size: int = 16,
        adapter_vit_n_heads: int = 4,
        # Robustness-first loss weights
        lambda_preserve: float = 1.0,
        lambda_img: float = 0.1,
        lambda_lpips: float = 0.0,       # LPIPS perceptual loss weight (0=비활성)
        lambda_freq: float = 0.0,        # delta 고주파 TV 페널티 (0=비활성, 아티팩트 억제)
        lambda_chroma: float = 0.0,      # adapter delta 색차(Pb/Pr) 패널티 — 물결 억제
        lambda_csf: float = 0.0,         # CSF 가중 가시성 패널티 (0=비활성)
        csf_weights: tuple = (0.2, 0.6, 1.0, 1.0, 0.3),  # 옥타브 대역 B0~B4 가중치
        lambda_adv: float = 0.0,         # PatchGAN adversarial loss weight (0=비활성)
        attack_prob: float = 0.9,
        # Texture masking
        use_texture_mask: bool = False,  # Laplacian 기반 텍스처 마스크로 delta 조절
        texture_mask_power: float = 0.5, # 마스크 강도 (0=균등, 1=완전 마스킹)
        # PixelSeal JND 학습 통합
        use_ps_jnd: bool = False,        # 학습 중 PixelSeal JND 마스킹 적용
        ps_jnd_min_val: float = 0.5,     # 마스크 최솟값 [0,1], 클수록 억제 약함
        ps_jnd_sharpness: float = 50.0,  # sigmoid steepness
        # VQ보정 baked-in 학습 (JND 없이 순수 VQ 보정만)
        use_vq_correction: bool = False,  # W' = x_orig + (W - G(z)) 로 loss 계산
        # VQ보정+JND baked-in 학습 (v17)
        use_vq_jnd: bool = False,        # 손실을 W''=x+(W-Gz)*JND에서 계산
        vq_jnd_sharpness: float = 200.0, # 분산기반 JND sharpness
        vq_jnd_min_val: float = 0.05,    # 분산기반 JND min_val
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
        self.lambda_freq        = lambda_freq
        self.lambda_chroma      = lambda_chroma
        self.lambda_csf         = lambda_csf
        self.csf_weights        = tuple(csf_weights)
        self.lambda_adv         = lambda_adv
        self.disc_lr_ratio      = disc_lr_ratio
        self.attack_prob        = attack_prob
        self.use_texture_mask   = use_texture_mask
        self.texture_mask_power = texture_mask_power
        self.use_ps_jnd         = use_ps_jnd
        self.ps_jnd_min_val     = ps_jnd_min_val
        self.ps_jnd_sharpness   = ps_jnd_sharpness
        self.use_vq_correction  = use_vq_correction
        self.use_vq_jnd         = use_vq_jnd
        self.vq_jnd_sharpness   = vq_jnd_sharpness
        self.vq_jnd_min_val     = vq_jnd_min_val
        self.finetune_decoder   = finetune_decoder
        if use_ps_jnd:
            self.ps_jnd = PixelSealJND()
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

        # ── PatchGAN Discriminator (lambda_adv > 0 일 때만 사용) ───────
        if lambda_adv > 0:
            self.discriminator = PatchDiscriminator(in_ch=3, base_ch=64, n_layers=3)
            self.automatic_optimization = False  # GAN: manual optimizer step
        else:
            self.discriminator = None

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
                img_channels      = 3,
                base_ch           = adapter_hidden,
                n_down            = adapter_n_down,
                n_res_blocks      = adapter_n_blocks,
                secret_len        = secret_len,
                cond_dim          = adapter_cond_dim,
                smooth_delta      = adapter_smooth_delta,
                smooth_sigma      = adapter_smooth_sigma,
                use_blurpool      = adapter_use_blurpool,
                use_jnd           = adapter_use_jnd,
                jnd_ksize         = adapter_jnd_ksize_ae,
                jnd_sharpness     = adapter_jnd_sharpness_ae,
                use_tanh          = adapter_use_tanh,
                use_skip          = adapter_use_skip,
                use_lctc          = adapter_use_lctc,
                lctc_kernel       = adapter_lctc_kernel,
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
        elif adapter_type == 'channel_bn':
            self.adapter = ChannelBNAdapter(
                base_ch       = adapter_hidden,
                n_res_enc     = adapter_n_blocks,
                n_res_dec     = adapter_n_blocks,
                secret_len    = secret_len,
                cond_dim      = adapter_cond_dim,
                max_residual  = adapter_max_residual,
                smooth_delta  = adapter_smooth_delta,
                smooth_sigma  = adapter_smooth_sigma,
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
        _pc = getattr(self.adapter, 'param_count', None)
        adap_p = _pc if isinstance(_pc, int) else (_pc() if callable(_pc) else sum(p.numel() for p in self.adapter.parameters()))
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
        if self.adapter_type == 'channel_bn':
            print(f"  bottleneck        : Channel (GAP+MLP, 공간 격자 없음)")
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

        # ── VQ보정만 (JND 없음) ──────────────────────────────────────────
        # W' = x_orig + (W - G(z))  → 어댑터가 G 분포가 아닌 픽셀 공간에서 학습
        if self.use_vq_correction and not self.use_vq_jnd:
            with torch.no_grad():
                Gz = self.decode_first_stage(x).clamp(-1., 1.)
            wm        = image_out - Gz          # pure watermark delta
            image_out = (img.detach() + wm).clamp(-1., 1.)

        # ── VQ보정+JND baked-in (v17) ────────────────────────────────────
        # W'' = x_orig + (W - Gz) * JND(x_orig)
        # 손실 전체를 W''에서 계산 → 어댑터가 JND 억제를 보상하도록 학습
        if self.use_vq_jnd:
            with torch.no_grad():
                # Gz = G(z): 워터마크 없는 VQ-GAN 재구성
                Gz = self.decode_first_stage(x).clamp(-1., 1.)
                # 분산 기반 JND 마스크
                gray = img.clamp(-1,1).mean(dim=1, keepdim=True)
                p = 7  # ksize//2
                gray_pad = thf.pad(gray, (p,p,p,p), mode='reflect')
                mean_l = thf.avg_pool2d(gray_pad, 15, stride=1)
                sq_l   = thf.avg_pool2d(gray_pad**2, 15, stride=1)
                var    = (sq_l - mean_l**2).clamp(min=0)
                jnd_mask = torch.sigmoid(
                    self.vq_jnd_sharpness * var
                ).clamp(min=self.vq_jnd_min_val, max=1.0)
            # 순수 워터마크 신호
            wm        = image_out - Gz
            # W'' = x_orig + wm * JND_mask  (gradient flows through wm → image_out)
            image_out = (img.detach() + wm * jnd_mask).clamp(-1., 1.)

        # PixelSeal JND 마스킹: delta를 텍스처 영역에 집중
        # W_jnd = img + (image_out - img) * ps_mask
        if self.use_ps_jnd:
            # ps_mask 계산은 no_grad (img 고정), 마스킹 자체는 grad 흐름 유지
            with torch.no_grad():
                imgs_01 = (img.clamp(-1, 1) * 0.5 + 0.5)
                ps_mask = self.ps_jnd.get_mask(imgs_01,
                                               sharpness=self.ps_jnd_sharpness,
                                               min_val=self.ps_jnd_min_val)
            delta     = image_out - img.clamp(-1, 1).detach()
            image_out = (img.detach() + delta * ps_mask).clamp(-1., 1.)

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

        # 색차 패널티: adapter delta의 Pb/Pr 억제 → 보라·초록 물결 제거
        if self.lambda_chroma > 0:
            dlt = image_out - image_wm.detach()
            r, g, b = dlt[:, 0:1], dlt[:, 1:2], dlt[:, 2:3]
            y  = 0.299 * r + 0.587 * g + 0.114 * b
            pb = (b - y) / 1.772
            pr = (r - y) / 1.402
            l_chroma = pb.abs().mean() + pr.abs().mean()
            loss = loss + self.lambda_chroma * l_chroma
            loss_dict['l_chroma'] = l_chroma.detach()

        # CSF 가중 가시성 패널티: delta를 옥타브 대역으로 나눠 대역별로 다르게 벌점.
        # 사람 눈의 대비감도는 중간 주파수에서 최대이므로 B2/B3에 큰 가중치를 주면
        # 어댑터가 정보를 눈에 덜 띄는 B0(입자)·B4(완만한 그라디언트)로 옮기게 된다.
        if self.lambda_csf > 0:
            dlt    = image_out - img.detach()
            blurs  = [_gaussian_blur(dlt, s) for s in (1.0, 2.0, 4.0, 8.0)]
            bands  = [dlt - blurs[0]] + \
                     [blurs[i] - blurs[i + 1] for i in range(len(blurs) - 1)] + \
                     [blurs[-1]]
            l_csf  = sum(w * b.pow(2).mean() for w, b in zip(self.csf_weights, bands))
            loss   = loss + self.lambda_csf * l_csf
            loss_dict['l_csf'] = l_csf.detach()

        # 주파수 페널티: delta의 고주파 성분 억제 (격자 아티팩트 방지)
        if self.lambda_freq > 0:
            delta_f = image_out - image_wm
            # Total Variation: 인접 픽셀 차분의 L1 합 (고주파=큰 TV)
            tv_h = (delta_f[:, :, 1:, :] - delta_f[:, :, :-1, :]).abs().mean()
            tv_w = (delta_f[:, :, :, 1:] - delta_f[:, :, :, :-1]).abs().mean()
            l_freq = tv_h + tv_w
            loss = loss + self.lambda_freq * l_freq
            loss_dict['l_freq'] = l_freq.detach()

        # Adversarial loss (Generator): LSGAN style — D가 real로 예측하도록
        if self.lambda_adv > 0 and self.discriminator is not None:
            pred_fake = self.discriminator(image_out)
            l_adv = thf.mse_loss(pred_fake, torch.ones_like(pred_fake))
            loss = loss + self.lambda_adv * l_adv
            loss_dict['l_adv_g'] = l_adv.detach()

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
        # ── GAN mode: manual optimization ────────────────────────────────
        if self.lambda_adv > 0 and self.discriminator is not None:
            opt_g, opt_d = self.optimizers()

            # ── Generator step ──────────────────────────────────────────
            loss_g, loss_dict = self.shared_step(batch)
            opt_g.zero_grad()
            self.manual_backward(loss_g)
            self.clip_gradients(opt_g, gradient_clip_val=1.0, gradient_clip_algorithm='norm')
            opt_g.step()

            # ── Discriminator step ──────────────────────────────────────
            x, img, s = self.get_input(batch)
            with torch.no_grad():
                image_out, _, _ = self(x, img, s)
                image_out = image_out.detach()

            pred_real = self.discriminator(img.clamp(-1, 1))
            pred_fake = self.discriminator(image_out)
            l_d_real  = thf.mse_loss(pred_real, torch.ones_like(pred_real))
            l_d_fake  = thf.mse_loss(pred_fake, torch.zeros_like(pred_fake))
            loss_d    = 0.5 * (l_d_real + l_d_fake)

            opt_d.zero_grad()
            self.manual_backward(loss_d)
            self.clip_gradients(opt_d, gradient_clip_val=1.0, gradient_clip_algorithm='norm')
            opt_d.step()

            loss_dict['l_adv_d'] = loss_d.detach()
            self.log_dict({f'train/{k}': v for k, v in loss_dict.items()},
                          prog_bar=True, logger=True, on_step=True, on_epoch=True,
                          sync_dist=True)
            return loss_g

        # ── Normal mode: automatic optimization ──────────────────────────
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
        param_groups = [
            {'params': list(self.adapter.parameters()), 'lr': lr, 'weight_decay': 1e-4},
        ]
        if self.finetune_decoder:
            param_groups.append(
                {'params': list(self.decoder.parameters()), 'lr': lr * 0.1, 'weight_decay': 1e-4}
            )
        opt_g = torch.optim.AdamW(param_groups)
        n_adapter = sum(p.numel() for p in self.adapter.parameters())
        n_dec     = sum(p.numel() for p in self.decoder.parameters()) if self.finetune_decoder else 0
        print(f'[PostG] AdamW: adapter lr={lr:.2e} ({n_adapter:,}p)'
              + (f', decoder lr={lr*0.1:.2e} ({n_dec:,}p)' if self.finetune_decoder else ''))

        # GAN mode: discriminator optimizer 추가
        if self.lambda_adv > 0 and self.discriminator is not None:
            d_lr = lr * self.disc_lr_ratio
            opt_d = torch.optim.AdamW(
                self.discriminator.parameters(), lr=d_lr, betas=(0.5, 0.999), weight_decay=1e-4
            )
            n_disc = sum(p.numel() for p in self.discriminator.parameters())
            print(f'[PostG] Discriminator AdamW: lr={d_lr:.2e} ({n_disc:,}p)')
            return [opt_g, opt_d]

        return opt_g

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
