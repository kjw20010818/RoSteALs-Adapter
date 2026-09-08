# ── RoSteALS + SmallAE-v2 Adapter + VQ Correction ─────────────────────────
# Base: CUDA 11.3 + cuDNN 8 + Python 3.8
FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

LABEL maintainer="kjw20010818" \
      description="RoSteALS + SmallAE-v2 adapter with VQ-GAN error correction" \
      version="1.0"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# ── System packages ─────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.8 python3.8-dev python3-pip \
    git wget curl libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf python3.8 /usr/bin/python3 && ln -sf python3 /usr/bin/python

# ── PyTorch (CUDA 11.3) ─────────────────────────────────────────────────────
RUN pip install --upgrade pip && \
    pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 \
        --extra-index-url https://download.pytorch.org/whl/cu113

# ── Python dependencies ──────────────────────────────────────────────────────
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# ── Source code ─────────────────────────────────────────────────────────────
WORKDIR /workspace/RoSteALS-main
COPY . .

# ── Model download (optional: bake-in weights at build time) ─────────────────
# Uncomment to download weights during build (requires internet access):
# RUN bash download_models.sh

# ── Default command: show usage ───────────────────────────────────────────────
CMD ["python", "inference_v2.py", "--help"]
