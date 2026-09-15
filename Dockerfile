# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RoSteALS-Adapter
# 현재 실험과 동일한 환경:
#   PyTorch 2.5.1 + CUDA 12.1 + cuDNN 9 + Python 3.10
#   Lightning 2.4.0, diffusers 0.40.0
#
# 포함 실험:
#   1) FLUX.1 VAE 백본 학습 (512×512, L=64, 단순 resize)  — 원 논문 KL-f8 재현
#   2) VQ-GAN 위 SmallAE-v2 Post-G 어댑터 학습
#   3) FLUX 백본 위 SmallAE 어댑터 학습
#   4) baseline / v2 / VQ보정 / B0제거 평가
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace/rosteals_adapter \
    DATA_DIR=/data \
    WEIGHT_DIR=/weights \
    HF_HOME=/root/.cache/huggingface \
    HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface \
    ROSTEALS_CKPT=/weights/rosteals/epoch=000017-step=000449999.ckpt \
    VQGAN_CKPT=/weights/vq-f4/model.ckpt \
    FLUX_BACKBONE_CKPT=/weights/flux_512/last.ckpt

RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget curl unzip \
        build-essential \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
        imagemagick libmagickwand-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/rosteals_adapter

RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir \
        torch==2.5.1 \
        torchvision==0.20.1 \
        --index-url https://download.pytorch.org/whl/cu121

COPY requirements.txt .
# imagenet-c==0.0.3 은 opencv-python~=3.4 를 요구해서 py3.10에서 소스 빌드가 실패한다.
# 로컬 실험과 같이 OpenCV 4.8 + imagenet-c(--no-deps) 조합을 쓴다.
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --no-deps imagenet-c==0.0.3

COPY cldm/      cldm/
COPY ldm/       ldm/
COPY tools/     tools/
COPY models/    models/
COPY scripts/   scripts/
COPY prep_data/ prep_data/
COPY examples/  examples/
COPY train.py inference.py inference_v2.py download_models.sh ./

# 가중치·데이터·HF 캐시는 이미지에 넣지 않고 마운트한다.
VOLUME ["/data", "/weights", "/root/.cache/huggingface"]

WORKDIR /workspace/rosteals_adapter
CMD ["bash"]
