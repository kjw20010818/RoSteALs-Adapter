# RoSteALS-Adapter — PyTorch 2.5.1 + CUDA 12.1
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
    FLUX_BACKBONE_CKPT=/weights/flux_512/last.ckpt \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore

RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget curl unzip \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
        imagemagick libmagickwand-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/rosteals_adapter

# 이미 베이스 이미지에 torch가 들어 있다. 버전만 확인하고 CUDA wheel을 고정한다.
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir \
        torch==2.5.1 \
        torchvision==0.20.1 \
        --index-url https://download.pytorch.org/whl/cu121

COPY requirements.txt .

# conda로 깔린 numpy 등과 pip가 충돌하지 않게 --ignore-installed 사용.
# bchlib(추론 BCH용)는 C 소스 빌드라 학습 이미지에서 제외.
RUN pip install --no-cache-dir --prefer-binary --ignore-installed \
        -r requirements.txt

# imagenet-c 0.0.3 은 opencv-python~=3.4 를 요구해서 py3.10+ 에서 실패한다.
RUN pip install --no-cache-dir --no-deps imagenet-c==0.0.3

COPY cldm/      cldm/
COPY ldm/       ldm/
COPY tools/     tools/
COPY models/    models/
COPY scripts/   scripts/
COPY prep_data/ prep_data/
COPY examples/  examples/
COPY train.py inference.py inference_v2.py download_models.sh ./

VOLUME ["/data", "/weights", "/root/.cache/huggingface"]

CMD ["bash"]
