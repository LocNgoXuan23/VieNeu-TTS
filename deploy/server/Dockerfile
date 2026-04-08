# =============================================================================
# Stage 1: Download models from HuggingFace (discarded after build)
# =============================================================================
FROM python:3.12-slim AS model-downloader

RUN pip install --no-cache-dir huggingface_hub

# Download server turbo_gpu models to /models/
# Pin revisions for reproducible builds
ARG BACKBONE_REVISION=main
ARG CODEC_REVISION=main

RUN python3 -c "\
from huggingface_hub import snapshot_download, hf_hub_download; \
snapshot_download('pnnbao-ump/VieNeu-TTS-v2-Turbo', \
    local_dir='/models/backbone', revision='${BACKBONE_REVISION}'); \
hf_hub_download('pnnbao-ump/VieNeu-Codec', 'vieneu_decoder.onnx', \
    local_dir='/models', revision='${CODEC_REVISION}'); \
hf_hub_download('pnnbao-ump/VieNeu-Codec', 'vieneu_encoder.onnx', \
    local_dir='/models', revision='${CODEC_REVISION}')"

# =============================================================================
# Stage 2: Runtime
# =============================================================================
FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PHONEMIZER_ESPEAK_LIBRARY=/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1 \
    PATH="/workspace/.venv/bin:$PATH"

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    build-essential \
    espeak-ng \
    libespeak-ng1 \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /workspace

# Copy project metadata for dependency install (cache-friendly)
COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --group gpu --no-install-project

# Install FastAPI deps
RUN uv pip install fastapi "uvicorn[standard]" python-multipart

# Copy models from downloader stage (baked into image)
COPY --from=model-downloader /models /models

# Source code is NOT copied — mounted at runtime via:
#   -v ./src:/workspace/src
#   -v ./deploy:/workspace/deploy

# Model paths (used by deploy/common.py AppConfig)
ENV TTS_BACKBONE_PATH="/models/backbone" \
    TTS_DECODER_PATH="/models/vieneu_decoder.onnx" \
    TTS_ENCODER_PATH="/models/vieneu_encoder.onnx" \
    PYTHONPATH="/workspace/src:/workspace"

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "deploy.server.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
