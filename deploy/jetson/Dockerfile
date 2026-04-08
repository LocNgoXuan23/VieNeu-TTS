# =============================================================================
# Stage 1: Download models from HuggingFace (discarded after build)
# =============================================================================
FROM python:3.10-slim AS model-downloader

RUN pip install --no-cache-dir huggingface_hub

# Download Jetson turbo models to /models/
# Pin revisions for reproducible builds
ARG GGUF_REVISION=main
ARG CODEC_REVISION=main

RUN python3 -c "\
from huggingface_hub import hf_hub_download; \
hf_hub_download('pnnbao-ump/VieNeu-TTS-v2-Turbo-GGUF', 'vieneu-tts-v2-turbo.gguf', \
    local_dir='/models', revision='${GGUF_REVISION}'); \
hf_hub_download('pnnbao-ump/VieNeu-TTS-v2-Turbo-GGUF', 'voices.json', \
    local_dir='/models', revision='${GGUF_REVISION}'); \
hf_hub_download('pnnbao-ump/VieNeu-Codec', 'vieneu_decoder_int8.onnx', \
    local_dir='/models', revision='${CODEC_REVISION}'); \
hf_hub_download('pnnbao-ump/VieNeu-Codec', 'vieneu_encoder.onnx', \
    local_dir='/models', revision='${CODEC_REVISION}')"

# =============================================================================
# Stage 2: Runtime
# =============================================================================
FROM nvcr.io/nvidia/l4t-base:r36.2.0

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PHONEMIZER_ESPEAK_LIBRARY=/usr/lib/aarch64-linux-gnu/libespeak-ng.so.1 \
    CUDA_HOME=/usr/local/cuda-12.2 \
    PATH="/usr/local/cuda-12.2/bin:$PATH"

# System deps: Python 3.10, build tools, espeak-ng, cmake, CUDA compiler
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    cmake \
    ninja-build \
    cuda-nvcc-12-2 \
    cuda-cudart-dev-12-2 \
    libcublas-dev-12-2 \
    espeak-ng \
    libespeak-ng1 \
    libsndfile1 \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Compile llama-cpp-python with CUDA support for Jetson Orin (SM 8.7)
RUN ln -sf /usr/local/cuda-12.2/targets/aarch64-linux/lib/stubs/libcuda.so /usr/lib/aarch64-linux-gnu/libcuda.so.1 \
    && ln -sf /usr/lib/aarch64-linux-gnu/libcuda.so.1 /usr/lib/aarch64-linux-gnu/libcuda.so \
    && ldconfig
ENV CMAKE_ARGS="-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87"
ENV CUDACXX=/usr/local/cuda-12.2/bin/nvcc
RUN pip3 install --no-cache-dir llama-cpp-python==0.3.16
RUN rm -f /usr/lib/aarch64-linux-gnu/libcuda.so.1 /usr/lib/aarch64-linux-gnu/libcuda.so && ldconfig

# Install Python deps (pinned versions)
RUN pip3 install --no-cache-dir \
    "onnxruntime==1.20.1" \
    "sea-g2p==0.7.5" \
    "perth>=0.2.0" \
    "librosa==0.11.0" \
    "huggingface-hub==0.30.2" \
    "numpy==1.26.4" \
    "soundfile==0.13.1" \
    "PyYAML==6.0.2" \
    "requests==2.32.3" \
    "tqdm==4.67.1"

# Install FastAPI deps (pinned)
RUN pip3 install --no-cache-dir \
    "fastapi==0.115.12" \
    "uvicorn[standard]==0.34.2" \
    "python-multipart==0.0.20"

WORKDIR /workspace

# Copy models from downloader stage (baked into image)
COPY --from=model-downloader /models /models

# Source code is NOT copied — mounted at runtime via:
#   -v ./src:/workspace/src
#   -v ./deploy:/workspace/deploy

# Model paths (used by deploy/common.py AppConfig)
ENV TTS_BACKBONE_PATH="/models/vieneu-tts-v2-turbo.gguf" \
    TTS_DECODER_PATH="/models/vieneu_decoder_int8.onnx" \
    TTS_ENCODER_PATH="/models/vieneu_encoder.onnx" \
    PYTHONPATH="/workspace/src:/workspace"

EXPOSE 7860

CMD ["python3", "-m", "uvicorn", "deploy.jetson.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
