FROM nvcr.io/nvidia/l4t-base:r36.2.0

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface \
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
# Symlink CUDA driver stubs for build-time linking (real libcuda.so injected at runtime)
RUN ln -sf /usr/local/cuda-12.2/targets/aarch64-linux/lib/stubs/libcuda.so /usr/lib/aarch64-linux-gnu/libcuda.so.1 \
    && ln -sf /usr/lib/aarch64-linux-gnu/libcuda.so.1 /usr/lib/aarch64-linux-gnu/libcuda.so \
    && ldconfig
ENV CMAKE_ARGS="-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87"
ENV CUDACXX=/usr/local/cuda-12.2/bin/nvcc
RUN pip3 install --no-cache-dir llama-cpp-python==0.3.16
# Remove stubs symlink (real driver mounted at runtime)
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

# Copy only what's needed for serving
COPY src/vieneu/ ./src/vieneu/
COPY src/vieneu_utils/ ./src/vieneu_utils/
COPY deploy/common.py ./deploy/common.py
COPY deploy/jetson/main.py ./deploy/jetson/main.py

ENV PYTHONPATH="/workspace/src:/workspace"

EXPOSE 7860

CMD ["python3", "-m", "uvicorn", "deploy.jetson.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
