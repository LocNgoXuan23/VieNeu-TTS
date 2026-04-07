FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface \
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
RUN uv sync --no-dev --group gpu

# Install FastAPI deps
RUN uv pip install fastapi "uvicorn[standard]" python-multipart

# Copy only what's needed for serving
COPY src/vieneu/ ./src/vieneu/
COPY src/vieneu_utils/ ./src/vieneu_utils/
COPY deploy/common.py ./deploy/common.py
COPY deploy/server/main.py ./deploy/server/main.py

ENV PYTHONPATH="/workspace/src:/workspace"

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "deploy.server.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
