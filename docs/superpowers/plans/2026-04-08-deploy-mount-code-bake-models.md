# Deploy Redesign: Mount Code + Bake Models — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure Docker deployments so source code is mounted from host (fast iteration) and models are baked into images (self-contained, no runtime HF downloads).

**Architecture:** Multi-stage Docker builds — stage 1 downloads models from HuggingFace with pinned revisions, stage 2 is the runtime image with deps + models. Source code (`src/`, `deploy/`) is mounted at runtime via Docker volumes.

**Tech Stack:** Docker multi-stage builds, huggingface_hub, FastAPI/Uvicorn, NVIDIA Container Runtime

**Spec:** `docs/superpowers/specs/2026-04-08-deploy-mount-code-bake-models-design.md`

---

## File Structure

| Action | File | Responsibility |
|---|---|---|
| Modify | `deploy/common.py:24-31` | Add `backbone_path`, `decoder_path`, `encoder_path` to AppConfig |
| Modify | `deploy/common.py:123-140` | Update `load_engine()` to pass local model paths |
| Rewrite | `deploy/jetson/Dockerfile` | Multi-stage: download models + runtime (no COPY code) |
| Modify | `deploy/jetson/run.sh:37-48` | Add `-v` mounts for src/deploy, remove hf-cache volume |
| Modify | `deploy/jetson/docker-compose.yml` | Add volume mounts, remove hf-cache |
| Rewrite | `deploy/server/Dockerfile` | Multi-stage: download models + runtime (no COPY code) |
| Modify | `deploy/server/docker-compose.yml` | Add volume mounts, remove hf-cache |

No new files. No changes to `src/vieneu/` (engine already supports local paths).

---

### Task 1: Look up HuggingFace revision hashes

We need pinned commit hashes for reproducible builds. These must be looked up before writing Dockerfiles.

**Repos to pin:**
- `pnnbao-ump/VieNeu-TTS-v2-Turbo-GGUF` (Jetson backbone + voices.json)
- `pnnbao-ump/VieNeu-TTS-v2-Turbo` (Server backbone)
- `pnnbao-ump/VieNeu-Codec` (decoder + encoder ONNX for both)

- [ ] **Step 1: Get current commit hashes from HuggingFace**

Run for each repo:

```bash
# Install huggingface_hub if not available
pip install huggingface_hub

python3 -c "
from huggingface_hub import model_info
repos = [
    'pnnbao-ump/VieNeu-TTS-v2-Turbo-GGUF',
    'pnnbao-ump/VieNeu-TTS-v2-Turbo',
    'pnnbao-ump/VieNeu-Codec',
]
for repo in repos:
    info = model_info(repo)
    print(f'{repo}: {info.sha}')
"
```

Expected: Three lines with repo names and 40-char commit hashes. Save these — they go into the Dockerfiles in Tasks 3 and 5.

- [ ] **Step 2: Record hashes for use in later tasks**

Note the hashes somewhere accessible. Format:

```
GGUF_REVISION=<hash from VieNeu-TTS-v2-Turbo-GGUF>
BACKBONE_REVISION=<hash from VieNeu-TTS-v2-Turbo>
CODEC_REVISION=<hash from VieNeu-Codec>
```

No commit for this task — it's research.

---

### Task 2: Update deploy/common.py — model path config

**Files:**
- Modify: `deploy/common.py:24-31` (AppConfig dataclass)
- Modify: `deploy/common.py:123-140` (load_engine function)

- [ ] **Step 1: Add model path fields to AppConfig**

In `deploy/common.py`, add three fields after the existing `workers` field (line 32):

```python
@dataclass
class AppConfig:
    tts_mode: str = field(default_factory=lambda: os.environ.get("TTS_MODE", "turbo"))
    tts_device: str = field(default_factory=lambda: os.environ.get("TTS_DEVICE", "cuda"))
    tts_backend: str = field(default_factory=lambda: os.environ.get("TTS_BACKEND", ""))
    decoder_filename: str = field(default_factory=lambda: os.environ.get("TTS_DECODER", "vieneu_decoder_int8.onnx"))
    n_gpu_layers: int = field(default_factory=lambda: int(os.environ.get("TTS_GPU_LAYERS", "-1")))
    n_threads: int = field(default_factory=lambda: int(os.environ.get("TTS_THREADS", "0")))
    port: int = field(default_factory=lambda: int(os.environ.get("PORT", "7860")))
    workers: int = field(default_factory=lambda: int(os.environ.get("WORKERS", "1")))
    backbone_path: str = field(default_factory=lambda: os.environ.get("TTS_BACKBONE_PATH", ""))
    decoder_path: str = field(default_factory=lambda: os.environ.get("TTS_DECODER_PATH", ""))
    encoder_path: str = field(default_factory=lambda: os.environ.get("TTS_ENCODER_PATH", ""))
```

- [ ] **Step 2: Update load_engine() to pass local paths**

Replace the existing `load_engine` function with:

```python
def load_engine(config: AppConfig):
    from vieneu import Vieneu

    kwargs = {}

    if config.tts_backend:
        kwargs["backend"] = config.tts_backend

    if config.backbone_path:
        kwargs["backbone_repo"] = config.backbone_path
    if config.decoder_path:
        kwargs["decoder_repo"] = config.decoder_path
    if config.encoder_path:
        kwargs["encoder_repo"] = config.encoder_path

    if config.tts_mode in ("turbo",):
        kwargs["decoder_filename"] = config.decoder_filename
        if config.n_gpu_layers != -1:
            kwargs["n_gpu_layers"] = config.n_gpu_layers
        if config.n_threads > 0:
            kwargs["n_threads"] = config.n_threads
            kwargs["n_threads_batch"] = config.n_threads

    engine = Vieneu(mode=config.tts_mode, device=config.tts_device, **kwargs)
    return engine
```

Key change: the three `if config.*_path` blocks are added before the existing turbo-specific logic. When `decoder_repo` is a local file path, `decoder_filename` is ignored by the engine (the engine checks `os.path.exists(decoder_repo)` first).

- [ ] **Step 3: Verify backward compatibility**

```bash
cd /media/xuanlocserver/DellEMC12T/workingspace/Q100_project/q100_ai_project/VieNeu-TTS
python3 -c "
import os
# Ensure no TTS_*_PATH env vars are set (simulating non-Docker usage)
for k in ['TTS_BACKBONE_PATH', 'TTS_DECODER_PATH', 'TTS_ENCODER_PATH']:
    os.environ.pop(k, None)
from deploy.common import AppConfig
c = AppConfig()
assert c.backbone_path == '', f'Expected empty, got {c.backbone_path!r}'
assert c.decoder_path == '', f'Expected empty, got {c.decoder_path!r}'
assert c.encoder_path == '', f'Expected empty, got {c.encoder_path!r}'
print('OK: backward compatible — empty paths default to HF repos')
"
```

Expected: `OK: backward compatible — empty paths default to HF repos`

- [ ] **Step 4: Verify env var override works**

```bash
cd /media/xuanlocserver/DellEMC12T/workingspace/Q100_project/q100_ai_project/VieNeu-TTS
TTS_BACKBONE_PATH="/models/test.gguf" TTS_DECODER_PATH="/models/dec.onnx" TTS_ENCODER_PATH="/models/enc.onnx" \
python3 -c "
from deploy.common import AppConfig
c = AppConfig()
assert c.backbone_path == '/models/test.gguf', f'Got {c.backbone_path!r}'
assert c.decoder_path == '/models/dec.onnx', f'Got {c.decoder_path!r}'
assert c.encoder_path == '/models/enc.onnx', f'Got {c.encoder_path!r}'
print('OK: env vars override correctly')
"
```

Expected: `OK: env vars override correctly`

- [ ] **Step 5: Commit**

```bash
git add deploy/common.py
git commit -m "feat(deploy): add model path env vars to AppConfig and load_engine"
```

---

### Task 3: Rewrite Jetson Dockerfile — multi-stage with baked models

**Files:**
- Rewrite: `deploy/jetson/Dockerfile`

**Important:** The Jetson turbo mode uses `vieneu_decoder_int8.onnx` (int8 quantized), not `vieneu_decoder.onnx`. This is set by the `TTS_DECODER` env var default in common.py.

- [ ] **Step 1: Write the new multi-stage Dockerfile**

Replace `deploy/jetson/Dockerfile` entirely with:

```dockerfile
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
```

- [ ] **Step 2: Verify Dockerfile syntax**

```bash
cd /media/xuanlocserver/DellEMC12T/workingspace/Q100_project/q100_ai_project/VieNeu-TTS
docker build --check -f deploy/jetson/Dockerfile . 2>&1 || echo "Note: --check may not be available on older Docker versions, syntax will be validated during build"
```

- [ ] **Step 3: Commit**

```bash
git add deploy/jetson/Dockerfile
git commit -m "feat(deploy/jetson): multi-stage Dockerfile with baked models"
```

---

### Task 4: Update Jetson run.sh and docker-compose.yml

**Files:**
- Modify: `deploy/jetson/run.sh:37-48`
- Modify: `deploy/jetson/docker-compose.yml`

- [ ] **Step 1: Update run.sh — add volume mounts, remove hf-cache**

Replace the `docker run` block (lines 37-48) in `deploy/jetson/run.sh` with:

```bash
docker run -d \
  --name "$CONTAINER" \
  --runtime nvidia \
  --network host \
  -e TTS_DEVICE="$TTS_DEVICE" \
  -e TTS_GPU_LAYERS="$TTS_GPU_LAYERS" \
  -e TTS_THREADS="$TTS_THREADS" \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -v "$(pwd)/src:/workspace/src" \
  -v "$(pwd)/deploy:/workspace/deploy" \
  --restart unless-stopped \
  "$IMAGE"
```

Changes vs current:
- Removed: `-e HF_HOME=/root/.cache/huggingface`
- Removed: `-v vieneu-hf-cache:/root/.cache/huggingface`
- Added: `-v "$(pwd)/src:/workspace/src"`
- Added: `-v "$(pwd)/deploy:/workspace/deploy"`

- [ ] **Step 2: Update docker-compose.yml**

Replace `deploy/jetson/docker-compose.yml` entirely with:

```yaml
# NOTE: On Jetson Orin NX, docker-compose networking may fail due to
# missing iptable_raw kernel module. Use run.sh instead:
#
#   bash deploy/jetson/run.sh          # CPU mode (default)
#   bash deploy/jetson/run.sh gpu      # GPU mode
#   bash deploy/jetson/run.sh stop     # Stop
#
# This file is kept as a reference for the service configuration.

services:
  vieneu-tts:
    build:
      context: ../..
      dockerfile: deploy/jetson/Dockerfile
    network_mode: host
    runtime: nvidia
    environment:
      - TTS_DEVICE=${TTS_DEVICE:-cpu}
      - TTS_GPU_LAYERS=${TTS_GPU_LAYERS:-0}
      - TTS_THREADS=${TTS_THREADS:-4}
      - PYTHONUNBUFFERED=1
      - NVIDIA_VISIBLE_DEVICES=all
    volumes:
      - ../../src:/workspace/src
      - ../../deploy:/workspace/deploy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:7860/health"]
      interval: 30s
      timeout: 10s
      retries: 5
      start_period: 180s
    restart: unless-stopped
```

Changes vs current:
- Removed: `HF_HOME` env var
- Removed: `hf-cache` named volume
- Removed: `volumes:` section at bottom (no more named volumes)
- Added: bind mounts for `../../src` and `../../deploy`

- [ ] **Step 3: Commit**

```bash
git add deploy/jetson/run.sh deploy/jetson/docker-compose.yml
git commit -m "feat(deploy/jetson): mount code volumes, remove hf-cache"
```

---

### Task 5: Rewrite Server Dockerfile — multi-stage with baked models

**Files:**
- Rewrite: `deploy/server/Dockerfile`

**Important:** Server turbo_gpu mode uses `vieneu_decoder.onnx` (not int8). The backbone is a full HF model directory (for lmdeploy), downloaded with `snapshot_download`.

- [ ] **Step 1: Write the new multi-stage Dockerfile**

Replace `deploy/server/Dockerfile` entirely with:

```dockerfile
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
RUN uv sync --no-dev --group gpu

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
```

- [ ] **Step 2: Commit**

```bash
git add deploy/server/Dockerfile
git commit -m "feat(deploy/server): multi-stage Dockerfile with baked models"
```

---

### Task 6: Update Server docker-compose.yml

**Files:**
- Modify: `deploy/server/docker-compose.yml`

- [ ] **Step 1: Replace docker-compose.yml**

Replace `deploy/server/docker-compose.yml` entirely with:

```yaml
services:
  vieneu-api:
    build:
      context: ../..
      dockerfile: deploy/server/Dockerfile
    ports:
      - "8000:8000"
    environment:
      - TTS_MODE=turbo_gpu
      - TTS_DEVICE=cuda
      - TTS_BACKEND=lmdeploy
      - PYTHONUNBUFFERED=1
    volumes:
      - ../../src:/workspace/src
      - ../../deploy:/workspace/deploy
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 5
      start_period: 120s
    restart: unless-stopped
```

Changes vs current:
- Removed: `HF_HOME` env var
- Removed: `hf-cache` named volume and bottom `volumes:` section
- Added: bind mounts for `../../src` and `../../deploy`

- [ ] **Step 2: Commit**

```bash
git add deploy/server/docker-compose.yml
git commit -m "feat(deploy/server): mount code volumes, remove hf-cache"
```

---

### Task 7: Build and test Jetson image locally

This task can only run on the Jetson device or a machine with the l4t base image. If building on the Jetson itself:

- [ ] **Step 1: Build the image on Jetson**

```bash
cd /home/mic-711/workingspace/locnx/VieNeu-TTS
git pull origin main

docker build -f deploy/jetson/Dockerfile -t vieneu-tts-jetson:latest .
```

Expected: Build completes successfully. Stage 1 downloads ~500MB of models, Stage 2 installs deps. Total image size should be similar to before.

- [ ] **Step 2: Run and test**

```bash
bash deploy/jetson/run.sh cpu
```

Expected: Container starts, health check passes within 60s.

- [ ] **Step 3: Verify endpoints**

```bash
curl http://localhost:7860/health
curl http://localhost:7860/api/voices
curl -X POST http://localhost:7860/api/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Xin chao", "voice": "female_south"}' -o /tmp/test.wav
```

Expected: Health returns OK, voices returns list, TTS generates valid WAV file.

- [ ] **Step 4: Verify code mount works**

```bash
# Make a trivial change to verify mount is live
docker exec vieneu-tts ls /workspace/src/vieneu/factory.py
docker exec vieneu-tts ls /workspace/deploy/common.py
```

Expected: Both files exist inside the container (mounted from host).

---

### Task 8: Build and test Server image locally

- [ ] **Step 1: Build the image on xuanlocserver**

```bash
cd /media/xuanlocserver/DellEMC12T/workingspace/Q100_project/q100_ai_project/VieNeu-TTS

docker build -f deploy/server/Dockerfile -t vieneu-tts-server:latest .
```

Expected: Build completes. Stage 1 downloads backbone model (~1-2GB) + ONNX files.

- [ ] **Step 2: Run with docker-compose**

```bash
cd deploy/server
docker compose up -d
```

Expected: Container starts, health check passes.

- [ ] **Step 3: Verify endpoints**

```bash
curl http://localhost:8000/health
curl http://localhost:8000/api/voices
curl -X POST http://localhost:8000/api/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Xin chao", "voice": "female_south"}' -o /tmp/test_server.wav
```

Expected: All endpoints respond correctly.

- [ ] **Step 4: Verify code mount works**

```bash
docker exec $(docker ps -qf "ancestor=vieneu-tts-server:latest") ls /workspace/src/vieneu/factory.py
docker exec $(docker ps -qf "ancestor=vieneu-tts-server:latest") ls /workspace/deploy/common.py
```

Expected: Both files exist (mounted from host).

- [ ] **Step 5: Verify hot-reload workflow**

```bash
# Restart container — should pick up any code changes from host
docker compose restart
curl http://localhost:8000/health
```

Expected: Health returns OK after restart — proves code is read from mount, not baked in.
