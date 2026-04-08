# Deploy Redesign: Mount Code + Bake Models

**Date**: 2026-04-08
**Status**: Approved
**Platforms**: Jetson (edge) + Server (GPU)

## Problem

Current deployment bakes source code into Docker images. Any code change requires a full image rebuild + redeploy. Models are downloaded from HuggingFace at runtime, adding startup latency and network dependency.

## Solution

Invert the strategy:
- **Source code**: mount from host via Docker volumes (`src/`, `deploy/`)
- **Models**: bake into Docker image during build (multi-stage, pinned HF revisions)

## Architecture

```
Docker Image = System deps + Python deps + Models (/models/)
Runtime Mount = src/ + deploy/ (from host git repo)
```

### Update Workflow

| Change type | Action | Downtime |
|---|---|---|
| Code change | `git pull` + `docker restart` | ~seconds |
| Dependency change | Rebuild image + redeploy | ~minutes |
| Model change | Rebuild image + redeploy | ~minutes |

## Models Per Platform

### Jetson (mode=turbo)

| File | HF Repo | Path in Image |
|---|---|---|
| `vieneu-tts-v2-turbo.gguf` | `pnnbao-ump/VieNeu-TTS-v2-Turbo-GGUF` | `/models/vieneu-tts-v2-turbo.gguf` |
| `vieneu_decoder.onnx` | `pnnbao-ump/VieNeu-Codec` | `/models/vieneu_decoder.onnx` |
| `vieneu_encoder.onnx` | `pnnbao-ump/VieNeu-Codec` | `/models/vieneu_encoder.onnx` |
| `voices.json` | `pnnbao-ump/VieNeu-TTS-v2-Turbo-GGUF` | `/models/voices.json` |

### Server (mode=turbo_gpu, backend=lmdeploy)

| File | HF Repo | Path in Image |
|---|---|---|
| Full model dir | `pnnbao-ump/VieNeu-TTS-v2-Turbo` | `/models/backbone/` |
| `vieneu_decoder.onnx` | `pnnbao-ump/VieNeu-Codec` | `/models/vieneu_decoder.onnx` |
| `vieneu_encoder.onnx` | `pnnbao-ump/VieNeu-Codec` | `/models/vieneu_encoder.onnx` |
| `voices.json` | (included in backbone snapshot) | `/models/backbone/voices.json` |

## Dockerfile Design

Both platforms use multi-stage builds:

### Stage 1: Model Downloader

Lightweight Python image, installs `huggingface_hub`, downloads models to `/models/` with pinned revisions. Discarded after build.

- **Jetson**: `hf_hub_download()` for individual files (GGUF, ONNX, voices.json)
- **Server**: `snapshot_download()` for full backbone dir + `hf_hub_download()` for ONNX files

### Stage 2: Runtime

Platform-specific base image with system deps and Python packages. Source code is NOT copied.

```dockerfile
COPY --from=model-downloader /models /models

ENV TTS_BACKBONE_PATH="/models/..."
ENV TTS_DECODER_PATH="/models/vieneu_decoder.onnx"
ENV TTS_ENCODER_PATH="/models/vieneu_encoder.onnx"
```

## Volume Mounts

Both platforms mount the same two directories:

```yaml
volumes:
  - ./src:/workspace/src
  - ./deploy:/workspace/deploy
```

No more `hf-cache` volume or `HF_HOME` env var needed.

### Jetson run.sh

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

### Server docker-compose.yml

```yaml
services:
  vieneu-api:
    image: vieneu-tts-server:latest
    ports:
      - "8000:8000"
    environment:
      - TTS_MODE=turbo_gpu
      - TTS_DEVICE=cuda
      - TTS_BACKEND=lmdeploy
      - PYTHONUNBUFFERED=1
    volumes:
      - ./src:/workspace/src
      - ./deploy:/workspace/deploy
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    restart: unless-stopped
```

## Code Changes

### deploy/common.py

**AppConfig** - add 3 new fields:

```python
backbone_path: str = field(default_factory=lambda: os.environ.get("TTS_BACKBONE_PATH", ""))
decoder_path: str = field(default_factory=lambda: os.environ.get("TTS_DECODER_PATH", ""))
encoder_path: str = field(default_factory=lambda: os.environ.get("TTS_ENCODER_PATH", ""))
```

**load_engine()** - pass local paths when set:

```python
if config.backbone_path:
    kwargs["backbone_repo"] = config.backbone_path
if config.decoder_path:
    kwargs["decoder_repo"] = config.decoder_path
if config.encoder_path:
    kwargs["encoder_repo"] = config.encoder_path
```

### No changes to src/vieneu/

The TTS engine already supports local paths:
- `turbo.py`: `os.path.exists(backbone_repo)` -> uses as file path
- `_load_decoder/encoder`: `os.path.exists(decoder_repo)` -> uses as file path
- `_load_voices()`: `Path(backbone_repo).exists()` -> finds `voices.json` in same dir/parent

### Backward Compatibility

Without env vars set (running outside Docker), paths default to empty string -> engine falls back to HF repo defaults. No breaking change.

## Environment Variables

### New

| Variable | Jetson Default | Server Default | Description |
|---|---|---|---|
| `TTS_BACKBONE_PATH` | `/models/vieneu-tts-v2-turbo.gguf` | `/models/backbone` | Local path to backbone model |
| `TTS_DECODER_PATH` | `/models/vieneu_decoder.onnx` | `/models/vieneu_decoder.onnx` | Local path to ONNX decoder |
| `TTS_ENCODER_PATH` | `/models/vieneu_encoder.onnx` | `/models/vieneu_encoder.onnx` | Local path to ONNX encoder |

### Removed from Docker config

| Variable | Reason |
|---|---|
| `HF_HOME` | Models baked in, no HF cache needed |

### Kept (unchanged)

`TTS_MODE`, `TTS_DEVICE`, `TTS_BACKEND`, `TTS_GPU_LAYERS`, `TTS_THREADS`, `PORT`

## Deployment Targets

| Target | Machine | Repo Path | Image Name |
|---|---|---|---|
| Jetson | `100.66.57.32` (mic-711) | `/home/mic-711/workingspace/locnx/VieNeu-TTS` | `vieneu-tts-jetson:latest` |
| Server | `xuanlocserver` | `/media/xuanlocserver/DellEMC12T/workingspace/Q100_project/q100_ai_project/VieNeu-TTS` | `vieneu-tts-server:latest` |
