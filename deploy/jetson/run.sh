#!/bin/bash
# VieNeu-TTS Jetson Edge — Docker run script
# Usage:
#   bash run.sh              # CPU mode (default, recommended)
#   bash run.sh gpu          # GPU mode (all layers on GPU)
#   bash run.sh stop         # Stop container

set -e

IMAGE="vieneu-tts-jetson:latest"
CONTAINER="vieneu-tts"

case "${1:-cpu}" in
  stop)
    echo "Stopping ${CONTAINER}..."
    docker stop "$CONTAINER" 2>/dev/null && docker rm "$CONTAINER" 2>/dev/null
    echo "Done."
    exit 0
    ;;
  gpu)
    TTS_DEVICE="cuda"
    TTS_GPU_LAYERS="-1"
    TTS_THREADS="0"
    echo "Starting in GPU mode (all layers on GPU)..."
    ;;
  cpu|*)
    TTS_DEVICE="cpu"
    TTS_GPU_LAYERS="0"
    TTS_THREADS="4"
    echo "Starting in CPU mode (GPU free, 4 threads)..."
    ;;
esac

# Stop existing container if running
docker stop "$CONTAINER" 2>/dev/null && docker rm "$CONTAINER" 2>/dev/null || true

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

echo "Container started. Waiting for health..."
for i in $(seq 1 60); do
  if curl -sf http://localhost:7860/health > /dev/null 2>&1; then
    echo "Ready! $(curl -sf http://localhost:7860/health)"
    exit 0
  fi
  sleep 1
done
echo "Timeout waiting for health. Check logs: docker logs $CONTAINER"
