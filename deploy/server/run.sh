#!/bin/bash
# VieNeu-TTS GPU Server — Docker run script
# Usage:
#   bash run.sh              # Start server
#   bash run.sh stop         # Stop server

set -e

IMAGE="vieneu-tts:v1.0.0-turbogpu-cuda12.8"
CONTAINER="vieneu-tts-server"
PORT="${PORT:-8000}"

case "${1:-start}" in
  stop)
    echo "Stopping ${CONTAINER}..."
    docker stop "$CONTAINER" 2>/dev/null && docker rm "$CONTAINER" 2>/dev/null
    echo "Done."
    exit 0
    ;;
  start|*)
    echo "Starting GPU server (lmdeploy, port ${PORT})..."
    ;;
esac

# Stop existing container if running
docker stop "$CONTAINER" 2>/dev/null && docker rm "$CONTAINER" 2>/dev/null || true

docker run -d \
  --name "$CONTAINER" \
  --gpus all \
  -p "${PORT}:${PORT}" \
  -e TTS_MODE="turbo_gpu" \
  -e TTS_DEVICE="cuda" \
  -e TTS_BACKEND="lmdeploy" \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -v "$(pwd)/src:/workspace/src" \
  -v "$(pwd)/deploy:/workspace/deploy" \
  --restart unless-stopped \
  "$IMAGE"

echo "Container started. Waiting for health..."
for i in $(seq 1 60); do
  if curl -sf http://localhost:${PORT}/health > /dev/null 2>&1; then
    echo "Ready! $(curl -sf http://localhost:${PORT}/health)"
    exit 0
  fi
  sleep 1
done
echo "Timeout waiting for health. Check logs: docker logs $CONTAINER"
