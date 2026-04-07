"""VieNeu-TTS Jetson Edge API.

Endpoints: /health, /api/voices, /api/tts, /api/tts/stream

Default env vars:
    TTS_MODE=turbo  TTS_DEVICE=cpu  TTS_GPU_LAYERS=0  TTS_THREADS=4  PORT=7860
"""

import numpy as np

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse

from deploy.common import (
    AppConfig, HealthResponse, VoicesResponse, VoiceItem, TTSRequest,
    check_engine, resolve_voice, audio_to_wav_bytes,
    create_lifespan, logger,
)
import deploy.common as state

# ---------------------------------------------------------------------------
# Config — env var defaults tuned for Jetson edge (CPU-only)
# ---------------------------------------------------------------------------

import os
os.environ.setdefault("TTS_MODE", "turbo")
os.environ.setdefault("TTS_DEVICE", "cpu")
os.environ.setdefault("TTS_GPU_LAYERS", "0")
os.environ.setdefault("TTS_THREADS", "4")
os.environ.setdefault("PORT", "7860")

config = AppConfig()

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="VieNeu-TTS Edge API",
    description="Edge API for VieNeu-TTS-v2-Turbo on Jetson Orin NX",
    version="1.0.0",
    lifespan=create_lifespan(config),
)

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health():
    check_engine()
    return HealthResponse(
        status="ok",
        model="VieNeu-TTS-v2-Turbo",
        backend=f"llama-cpp-{config.tts_device}",
        device="jetson-orin-nx",
    )


@app.get("/api/voices", response_model=VoicesResponse)
async def list_voices():
    check_engine()
    preset = state.tts_engine.list_preset_voices()
    return VoicesResponse(voices=[VoiceItem(id=vid, description=desc) for desc, vid in preset])


@app.post("/api/tts")
async def text_to_speech(req: TTSRequest):
    check_engine()
    voice = resolve_voice(req.voice_id)
    try:
        audio = state.tts_engine.infer(
            text=req.text, voice=voice,
            temperature=req.temperature, top_k=req.top_k, show_progress=False,
        )
    except Exception as e:
        logger.exception("Inference error")
        raise HTTPException(status_code=500, detail=str(e))
    return Response(content=audio_to_wav_bytes(audio, state.tts_engine.sample_rate), media_type="audio/wav")


@app.post("/api/tts/stream")
async def text_to_speech_stream(req: TTSRequest):
    check_engine()
    voice = resolve_voice(req.voice_id)

    def generate():
        try:
            for chunk in state.tts_engine.infer_stream(
                text=req.text, voice=voice,
                temperature=req.temperature, top_k=req.top_k,
            ):
                yield chunk.astype(np.float32).tobytes()
        except Exception:
            logger.exception("Streaming inference error")

    return StreamingResponse(
        generate(), media_type="application/octet-stream",
        headers={"X-Sample-Rate": "24000", "X-Channels": "1", "X-Dtype": "float32"},
    )
