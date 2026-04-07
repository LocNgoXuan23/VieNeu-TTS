"""VieNeu-TTS GPU Server API.

Endpoints: /health, /api/voices, /api/tts, /api/tts/stream, /api/tts/batch, /api/clone

Default env vars:
    TTS_MODE=turbo_gpu  TTS_DEVICE=cuda  TTS_BACKEND=lmdeploy  PORT=8000
"""

import base64
import tempfile
import numpy as np
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import Response, StreamingResponse

from deploy.common import (
    AppConfig, HealthResponse, VoicesResponse, VoiceItem,
    TTSRequest, BatchTTSRequest, BatchTTSResponse, BatchResultItem,
    CloneResponse,
    tts_engine, check_engine, resolve_voice, audio_to_wav_bytes,
    create_lifespan, logger,
)
import deploy.common as state

# ---------------------------------------------------------------------------
# Config — env var defaults tuned for GPU server
# ---------------------------------------------------------------------------

import os
os.environ.setdefault("TTS_MODE", "turbo_gpu")
os.environ.setdefault("TTS_DEVICE", "cuda")
os.environ.setdefault("TTS_BACKEND", "lmdeploy")
os.environ.setdefault("PORT", "8000")

config = AppConfig()

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="VieNeu-TTS API",
    description="Production GPU API for VieNeu-TTS-v2-Turbo",
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
        backend=config.tts_mode + "/" + (config.tts_backend or config.tts_device),
        device=config.tts_device,
    )


@app.get("/api/voices", response_model=VoicesResponse)
async def list_voices():
    check_engine()
    preset = state.tts_engine.list_preset_voices()
    return VoicesResponse(voices=[VoiceItem(id=vid, description=desc) for desc, vid in preset])


@app.post("/api/tts")
async def text_to_speech(req: TTSRequest):
    check_engine()
    voice = resolve_voice(req.voice_id, req.voice_embedding)
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
    voice = resolve_voice(req.voice_id, req.voice_embedding)

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


@app.post("/api/tts/batch", response_model=BatchTTSResponse)
async def text_to_speech_batch(req: BatchTTSRequest):
    check_engine()
    voice = resolve_voice(req.voice_id, req.voice_embedding)
    try:
        audios = state.tts_engine.infer_batch(
            texts=req.texts, voice=voice,
            temperature=req.temperature, top_k=req.top_k,
        )
    except Exception as e:
        logger.exception("Batch inference error")
        raise HTTPException(status_code=500, detail=str(e))
    results = []
    for i, audio in enumerate(audios):
        wav_bytes = audio_to_wav_bytes(audio, state.tts_engine.sample_rate)
        results.append(BatchResultItem(index=i, audio_base64=base64.b64encode(wav_bytes).decode()))
    return BatchTTSResponse(results=results)


@app.post("/api/clone", response_model=CloneResponse)
async def clone_voice(file: UploadFile = File(...)):
    check_engine()
    allowed = {".wav", ".mp3", ".flac"}
    ext = Path(file.filename).suffix.lower() if file.filename else ""
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported format '{ext}'. Use: {allowed}")
    try:
        content = await file.read()
        with tempfile.NamedTemporaryFile(suffix=ext, delete=True) as tmp:
            tmp.write(content)
            tmp.flush()
            embedding = state.tts_engine.encode_reference(tmp.name)
        emb_list = np.array(embedding, dtype=np.float32).flatten().tolist()
    except Exception as e:
        logger.exception("Voice cloning error")
        raise HTTPException(status_code=500, detail=str(e))
    return CloneResponse(voice_embedding=emb_list, message="Voice encoded successfully")
