"""Shared code for VieNeu-TTS API deployments (server + jetson)."""

import io
import os
import logging
import numpy as np
import soundfile as sf
from contextlib import asynccontextmanager
from typing import Optional, List
from dataclasses import dataclass, field

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger("vieneu-api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


# ---------------------------------------------------------------------------
# Configuration (all via env vars)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=3000)
    voice_id: Optional[str] = None
    voice_embedding: Optional[List[float]] = None
    temperature: float = Field(default=0.4, ge=0.0, le=2.0)
    top_k: int = Field(default=50, ge=1, le=200)


class BatchTTSRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=10)
    voice_id: Optional[str] = None
    voice_embedding: Optional[List[float]] = None
    temperature: float = Field(default=0.4, ge=0.0, le=2.0)
    top_k: int = Field(default=50, ge=1, le=200)


class BatchResultItem(BaseModel):
    index: int
    audio_base64: str


class BatchTTSResponse(BaseModel):
    results: List[BatchResultItem]


class CloneResponse(BaseModel):
    voice_embedding: List[float]
    message: str


class VoiceItem(BaseModel):
    id: str
    description: str


class VoicesResponse(BaseModel):
    voices: List[VoiceItem]


class HealthResponse(BaseModel):
    status: str
    model: str
    backend: str
    device: str


# ---------------------------------------------------------------------------
# Global engine reference
# ---------------------------------------------------------------------------

tts_engine = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_voice(voice_id: Optional[str], voice_embedding: Optional[List[float]] = None):
    if voice_embedding is not None:
        return {"codes": voice_embedding, "text": ""}
    if voice_id is not None:
        try:
            return tts_engine.get_preset_voice(voice_id)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Voice '{voice_id}' not found.")
    return None


def audio_to_wav_bytes(audio: np.ndarray, sample_rate: int = 24000) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV")
    buf.seek(0)
    return buf.read()


def check_engine():
    if tts_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")


# ---------------------------------------------------------------------------
# Engine loader
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

def create_lifespan(config: AppConfig):
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global tts_engine
        logger.info(f"Loading VieNeu-TTS (mode={config.tts_mode}, device={config.tts_device})...")
        tts_engine = load_engine(config)
        logger.info("Model loaded successfully.")
        yield
        logger.info("Shutting down...")
        if tts_engine is not None:
            tts_engine.close()
            tts_engine = None
    return lifespan
