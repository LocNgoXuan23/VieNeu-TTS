import os
import numpy as np
import logging
from typing import Optional, List, Dict, Any, Generator
from .base import BaseVieneuTTS
from vieneu_utils.phonemize_text import phonemize_text
from vieneu_utils.core_utils import split_into_chunks_v2, get_silence_duration_v2

logger = logging.getLogger("Vieneu.Turbo")

TURBO_VOICE_ID_MAP = {
    "Xuân Vĩnh (Nam - Miền Nam)": 3,
    "Đoan Trang (Nữ - Miền Bắc)": 0,
    "Thục Đoan (Nữ - Miền Nam)": 1,
    "Phạm Tuyên (Nam - Miền Bắc)": 2,
}

class TurboVieNeuTTS(BaseVieneuTTS):
    def __init__(
        self,
        backbone_repo: str = "pnnbao-ump/VieNeu-TTS-v2-Turbo-GGUF",
        backbone_filename: str = "vieneu-tts-v2-turbo.gguf",
        decoder_repo: str = "pnnbao-ump/VieNeu-Codec",
        decoder_filename: str = "vieneu_decoder.onnx",
        device: str = "cpu",
        hf_token: Optional[str] = None,
    ):
        super().__init__()
        self.backbone = None
        self.decoder_sess = None
        self._is_onnx_codec = True
        self.max_context = 4096
        self._load_backbone(backbone_repo, backbone_filename, device, hf_token)
        self._load_decoder(decoder_repo, decoder_filename, device, hf_token)
        self._load_voices()

    def _load_voices(self) -> None:
        self._preset_voices = TURBO_VOICE_ID_MAP.copy()
        if self._preset_voices:
            self._default_voice = next(iter(self._preset_voices))

    def list_preset_voices(self) -> List[str]:
        return list(self._preset_voices.keys())

    def get_preset_voice(self, name: str) -> Dict[str, Any]:
        if name not in self._preset_voices:
            raise ValueError(f"Voice '{name}' not found.")
        voice_id = self._preset_voices[name]
        embedding = None
        if voice_id == -1:
            embedding = self._speaker_embeddings.get(name)
        return {
            "name": name,
            "voice_id": voice_id,
            "codes": voice_id if voice_id >= 0 else embedding,
            "text": ""
        }

    def _load_backbone(self, backbone_repo, backbone_filename, device, hf_token=None):
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError("llama-cpp-python is required for Turbo mode.")

        if os.path.exists(backbone_repo):
            model_path = backbone_repo
        else:
            from huggingface_hub import hf_hub_download
            try:
                model_path = hf_hub_download(
                    repo_id=backbone_repo, filename=backbone_filename, token=hf_token
                )
            except Exception:
                if os.path.exists(backbone_filename):
                    model_path = backbone_filename
                else:
                    raise FileNotFoundError(f"Neither repo '{backbone_repo}' nor '{backbone_filename}' found.")

        self.backbone = Llama(
            model_path=model_path,
            n_ctx=self.max_context,
            n_gpu_layers=-1 if device in ("gpu", "cuda") else 0,
            mlock=True,
            flash_attn=device in ("gpu", "cuda"),
            verbose=False,
        )

    def _load_decoder(self, decoder_repo, decoder_filename, device, hf_token=None):
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("onnxruntime is required for Turbo mode.")

        if os.path.exists(decoder_repo):
            decoder_path = decoder_repo
        else:
            from huggingface_hub import hf_hub_download
            try:
                decoder_path = hf_hub_download(
                    repo_id=decoder_repo, filename=decoder_filename, token=hf_token
                )
            except Exception:
                if os.path.exists(decoder_filename):
                    decoder_path = decoder_filename
                else:
                    raise FileNotFoundError(f"Neither repo '{decoder_repo}' nor '{decoder_filename}' found.")

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device in ("gpu", "cuda") else ["CPUExecutionProvider"]
        self.decoder_sess = ort.InferenceSession(decoder_path, providers=providers)

    def _get_voice_params(self, ref_codes: Any) -> tuple:
        if ref_codes is None:
            return -1, np.zeros((1, 128), dtype=np.float32)
        if isinstance(ref_codes, int):
            return ref_codes, np.zeros((1, 128), dtype=np.float32)
        if isinstance(ref_codes, dict):
            voice_id = ref_codes.get("voice_id", -1)
            embedding = ref_codes.get("codes")
            if embedding is None or isinstance(embedding, (int, float)):
                embedding = np.zeros((1, 128), dtype=np.float32)
            elif isinstance(embedding, list):
                embedding = np.array(embedding, dtype=np.float32)
            if isinstance(embedding, np.ndarray) and embedding.ndim == 1:
                embedding = embedding[None, :]
            return voice_id, embedding
        if isinstance(ref_codes, np.ndarray):
            if ref_codes.ndim == 1:
                ref_codes = ref_codes[None, :]
            return -1, ref_codes
        return -1, np.zeros((1, 128), dtype=np.float32)

    def _decode(self, codes_str: str, voice_id: int = -1, embedding: Optional[np.ndarray] = None) -> np.ndarray:
        from .utils import extract_speech_ids
        speech_ids = extract_speech_ids(codes_str)
        if not speech_ids:
            return np.array([], dtype=np.float32)
        tokens = np.array(speech_ids, dtype=np.int64)[None, :]
        v_id = np.array([voice_id], dtype=np.int64)
        if embedding is None:
            embedding = np.zeros((1, 128), dtype=np.float32)
        inputs = {"content_ids": tokens, "voice_id": v_id, "evoice_embedding": embedding}
        audio = self.decoder_sess.run(None, inputs)[0]
        if audio.ndim == 3:
            return audio[0, 0, :]
        elif audio.ndim == 2:
            return audio[0, :]
        return audio.flatten()

    def infer(
        self,
        text: str,
        ref_codes: Optional[Any] = None,
        temperature: float = 0.4,
        top_k: int = 50,
        max_chars: int = 256,
        skip_normalize: bool = False,
        skip_phonemize: bool = False,
        **kwargs
    ) -> np.ndarray:
        phonemes = phonemize_text(text) if not skip_phonemize else text

        chunks = split_into_chunks_v2(phonemes, max_chunk_size=max_chars)
        if not chunks:
            return np.array([], dtype=np.float32)

        if ref_codes is None and self._default_voice:
            ref_codes = self.get_preset_voice(self._default_voice)

        voice_id, emb = self._get_voice_params(ref_codes)

        all_wavs = []
        for i, chunk in enumerate(chunks):
            prompt = self._format_turbo_prompt(chunk.text)  # ← .text

            self.backbone.reset()
            result = self.backbone(
                prompt,
                max_tokens=2048,
                temperature=temperature,
                top_k=top_k,
                top_p=0.95,
                min_p=0.05,
                stop=["<|SPEECH_GENERATION_END|>"],
                repeat_penalty=1.15,
                echo=False,
            )
            wav = self._decode(result["choices"][0]["text"], voice_id, emb)
            all_wavs.append(wav)

            if i < len(chunks) - 1:
                silence_dur = get_silence_duration_v2(chunk)  # ← PhoneChunk
                if silence_dur > 0:
                    all_wavs.append(np.zeros(int(self.sample_rate * silence_dur), dtype=np.float32))

        final_wav = np.concatenate(all_wavs) if len(all_wavs) > 1 else all_wavs[0]
        return self._apply_watermark(final_wav)

    def _format_turbo_prompt(self, phonemes: str) -> str:
        return (
            f"<|speaker_16|>"
            f"<|TEXT_PROMPT_START|>{phonemes}<|TEXT_PROMPT_END|>"
            f"<|SPEECH_GENERATION_START|>"
        )

    def infer_stream(
        self,
        text: str,
        ref_codes: Optional[Any] = None,
        temperature: float = 0.4,
        top_k: int = 50,
        max_chars: int = 256,
        skip_normalize: bool = False,
        skip_phonemize: bool = False,
        **kwargs
    ) -> Generator[np.ndarray, None, None]:
        phonemes = phonemize_text(text) if not skip_phonemize else text

        chunks = split_into_chunks_v2(phonemes, max_chunk_size=max_chars)

        if ref_codes is None and self._default_voice:
            ref_codes = self.get_preset_voice(self._default_voice)

        voice_id, emb = self._get_voice_params(ref_codes)

        for i, chunk in enumerate(chunks):
            prompt = self._format_turbo_prompt(chunk.text)  # ← .text

            self.backbone.reset()
            result = self.backbone(
                prompt,
                max_tokens=2048,
                temperature=temperature,
                top_k=top_k,
                top_p=0.95,
                min_p=0.05,
                stop=["<|SPEECH_GENERATION_END|>"],
                repeat_penalty=1.15,
                echo=False,
            )
            wav = self._decode(result["choices"][0]["text"], voice_id, emb)
            yield self._apply_watermark(wav)

            if i < len(chunks) - 1:
                silence_dur = get_silence_duration_v2(chunk)  # ← PhoneChunk
                if silence_dur > 0:
                    yield np.zeros(int(self.sample_rate * silence_dur), dtype=np.float32)

    def infer_batch(self, texts: List[str], **kwargs) -> List[np.ndarray]:
        return [self.infer(t, **kwargs) for t in texts]

    def close(self):
        if self.backbone:
            self.backbone.close()
            self.backbone = None
        self.decoder_sess = None