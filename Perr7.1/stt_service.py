"""perr71-stt — локальный сервис распознавания речи для Perr7.1.

Отдельный от продакшн-бота «Тарис» процесс. n8n работает в отдельной docker-сети
и не видит 127.0.0.1 хоста, поэтому сервис слушает 0.0.0.0 (все интерфейсы) —
это технически расширяет видимость порта, поэтому запросы без правильного
токена (PERR71_STT_TOKEN, заголовок X-Perr71-Token) отклоняются. Переиспользует
faster-whisper и уже скачанный кэш моделей бота «Тарис» (HF_HUB_CACHE), новую
модель не качает.

POST /transcribe  (multipart/form-data, поле "audio", заголовок X-Perr71-Token)
                  -> {"text": "...", "duration_s": ...}
GET  /health      -> {"status": "ok"}  (без токена, для проверки живости)

Развёрнут на SSH_HOST в ~/perr71-stt (см. Perr7.1/02-развертывание.md).
"""

import asyncio
import os
import subprocess
import tempfile
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_CACHE", "/opt/taris-docker/data/whisper/hub")

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from faster_whisper import WhisperModel

MODEL_SIZE = os.environ.get("PERR71_WHISPER_MODEL", "small")
COMPUTE_TYPE = os.environ.get("PERR71_WHISPER_COMPUTE", "int8")
THREADS = int(os.environ.get("PERR71_WHISPER_THREADS", "4"))
STT_TOKEN = os.environ.get("PERR71_STT_TOKEN", "")

app = FastAPI(title="perr71-stt")
_model: WhisperModel | None = None


def get_model() -> WhisperModel:
    global _model
    if _model is None:
        _model = WhisperModel(MODEL_SIZE, device="cpu", compute_type=COMPUTE_TYPE, cpu_threads=THREADS)
    return _model


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_SIZE}


MAX_PROCESSING_SECONDS = int(os.environ.get("PERR71_MAX_PROCESSING_SECONDS", "240"))


def _run_ffmpeg(tmp_in_path: str, wav_path: str) -> None:
    ff = subprocess.run(
        ["ffmpeg", "-y", "-i", tmp_in_path, "-ar", "16000", "-ac", "1", wav_path],
        capture_output=True, timeout=60,
    )
    if ff.returncode != 0 or not Path(wav_path).exists():
        raise RuntimeError(f"ffmpeg failed: {ff.stderr.decode('utf-8', 'replace')[:500]}")


def _run_transcribe(wav_path: str) -> tuple[str, float]:
    model = get_model()
    # beam_size=1 и без vad_filter — на маленьких CPU-моделях (small) более сложные
    # настройки иногда приводят к зависанию/зацикливанию на длинных/зашумлённых записях
    # (см. Perr6.5 — похожая проблема была с Ollama). Простой greedy-декодинг стабильнее.
    segments, info = model.transcribe(wav_path, language="ru", beam_size=1, vad_filter=False)
    text = " ".join(seg.text.strip() for seg in segments).strip()
    return text, info.duration


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...), x_perr71_token: str = Header(default="")):
    if STT_TOKEN and x_perr71_token != STT_TOKEN:
        raise HTTPException(401, "Invalid token")
    data = await audio.read()
    if not data:
        raise HTTPException(400, "Empty audio")

    suffix = Path(audio.filename or "audio.bin").suffix or ".bin"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_in:
        tmp_in.write(data)
        tmp_in_path = tmp_in.name

    wav_path = tmp_in_path + ".wav"
    try:
        t0 = time.monotonic()
        try:
            await asyncio.wait_for(asyncio.to_thread(_run_ffmpeg, tmp_in_path, wav_path), timeout=90)
        except asyncio.TimeoutError:
            raise HTTPException(504, "ffmpeg timed out")
        except RuntimeError as e:
            raise HTTPException(500, str(e))

        try:
            text, audio_duration = await asyncio.wait_for(
                asyncio.to_thread(_run_transcribe, wav_path), timeout=MAX_PROCESSING_SECONDS
            )
        except asyncio.TimeoutError:
            raise HTTPException(504, f"transcription exceeded {MAX_PROCESSING_SECONDS}s — likely stuck/looping")

        duration_s = round(time.monotonic() - t0, 2)
        return {"text": text, "duration_s": duration_s, "audio_duration_s": round(audio_duration, 2)}
    finally:
        for p in (tmp_in_path, wav_path):
            try:
                os.remove(p)
            except OSError:
                pass
