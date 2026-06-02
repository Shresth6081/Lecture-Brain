from typing import Optional

import torch
import whisper

from config import WHISPER_MODEL_SIZE

_model = None
_loaded_size: Optional[str] = None


def get_model(size: Optional[str] = None) -> whisper.Whisper:
    """Return (and lazily load) the Whisper model."""
    global _model, _loaded_size
    size = size or WHISPER_MODEL_SIZE

    if _model is None or _loaded_size != size:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Whisper] Loading '{size}' model on {device} …")
        _model = whisper.load_model(size, device=device)
        _loaded_size = size
        print(f"[Whisper] Model ready.")

    return _model


def transcribe(
    audio_path: str,
    language: Optional[str] = None,
    model_size: Optional[str] = None,
) -> dict:
    model = get_model(model_size)

    options: dict = {"verbose": False}
    if language:
        options["language"] = language

    result = model.transcribe(audio_path, **options)

    segments = [
        {
            "id":    s["id"],
            "start": round(s["start"], 2),
            "end":   round(s["end"],   2),
            "text":  s["text"].strip(),
        }
        for s in result.get("segments", [])
    ]

    duration = segments[-1]["end"] if segments else 0.0

    return {
        "text":     result["text"].strip(),
        "language": result.get("language", "unknown"),
        "segments": segments,
        "duration": duration,
    }
