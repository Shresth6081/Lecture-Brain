import subprocess
from pathlib import Path

from config import SUPPORTED_AUDIO_FORMATS, UPLOADS_DIR


def validate_format(file_path: str) -> bool:
    return Path(file_path).suffix.lower() in SUPPORTED_AUDIO_FORMATS


def convert_to_wav(input_path: str) -> str:
    src = Path(input_path)

    if src.suffix.lower() == ".wav":
        return str(src)

    dst = UPLOADS_DIR / f"{src.stem}_converted.wav"
    # get this command from the web
    cmd = [
        "ffmpeg", "-y",
        "-i",  str(src),
        "-ar", "16000",       # sample rate Whisper expects
        "-ac", "1",           # mono
        "-c:a", "pcm_s16le",  # standard 16-bit PCM
        str(dst),
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg conversion failed for {src.name}:\n{proc.stderr[-800:]}"
        )

    return str(dst)


def get_audio_duration(file_path: str) -> float:
    # get this command from the web
    cmd = [
        "ffprobe", "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        file_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return float(proc.stdout.strip())
    except ValueError:
        return 0.0