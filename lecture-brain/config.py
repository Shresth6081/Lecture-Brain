from pathlib import Path

BASE_DIR   = Path(__file__).parent
STORAGE_DIR  = BASE_DIR / "storage"
UPLOADS_DIR  = STORAGE_DIR / "uploads"
TRANSCRIPTS_DIR = STORAGE_DIR / "transcripts"
CHROMA_DIR  = STORAGE_DIR / "chroma_db"

for _d in [UPLOADS_DIR, TRANSCRIPTS_DIR, CHROMA_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

WHISPER_MODEL_SIZE = "base"
SUPPORTED_AUDIO_FORMATS = {
    ".mp3", ".mp4", ".wav", ".m4a",
    ".ogg", ".flac", ".webm", ".mkv", ".avi", ".mov"
}

OLLAMA_BASE_URL  = "http://localhost:11434"
OLLAMA_EMBED_MODEL = "mxbai-embed-large"
OLLAMA_LLM_MODEL  = "llama3"

CHUNK_SIZE  = 500 
CHUNK_OVERLAP = 60 
TOP_K_RESULTS = 5