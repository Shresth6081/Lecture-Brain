from pathlib import Path

import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from api.routes_files import router as files_router
from api.routes_rag import router as rag_router
from api.routes_transcribe import router as transcribe_router
from config import OLLAMA_BASE_URL

BASE_DIR  = Path(__file__).parent

REACT_DIST  = BASE_DIR.parent
REACT_ASSETS = REACT_DIST / "assets"

app = FastAPI(
    title  = "LectureBrain API",
    description = "AI-powered lecture transcription and RAG Q&A",
    version  = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_credentials = True,
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

app.include_router(transcribe_router, prefix="/api", tags=["Transcription"])
app.include_router(rag_router,  prefix="/api", tags=["RAG"])
app.include_router(files_router,  prefix="/api", tags=["Files"])

if REACT_ASSETS.exists():
    app.mount("/assets", StaticFiles(directory=str(REACT_ASSETS)), name="assets")

    @app.get("/", include_in_schema=False)
    @app.get("/{full_path:path}", include_in_schema=False)
    def serve_react(full_path: str = ""):
        """Serve React SPA — all non-API routes return index.html."""
        index = REACT_DIST / "index.html"
        if index.exists():
            return FileResponse(str(index))
        return JSONResponse({"detail": "Frontend not built. Run: cd frontend && npm run build"}, status_code=503)

@app.get("/health", tags=["Health"])
def health_check():
    """Returns status of the API and Ollama connectivity."""
    ollama_ok = False
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        ollama_ok = r.status_code == 200
    except Exception:
        pass

    return JSONResponse({
        "api":    "ok",
        "ollama": "ok" if ollama_ok else "unreachable",
    })