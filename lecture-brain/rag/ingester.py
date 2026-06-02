import uuid
from pathlib import Path

import fitz  # PyMuPDF

from config import CHUNK_SIZE, CHUNK_OVERLAP
from rag.embedder import embed_texts
from rag.vector_store import get_or_create_collection

def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        piece = text[start : start + chunk_size].strip()
        if piece:
            chunks.append(piece)
        start += chunk_size - overlap
    return chunks


def ingest_text(
    text: str,
    session_id: str,
    source_name: str = "transcript",
) -> dict:
    """Chunk *text*, embed, and upsert into ChromaDB for *session_id*."""
    chunks     = chunk_text(text)
    embeddings = embed_texts(chunks)
    collection = get_or_create_collection(session_id)

    ids       = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [
        {"source": source_name, "chunk_index": i}
        for i in range(len(chunks))
    ]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
    )

    return {"chunks_added": len(chunks), "session_id": session_id}


def ingest_pdf(pdf_path: str, session_id: str) -> dict:
    """Extract text from every PDF page and ingest it."""
    doc       = fitz.open(pdf_path)
    full_text = "\n".join(page.get_text() for page in doc)
    doc.close()

    return ingest_text(
        full_text,
        session_id,
        source_name=Path(pdf_path).name,
    )