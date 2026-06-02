from typing import Optional
import chromadb
from config import CHROMA_DIR

_client: Optional[chromadb.PersistentClient] = None

def get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return _client


def get_or_create_collection(session_id: str):
    return get_client().get_or_create_collection(
        name=session_id,
        metadata={"hnsw:space": "cosine"},
    )


def list_collections() -> list[str]:
    return [c.name for c in get_client().list_collections()]


def delete_collection(session_id: str) -> None:
    get_client().delete_collection(session_id)


def collection_count(session_id: str) -> int:
    try:
        return get_or_create_collection(session_id).count()
    except Exception:
        return 0