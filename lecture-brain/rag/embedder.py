from typing import Optional
from langchain_community.embeddings import OllamaEmbeddings
from config import OLLAMA_BASE_URL, OLLAMA_EMBED_MODEL

_embedder: Optional[OllamaEmbeddings] = None

def get_embedder() -> OllamaEmbeddings:
    global _embedder
    if _embedder is None:
        _embedder = OllamaEmbeddings(
            model=OLLAMA_EMBED_MODEL,
            base_url=OLLAMA_BASE_URL,
        )
    return _embedder


def embed_texts(texts: list[str]) -> list[list[float]]:
    return get_embedder().embed_documents(texts)


def embed_query(query: str) -> list[float]:
    return get_embedder().embed_query(query)