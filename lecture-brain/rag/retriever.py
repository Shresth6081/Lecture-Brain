from config import TOP_K_RESULTS
from rag.embedder import embed_query
from rag.vector_store import get_or_create_collection, collection_count

def semantic_search(
    query: str,
    session_id: str,
    top_k: int = TOP_K_RESULTS,
) -> list[dict]:
    count = collection_count(session_id)
    if count == 0:
        return []

    q_embedding = embed_query(query)
    collection  = get_or_create_collection(session_id)

    results = collection.query(
        query_embeddings=[q_embedding],
        n_results=min(top_k, count),
        include=["documents", "metadatas", "distances"],
    )

    chunks: list[dict] = []
    for i, doc in enumerate(results["documents"][0]):
        meta = results["metadatas"][0][i]
        distance = results["distances"][0][i]
        chunks.append(
            {
                "text":            doc,
                "source":          meta.get("source", "unknown"),
                "chunk_index":     meta.get("chunk_index", i),
                "relevance_score": round(max(0.0, 1.0 - distance), 4),
            }
        )

    return chunks
