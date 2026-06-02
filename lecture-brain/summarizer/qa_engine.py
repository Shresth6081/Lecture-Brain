import requests

from config import OLLAMA_BASE_URL, OLLAMA_LLM_MODEL
from rag.retriever import semantic_search


def answer_question(
    query: str,
    session_id: str,
    use_llm: bool = True,
) -> dict:
    
    chunks = semantic_search(query, session_id)

    if not chunks:
        return {
            "answer":  "No relevant content found. Please ingest the transcript first.",
            "sources": [],
            "mode":    "none",
        }

    # Pure retrieval mode
    if not use_llm:
        top_text = "\n\n---\n\n".join(c["text"] for c in chunks[:3])
        return {"answer": top_text, "sources": chunks, "mode": "retrieval"}

    # LLM synthesis via Ollama
    context = "\n\n".join(
        f"[Source {i+1} — {c['source']}]:\n{c['text']}"
        for i, c in enumerate(chunks)
    )

    prompt = (
        "You are a helpful study assistant.\n"
        "Answer the question using ONLY the lecture content provided below.\n"
        "If the answer is not present, say: 'This topic was not covered in the lecture.'\n\n"
        f"Lecture Content:\n{context}\n\n"
        f"Question: {query}\n\nAnswer:"
    )

    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": OLLAMA_LLM_MODEL, "prompt": prompt, "stream": False},
            timeout=90,
        )
        resp.raise_for_status()
        answer = resp.json().get("response", "").strip()
        mode   = "llm"
    except Exception as exc:
        answer = f"[Ollama unavailable: {exc}]\n\n{chunks[0]['text']}"
        mode   = "retrieval_fallback"

    return {"answer": answer, "sources": chunks, "mode": mode}
