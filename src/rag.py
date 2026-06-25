"""
rag.py
--------
Lets the agent answer documentation questions (e.g. "what does inventory
cover mean?") by retrieving the most relevant passage from your project's
markdown docs.

DESIGN CHOICE: uses sentence-transformers + plain cosine similarity
instead of Chroma. Chroma is what the project doc lists, but it adds
its own setup overhead (a local database, occasional install quirks on
some machines). For a handful of short markdown docs, plain embeddings
+ numpy does the same job and is simpler to get running today. You can
swap in Chroma later without changing load_documents()/build_index()'s
logic — just swap what stores/searches the vectors.

REQUIRES: pip install sentence-transformers (already in requirements.txt)
FIRST RUN: downloads a small embedding model (~80MB, needs internet once,
then cached locally).
"""

import os
import glob
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_ollama import ChatOllama

_llm = None

def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOllama(model="mistral")
    return _llm


def rephrase_with_ollama(question: str, retrieved_text: str) -> str:
    """
    Asks the local LLM to turn the raw retrieved passage into a natural
    answer. Falls back to the raw text if Ollama isn't running, so a
    missing/stopped Ollama never crashes the whole answer.
    """
    try:
        llm = _get_llm()
        prompt = (
            f"Using ONLY this information, answer the question naturally "
            f"in 1-2 sentences:\n\n{retrieved_text}\n\nQuestion: {question}"
        )
        response = llm.invoke(prompt)
        content = getattr(response, "content", response)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(str(item) for item in content)
        return str(content)
    except Exception as e:
        print(f"Ollama not available ({e}) — returning raw retrieved text instead.")
        return retrieved_text


_model = None  # loaded once, lazily, on first use


def _get_model():
    global _model
    if _model is None:
        print("Loading embedding model (first call only, can take a few seconds)...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def load_documents(docs_folder: str = "docs/") -> list:
    """
    Read every .md file in docs_folder and split each into paragraph-sized
    chunks (split on blank lines). Returns a list of
    {"source": filename, "text": chunk}.
    """
    chunks = []
    for filepath in glob.glob(os.path.join(docs_folder, "*.md")):
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        for paragraph in paragraphs:
            chunks.append({"source": os.path.basename(filepath), "text": paragraph})
    return chunks


def build_index(chunks: list) -> np.ndarray:
    """Embed every chunk's text. Returns a matrix of shape (num_chunks, 384)."""
    if not chunks:
        return np.empty((0, 384))  # all-MiniLM-L6-v2 produces 384-dim vectors
    model = _get_model()
    texts = [c["text"] for c in chunks]
    return np.asarray(model.encode(texts))


def _cosine_similarity(query_vec: np.ndarray, doc_matrix: np.ndarray) -> np.ndarray:
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
    doc_norms = doc_matrix / (np.linalg.norm(doc_matrix, axis=1, keepdims=True) + 1e-8)
    return doc_norms @ query_norm


def query_docs(question: str, chunks: list, doc_matrix: np.ndarray, top_k: int = 3) -> list:
    """Return the top_k most relevant chunks for a question."""
    if len(chunks) == 0:
        return []
    model = _get_model()
    query_vec = np.asarray(model.encode([question])[0])
    scores = _cosine_similarity(query_vec, doc_matrix)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [{**chunks[i], "score": float(scores[i])} for i in top_indices]


def answer_doc_question(question: str, docs_folder: str = "docs/") -> dict:
    """
    Main entry point: load docs, retrieve the most relevant passage(s),
    return them as the answer.

    NOT YET DONE (optional next step): piping this through Ollama to
    rephrase the retrieved passage more conversationally, instead of
    returning the raw text. This is the natural point in the sprint to
    add that, per the earlier "when does Ollama get used" answer.
    """
    chunks = load_documents(docs_folder)
    if not chunks:
        return {
            "question": question,
            "answer": f"No documents found in {docs_folder} yet — add your .md files there first.",
            "sources": [],
        }

    doc_matrix = build_index(chunks)
    top_matches = query_docs(question, chunks, doc_matrix, top_k=2)

    answer_text = "\n\n".join(m["text"] for m in top_matches)
    natural_answer = rephrase_with_ollama(question, answer_text)
    sources = list({m["source"] for m in top_matches})
    return {"question": question, "answer": natural_answer, "sources": sources}


if __name__ == "__main__":
    # Run with: python src/rag.py
    # Make sure docs/ has at least one .md file (even a short draft) first.
    result = answer_doc_question("What does inventory cover days mean?")
    print("Q:", result["question"])
    print("A:", result["answer"])
    print("Sources:", result["sources"])