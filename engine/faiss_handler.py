import faiss
import numpy as np
from typing import List, Tuple, Dict
from sentence_transformers import SentenceTransformer
from config import Config
from engine.db import save_chunks_to_db, fetch_chunks_from_db

# ===== Singleton Embedding Model =====
_EMBEDDING_MODEL = None


def get_embedding_model() -> SentenceTransformer:
    """Load embedding model only once per runtime."""
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        _EMBEDDING_MODEL = SentenceTransformer(Config.EMBEDDING_MODEL_NAME)
    return _EMBEDDING_MODEL


# ===== In-Memory FAISS Index Store =====
# Keeps per-document embeddings + chunks for fast retrieval
_IN_MEMORY_INDEX: Dict[str, Tuple[faiss.IndexFlatL2, List[str]]] = {}


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """Split text into overlapping chunks."""
    tokens = text.split()
    chunks = []
    step = max(chunk_size - overlap, 1)
    for i in range(0, len(tokens), step):
        chunk = " ".join(tokens[i:i + chunk_size]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def embed_chunks(chunks: List[str]) -> np.ndarray:
    """Generate dense embeddings for text chunks."""
    model = get_embedding_model()
    return model.encode(chunks, convert_to_numpy=True)


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    """Create a FAISS index for similarity search."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


def process_and_store_document(
    doc_id: str,
    doc_type: str,
    text: str,
    source: str = None
) -> Tuple[List[str], np.ndarray]:
    """
    Chunk → embed → save to DB + in-memory FAISS.
    Skips recomputation if already in DB.
    """
    existing = fetch_chunks_from_db(doc_id)
    if existing:
        chunks = [c["text"] for c in existing]
        embeddings = np.vstack([c["embedding"] for c in existing])
    else:
        chunks = chunk_text(text, Config.CHUNK_SIZE, Config.OVERLAP_SIZE)
        embeddings = embed_chunks(chunks)
        save_chunks_to_db(doc_id, doc_type, chunks, embeddings, source=source)

    # Always refresh in-memory FAISS for this doc
    _IN_MEMORY_INDEX[doc_id] = (build_faiss_index(embeddings), chunks)
    return chunks, embeddings


def retrieve_top_chunks(query: str, doc_id: str, top_k: int = 3) -> List[str]:
    """
    Retrieve top-K chunks for a query.
    Uses in-memory FAISS if available, else rebuilds from DB.
    """
    if doc_id not in _IN_MEMORY_INDEX:
        stored_chunks = fetch_chunks_from_db(doc_id)
        if not stored_chunks:
            return []
        chunks = [c["text"] for c in stored_chunks]
        embeddings = np.vstack([c["embedding"] for c in stored_chunks])
        _IN_MEMORY_INDEX[doc_id] = (build_faiss_index(embeddings), chunks)

    index, chunks = _IN_MEMORY_INDEX[doc_id]
    query_vec = embed_chunks([query])
    distances, indices = index.search(query_vec, top_k)
    return [chunks[i] for i in indices[0] if 0 <= i < len(chunks)]
