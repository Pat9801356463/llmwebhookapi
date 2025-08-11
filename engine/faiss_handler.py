import faiss
import numpy as np
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
from config import Config
from engine.db import save_chunks_to_db, fetch_chunks_from_db

# === Lazy load embedding model ===
EMBEDDING_MODEL = None

def get_embedding_model():
    """
    Loads the embedding model only once at runtime.
    This avoids downloading the model during Railway build.
    """
    global EMBEDDING_MODEL
    if EMBEDDING_MODEL is None:
        EMBEDDING_MODEL = SentenceTransformer(Config.EMBEDDING_MODEL_NAME)
    return EMBEDDING_MODEL


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """Split text into overlapping chunks for embeddings."""
    tokens = text.split()
    chunks = []
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk = " ".join(tokens[i:i + chunk_size]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def embed_chunks(chunks: List[str]) -> np.ndarray:
    """Generate dense embeddings for a list of text chunks."""
    model = get_embedding_model()
    return model.encode(chunks, convert_to_numpy=True)


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    """Create a FAISS index for similarity search."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


def process_and_store_document(doc_id: str, doc_type: str, text: str, source: str = None) -> Tuple[List[str], np.ndarray]:
    """Chunk → embed → save to DB. Skip if already exists."""
    existing = fetch_chunks_from_db(doc_id)
    if existing:
        chunks = [c["text"] for c in existing]
        embeddings = np.vstack([c["embedding"] for c in existing])
        return chunks, embeddings

    chunks = chunk_text(text, Config.CHUNK_SIZE, Config.OVERLAP_SIZE)
    embeddings = embed_chunks(chunks)
    save_chunks_to_db(doc_id, doc_type, chunks, embeddings, source=source)
    return chunks, embeddings


def retrieve_top_chunks(query: str, doc_id: str, top_k: int = 3) -> List[str]:
    """Retrieve top-K chunks for a query from DB + FAISS search."""
    stored_chunks = fetch_chunks_from_db(doc_id)
    if not stored_chunks:
        return []

    chunks = [c["text"] for c in stored_chunks]
    embeddings = np.vstack([c["embedding"] for c in stored_chunks])

    query_vec = embed_chunks([query])
    index = build_faiss_index(embeddings)
    distances, indices = index.search(query_vec, top_k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]
