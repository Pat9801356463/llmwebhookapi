# engine/faiss_handler.py
import faiss
import numpy as np
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
from config import Config
from engine.db import save_chunks_to_db, fetch_chunks_from_db

# Load embedding model
EMBEDDING_MODEL = SentenceTransformer(Config.EMBEDDING_MODEL_NAME)


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
    """Get embeddings for text chunks."""
    return EMBEDDING_MODEL.encode(chunks, convert_to_numpy=True)


def build_faiss_index_from_embeddings(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    """Build a FAISS index directly from precomputed embeddings."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


def retrieve_top_chunks(query: str, chunks: List[str], embeddings: np.ndarray, top_k: int = 3) -> List[str]:
    """
    Retrieve most relevant chunks for a query given in-memory chunks + embeddings.
    This avoids re-fetching from DB for every search.
    """
    query_vec = embed_chunks([query])
    index = build_faiss_index_from_embeddings(embeddings)
    distances, indices = index.search(query_vec, top_k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]


def process_and_store_document(doc_id: str, doc_type: str, text: str) -> Tuple[List[str], np.ndarray]:
    """
    Process a new document into chunks + embeddings and save to DB.
    Returns (chunks, embeddings).
    """
    chunks = chunk_text(text, Config.CHUNK_SIZE, Config.OVERLAP_SIZE)
    embeddings = embed_chunks(chunks)
    save_chunks_to_db(
        [{"doc_id": doc_id, "doc_type": doc_type, "chunk_id": i, "text": c} for i, c in enumerate(chunks)],
        embeddings
    )
    return chunks, embeddings
