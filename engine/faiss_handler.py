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

def build_faiss_index(chunks: List[str]) -> Tuple[faiss.IndexFlatL2, dict]:
    """
    Build FAISS index from chunks and return index + mapping dict.
    chunk_map: {idx: text_chunk}
    """
    embeddings = embed_chunks(chunks)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    chunk_map = {i: chunk for i, chunk in enumerate(chunks)}
    return index, chunk_map

def retrieve_top_chunks(query: str, index: faiss.IndexFlatL2, chunk_map: dict, top_k: int = 3) -> List[str]:
    """Retrieve most relevant chunks for a query."""
    query_vec = embed_chunks([query])
    distances, indices = index.search(query_vec, top_k)
    return [chunk_map[i] for i in indices[0] if i in chunk_map]

# === DB-Aware Wrappers ===
def process_and_store_document(doc_id: str, text: str, doc_type: str = "policy") -> Tuple[faiss.IndexFlatL2, dict]:
    """
    Process text into chunks, store in DB, and return FAISS index + mapping.
    If doc_id exists in DB, skip embedding and build from DB data.
    """
    existing_chunks = fetch_chunks_from_db(doc_id)
    if existing_chunks:
        chunks = [row["text"] for row in existing_chunks]
        embeddings = np.vstack([row["embedding"] for row in existing_chunks])
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        chunk_map = {i: chunk for i, chunk in enumerate(chunks)}
        return index, chunk_map

    # New doc — process & save
    chunks = chunk_text(text, Config.CHUNK_SIZE, Config.OVERLAP_SIZE)
    embeddings = embed_chunks(chunks)
    save_chunks_to_db(doc_id, doc_type, chunks, embeddings)
    index, chunk_map = build_faiss_index(chunks)
    return index, chunk_map
