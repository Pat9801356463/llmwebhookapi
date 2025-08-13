# engine/faiss_handler.py
import os
import faiss
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
from config import Config
from pathlib import Path
import fitz  # PyMuPDF - fast PDF parser
from engine import db  # ✅ Added import to save to Postgres

# === Global In-Memory Store ===
_EMBED_MODEL = None
_FAISS_INDEX = None
_CHUNKS = []  # text chunks in same order as vectors


def get_embedding_model():
    """Load embedding model once."""
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        _EMBED_MODEL = SentenceTransformer(Config.EMBEDDING_MODEL_NAME)
    return _EMBED_MODEL


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """Split into overlapping chunks."""
    tokens = text.split()
    chunks = []
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk = " ".join(tokens[i:i + chunk_size]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def embed_texts(texts: List[str]) -> np.ndarray:
    """Batch encode texts into embeddings."""
    model = get_embedding_model()
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=False)


def load_pdf_text(pdf_path: str) -> str:
    """Extract text from PDF."""
    doc = fitz.open(pdf_path)
    text = "\n".join(page.get_text("text") for page in doc)
    return text.strip()


def ingest_pdf_to_memory(pdf_path: str, doc_id: str = "default", doc_type: str = "pdf", source: str = None) -> None:
    """
    Ingest PDF into in-memory FAISS index AND save chunks/embeddings to Postgres.
    """
    global _FAISS_INDEX, _CHUNKS

    # 1. Parse PDF
    text = load_pdf_text(pdf_path)
    # 2. Chunk text
    chunks = chunk_text(text, Config.CHUNK_SIZE, Config.OVERLAP_SIZE)
    # 3. Embed all chunks
    embeddings = embed_texts(chunks)

    # 4. Build FAISS index in memory
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # 5. Store in globals
    _FAISS_INDEX = index
    _CHUNKS = chunks

    # 6. Save chunks + embeddings to Postgres
    try:
        print(f"💾 Saving {len(chunks)} chunks to Postgres for doc_id={doc_id}...")
        db.save_chunks_to_db(doc_id=doc_id, doc_type=doc_type, chunks=chunks, embeddings=embeddings, source=source)
        print("✅ Chunks + embeddings saved to Postgres.")
    except Exception as e:
        print(f"❌ Failed to save chunks to Postgres: {e}")


def retrieve_top_chunks_batch(queries: List[str], top_k: int = 3) -> List[List[str]]:
    """
    Retrieve top-k chunks for each query in batch mode.
    Returns: List of results per query.
    """
    global _FAISS_INDEX, _CHUNKS
    if _FAISS_INDEX is None:
        raise ValueError("FAISS index is empty. Call ingest_pdf_to_memory() first.")

    # Embed all queries in one batch
    query_vecs = embed_texts(queries)

    # Search all queries
    distances, indices = _FAISS_INDEX.search(query_vecs, top_k)

    results = []
    for idx_list in indices:
        result_chunks = [_CHUNKS[i] for i in idx_list if i < len(_CHUNKS)]
        results.append(result_chunks)

    return results


def ensure_faiss_index():
    """
    Ensure FAISS index exists on disk.
    Creates an empty index if it doesn't exist.
    """
    index_path = getattr(Config, "FAISS_INDEX_PATH", "data/faiss_index.bin")
    os.makedirs(os.path.dirname(index_path), exist_ok=True)

    if not os.path.exists(index_path):
        dim = Config.EMBEDDING_DIM
        index = faiss.IndexFlatL2(dim)
        faiss.write_index(index, index_path)
        print(f"📂 Created new FAISS index at {index_path}")
    else:
        print(f"📂 FAISS index already exists at {index_path}")
