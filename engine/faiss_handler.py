# engine/faiss_handler.py
import os
import faiss
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
from config import Config
import fitz  # PyMuPDF
from engine import db  # Postgres storage

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
    Ingest PDF into FAISS index AND save to Postgres.
    """
    global _FAISS_INDEX, _CHUNKS

    text = load_pdf_text(pdf_path)
    chunks = chunk_text(text, Config.CHUNK_SIZE, Config.OVERLAP_SIZE)
    embeddings = embed_texts(chunks)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    _FAISS_INDEX = index
    _CHUNKS = chunks

    try:
        print(f"💾 Saving {len(chunks)} chunks to Postgres for doc_id={doc_id}...")
        db.save_chunks_to_db(doc_id=doc_id, doc_type=doc_type, chunks=chunks, embeddings=embeddings, source=source)
        print("✅ Chunks + embeddings saved to Postgres.")
    except Exception as e:
        print(f"❌ Failed to save chunks to Postgres: {e}")


def rebuild_faiss_from_db(doc_id: str) -> bool:
    """
    Load chunks + embeddings from Postgres and rebuild FAISS index in memory.
    """
    global _FAISS_INDEX, _CHUNKS

    records = db.fetch_chunks_from_db(doc_id)
    if not records:
        print(f"⚠️ No stored chunks found for doc_id={doc_id}. Cannot rebuild FAISS index.")
        return False

    _CHUNKS = [rec["text"] for rec in records]
    embeddings = np.array([rec["embedding"] for rec in records], dtype=np.float32)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    _FAISS_INDEX = index
    print(f"✅ FAISS index rebuilt from Postgres for doc_id={doc_id} with {len(_CHUNKS)} chunks.")
    return True


def retrieve_top_chunks_batch(queries: List[str], top_k: int = 3) -> List[List[str]]:
    """
    Retrieve top-k chunks for each query in batch mode.
    """
    global _FAISS_INDEX, _CHUNKS
    if _FAISS_INDEX is None:
        # Try to rebuild from Postgres if possible
        print("ℹ️ FAISS index empty. Attempting rebuild from DB...")
        if not rebuild_faiss_from_db("default"):
            raise ValueError("FAISS index is empty and no stored data found for doc_id='default'.")

    query_vecs = embed_texts(queries)
    distances, indices = _FAISS_INDEX.search(query_vecs, top_k)

    results = []
    for idx_list in indices:
        result_chunks = [_CHUNKS[i] for i in idx_list if i < len(_CHUNKS)]
        results.append(result_chunks)

    return results
