# engine/faiss_handler.py
import os
import faiss
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
from config import Config
import fitz  # PyMuPDF
from engine import db  # Postgres storage

_EMBED_MODEL = None
_FAISS_INDEX = None
_CHUNKS: List[str] = []

def get_embedding_model():
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        _EMBED_MODEL = SentenceTransformer(Config.EMBEDDING_MODEL_NAME)
    return _EMBED_MODEL

def embed_texts(texts: List[str]) -> np.ndarray:
    model = get_embedding_model()
    embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    embs = np.asarray(embs, dtype=np.float32)
    # pad/truncate to Config.EMBEDDING_DIM
    if embs.ndim == 1:
        embs = np.expand_dims(embs, axis=0)
    cur = embs.shape[1]
    target = Config.EMBEDDING_DIM
    if cur < target:
        pad = np.zeros((embs.shape[0], target - cur), dtype=np.float32)
        return np.concatenate([embs, pad], axis=1)
    if cur > target:
        return embs[:, :target]
    return embs

def load_pdf_text(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    text = "\n".join(page.get_text("text") for page in doc)
    return text.strip()

def ingest_pdf_to_memory(pdf_path: str, doc_id: str = "default", doc_type: str = "pdf", source: str = None) -> None:
    global _FAISS_INDEX, _CHUNKS
    text = load_pdf_text(pdf_path)
    from engine.embedding_handler import chunk_text
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
    global _FAISS_INDEX, _CHUNKS
    records = db.fetch_chunks_from_db(doc_id)
    if not records:
        print(f"⚠️ No stored chunks found for doc_id={doc_id}.")
        return False
    _CHUNKS = [rec["text"] for rec in records]
    embeddings = np.array([rec["embedding"] for rec in records], dtype=np.float32)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    _FAISS_INDEX = index
    print(f"✅ FAISS index rebuilt with {len(_CHUNKS)} chunks.")
    return True

def retrieve_top_chunks_batch(queries: List[str], top_k: int = 3) -> List[List[str]]:
    global _FAISS_INDEX, _CHUNKS
    if _FAISS_INDEX is None:
        print("ℹ️ FAISS index empty. Attempting rebuild from DB...")
        if not rebuild_faiss_from_db("default"):
            raise ValueError("FAISS index is empty and no stored data found.")
    query_vecs = embed_texts(queries)
    distances, indices = _FAISS_INDEX.search(query_vecs, top_k)
    results = []
    for idx_list in indices:
        result_chunks = [_CHUNKS[i] for i in idx_list if i < len(_CHUNKS)]
        results.append(result_chunks)
    return results
