# engine/pinecone_handler.py
from typing import List
import numpy as np
from pinecone import Pinecone, ServerlessSpec
from config import Config
from engine.db import save_chunks_to_db, fetch_chunks_from_db
from engine.embedding_handler import chunk_text, embed_chunks
from functools import lru_cache

# -----------------------------
# Pinecone client / index (v3)
# -----------------------------
_pc_client = Pinecone(api_key=Config.PINECONE_API_KEY)

def _ensure_index():
    indexes = {i["name"]: i for i in _pc_client.list_indexes()}
    if Config.PINECONE_INDEX_NAME not in indexes:
        print(f"[init] Creating Pinecone index '{Config.PINECONE_INDEX_NAME}' with dim={Config.EMBEDDING_DIM}")
        _pc_client.create_index(
            name=Config.PINECONE_INDEX_NAME,
            dimension=Config.EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    return _pc_client.Index(Config.PINECONE_INDEX_NAME)

_pc_index = _ensure_index()


def _pad_or_check_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Ensure embeddings are 2D and exactly EMBEDDING_DIM wide."""
    if embeddings is None or embeddings.size == 0:
        return np.zeros((0, Config.EMBEDDING_DIM), dtype=np.float32)
    if embeddings.ndim == 1:
        embeddings = embeddings[None, :]
    cur = embeddings.shape[1]
    target = Config.EMBEDDING_DIM
    if cur == target:
        return embeddings.astype(np.float32)
    if cur < target:
        pad = np.zeros((embeddings.shape[0], target - cur), dtype=np.float32)
        return np.hstack((embeddings.astype(np.float32), pad))
    return embeddings[:, :target].astype(np.float32)


def process_and_index_document(doc_id: str, doc_type: str, text: str, source: str = None) -> List[str]:
    """
    Chunk, embed, save to DB, and index in Pinecone.
    Called at ingestion time — not during queries.
    """
    existing = fetch_chunks_from_db(doc_id)
    if existing:
        chunks = [c["text"] for c in existing]
        embeddings = np.vstack([
            np.array(c.get("embedding") or np.zeros(Config.EMBEDDING_DIM), dtype=np.float32)
            for c in existing
        ])
    else:
        chunks = chunk_text(text, Config.CHUNK_SIZE, Config.OVERLAP_SIZE)
        if not chunks:
            print(f"[warn] No chunks produced for doc {doc_id}")
            return []
        embeddings = _pad_or_check_embeddings(embed_chunks(chunks))
        try:
            save_chunks_to_db(doc_id, doc_type, chunks, embeddings, source=source)
        except Exception as e:
            print(f"[warn] Failed to save chunks to DB for {doc_id}: {e}")

    vectors = [{
        "id": f"{doc_id}-{i}",
        "values": embeddings[i].tolist(),
        "metadata": {
            "text": chunks[i],
            "doc_id": doc_id,
            "doc_type": doc_type,
            "source": source or doc_id,
            "chunk_id": i,
        }
    } for i in range(len(chunks))]

    try:
        _pc_index.upsert(vectors=vectors, namespace=doc_id)
        print(f"[info] Upserted {len(vectors)} vectors into Pinecone ns='{doc_id}'")
    except Exception as e:
        print(f"[WARN] Pinecone upsert failed for doc {doc_id}: {e}")

    return chunks


@lru_cache(maxsize=512)
def retrieve_top_chunks(query: str, doc_id: str, top_k: int = 3) -> List[str]:
    """
    Query Pinecone for top_k chunks.
    - Embedding is batched & cached.
    - Smaller candidate pool for speed.
    - Optional local rerank only if vector values are present.
    """
    if not query or not doc_id:
        return []

    # Embed query
    q_vec = _pad_or_check_embeddings(embed_chunks([query]))
    if q_vec.size == 0:
        print(f"[warn] Query embedding is empty for query='{query}'")
        return []
    q_vec = q_vec[0]

    # Pinecone query
    try:
        res = _pc_index.query(
            vector=q_vec.tolist(),
            top_k=max(top_k * 2, 6),  # smaller candidate pool
            include_metadata=True,
            include_values=True,
            namespace=doc_id
        )
    except Exception as e:
        print(f"[WARN] Pinecone query failed: {e}")
        return []

    matches = getattr(res, "matches", None) or res.get("matches", [])
    if not matches:
        return []

    # Prepare candidates
    candidates = []
    for m in matches:
        md = m.get("metadata") or {}
        txt = (md.get("text") or "").strip()
        if not txt:
            continue
        raw_vals = m.get("values")
        vals = np.array(raw_vals, dtype=np.float32) if raw_vals else None
        score = float(m.get("score", 0.0))
        candidates.append((txt, vals, score))

    if not candidates:
        return []

    have_values = any(v is not None and v.size > 0 for _, v, _ in candidates)

    if have_values:
        def _cosine(a: np.ndarray, b_in):
            b = _pad_or_check_embeddings(np.array(b_in, dtype=np.float32))[0]
            denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
            return float(np.dot(a, b) / denom)
        candidates.sort(key=lambda x: _cosine(q_vec, x[1]), reverse=True)
    else:
        candidates.sort(key=lambda x: x[2], reverse=True)

    seen, ordered = set(), []
    for txt, _, _ in candidates:
        key = txt[:512]
        if key not in seen:
            seen.add(key)
            ordered.append(txt)

    return ordered[:top_k]
