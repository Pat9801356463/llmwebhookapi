# engine/pinecone_handler.py
from typing import List, Tuple
import numpy as np
from pinecone import Pinecone, ServerlessSpec
from config import Config
from engine.db import save_chunks_to_db, fetch_chunks_from_db
from engine.embedding_handler import chunk_text, embed_chunks

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
    """Ensure embeddings have shape (_, EMBEDDING_DIM) by padding or truncating; returns float32."""
    if embeddings is None or embeddings.size == 0:
        return np.zeros((0, Config.EMBEDDING_DIM), dtype=np.float32)
    if embeddings.ndim == 1:
        embeddings = np.expand_dims(embeddings, 0)
    cur = embeddings.shape[1]
    target = Config.EMBEDDING_DIM
    if cur == target:
        return embeddings.astype(np.float32)
    if cur < target:
        pad = np.zeros((embeddings.shape[0], target - cur), dtype=np.float32)
        return np.concatenate([embeddings.astype(np.float32), pad], axis=1)
    # cur > target
    return embeddings[:, :target].astype(np.float32)


def process_and_index_document(doc_id: str, doc_type: str, text: str, source: str = None) -> Tuple[List[str], np.ndarray]:
    """
    Chunk, embed, save to DB, and index in Pinecone.
    Returns (chunks, embeddings).
    """
    # 1) Load chunks/embeddings from DB if available; else generate
    existing = fetch_chunks_from_db(doc_id)
    if existing:
        chunks = [c["text"] for c in existing if c.get("text")]
        emb_list = []
        for c in existing:
            arr = np.array(c.get("embedding") or [], dtype=np.float32)
            if arr.size == 0:
                # Ensure we have a vector; use zeros as placeholder
                arr = np.zeros((Config.EMBEDDING_DIM,), dtype=np.float32)
            emb_list.append(arr)
        embeddings = np.vstack(emb_list) if emb_list else np.zeros((0, Config.EMBEDDING_DIM), dtype=np.float32)
    else:
        chunks = chunk_text(text, Config.CHUNK_SIZE, Config.OVERLAP_SIZE)
        if not chunks:
            print(f"[warn] No chunks produced for doc {doc_id}")
            return [], np.zeros((0, Config.EMBEDDING_DIM), dtype=np.float32)

        embeddings = embed_chunks(chunks)
        embeddings = _pad_or_check_embeddings(embeddings)

        # Persist to DB (best-effort)
        try:
            save_chunks_to_db(doc_id, doc_type, chunks, embeddings, source=source)
        except Exception as e:
            print(f"[warn] Failed to save chunks to DB for {doc_id}: {e}")

    # 2) Upsert into Pinecone with proper metadata
    vectors = []
    for i in range(len(chunks)):
        vec = embeddings[i].tolist() if embeddings.shape[0] > i else [0.0] * Config.EMBEDDING_DIM
        vectors.append({
            "id": f"{doc_id}-{i}",
            "values": vec,
            "metadata": {
                "text": chunks[i],
                "doc_id": doc_id,
                "doc_type": doc_type,
                "source": source or doc_id,
                "chunk_id": i,
            },
        })

    try:
        _pc_index.upsert(vectors=vectors, namespace=doc_id)
        print(f"[info] Upserted {len(vectors)} vectors into Pinecone ns='{doc_id}'")
    except Exception as e:
        print(f"[WARN] Pinecone upsert failed for doc {doc_id}: {e}")

    return chunks, embeddings


def retrieve_top_chunks(query: str, doc_id: str, top_k: int = 3) -> List[str]:
    """
    Query Pinecone and return the top-k chunk texts.
    We request include_values=True to enable optional local re-ranking.
    If values are missing/empty, we safely fall back to Pinecone's scores.
    If Pinecone returns no matches at all, we fall back to DB chunks as a last resort.
    """
    if not query or not doc_id:
        return []

    # Embed query
    q_vec = embed_chunks([query])
    q_vec = _pad_or_check_embeddings(q_vec)
    if q_vec.size == 0:
        print(f"[warn] Query embedding is empty for query='{query}'")
        return []
    q_vec = q_vec[0]

    # Query Pinecone
    try:
        res = _pc_index.query(
            vector=q_vec.tolist(),
            top_k=max(top_k * 2, 6),     # fetch more to allow local re-rank
            include_metadata=True,
            include_values=True,          # critical: get values for rerank
            namespace=doc_id
        )
    except Exception as e:
        print(f"[WARN] Pinecone query failed: {e}")
        res = None

    # Robust match extraction
    matches = []
    if res is not None:
        try:
            matches = getattr(res, "matches", None)
            if matches is None:
                # Some client returns dict-like
                matches = res.get("matches", [])  # type: ignore
        except Exception:
            matches = []

    if not matches:
        print(f"[info] No Pinecone matches for ns='{doc_id}' → attempting DB fallback.")
        # ---- DB fallback: use top_k first chunks so LLM gets context ----
        try:
            rows = fetch_chunks_from_db(doc_id) or []
            fallback_texts = [r["text"] for r in rows if r.get("text")]
            if fallback_texts:
                print(f"[info] DB fallback retrieved {len(fallback_texts)} texts; returning first {top_k}.")
                return fallback_texts[:top_k]
        except Exception as e:
            print(f"[warn] DB fallback retrieval failed: {e}")
        return []

    # Collect candidates: (text, values or None, score)
    candidates = []
    for m in matches:
        md = (m.get("metadata") or {}) if isinstance(m, dict) else getattr(m, "metadata", {}) or {}
        txt = md.get("text")
        if not txt:
            continue
        # values can be a list or missing; handle both
        raw_vals = (m.get("values") if isinstance(m, dict) else getattr(m, "values", None))
        vals = None
        if raw_vals is not None:
            try:
                vals = np.array(raw_vals, dtype=np.float32)
            except Exception:
                vals = None
        score = (m.get("score", 0.0) if isinstance(m, dict) else getattr(m, "score", 0.0)) or 0.0
        candidates.append((txt, vals, score))

    if not candidates:
        print(f"[info] No candidates extracted from matches for ns='{doc_id}' → DB fallback.")
        try:
            rows = fetch_chunks_from_db(doc_id) or []
            fallback_texts = [r["text"] for r in rows if r.get("text")]
            if fallback_texts:
                print(f"[info] DB fallback retrieved {len(fallback_texts)} texts; returning first {top_k}.")
                return fallback_texts[:top_k]
        except Exception as e:
            print(f"[warn] DB fallback retrieval failed: {e}")
        return []

    # Try local rerank if we actually have vectors; otherwise, use Pinecone's score
    have_any_values = any((v is not None and np.size(v) > 0) for _, v, _ in candidates)

    if have_any_values:
        # Local cosine re-rank (robust to size mismatch via padding)
        def cosine(a, b):
            b = _pad_or_check_embeddings(np.array(b, dtype=np.float32))
            if b.size == 0:
                return -1.0
            b = b[0]
            denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
            return float(np.dot(a, b) / denom)

        ranked = sorted(
            candidates,
            key=lambda x: cosine(q_vec, x[1]),
            reverse=True
        )
        top = [text for (text, _, __) in ranked[:top_k]]
        return top

    # Fallback: Use Pinecone scores directly (no values returned)
    ranked_by_score = sorted(candidates, key=lambda x: x[2], reverse=True)
    top_by_score = [text for (text, _, __) in ranked_by_score[:top_k]]
    return top_by_score
