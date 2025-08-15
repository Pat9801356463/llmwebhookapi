from typing import List, Tuple
import numpy as np
from pinecone import Pinecone, ServerlessSpec
from config import Config
from engine.db import save_chunks_to_db, fetch_chunks_from_db
from engine.embedding_handler import chunk_text, embed_chunks

_pc_client = Pinecone(api_key=Config.PINECONE_API_KEY)
_indexes = {i["name"]: i for i in _pc_client.list_indexes()}
if Config.PINECONE_INDEX_NAME not in _indexes:
    _pc_client.create_index(
        name=Config.PINECONE_INDEX_NAME,
        dimension=Config.EMBEDDING_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
_pc_index = _pc_client.Index(Config.PINECONE_INDEX_NAME)

def _pad_or_check_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Ensure embeddings have shape (_, EMBEDDING_DIM) by padding or truncating."""
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
    return embeddings[:, :target].astype(np.float32)

def process_and_index_document(doc_id: str, doc_type: str, text: str, source: str = None) -> Tuple[List[str], np.ndarray]:
    """Chunk, embed, save to DB, and index in Pinecone."""
    existing = fetch_chunks_from_db(doc_id)
    if existing:
        chunks = [c["text"] for c in existing]
        embeddings = np.vstack([np.array(c["embedding"], dtype=np.float32) for c in existing])
    else:
        chunks = chunk_text(text, Config.CHUNK_SIZE, Config.OVERLAP_SIZE)
        if not chunks:
            return [], np.zeros((0, Config.EMBEDDING_DIM), dtype=np.float32)
        embeddings = embed_chunks(chunks)
        embeddings = _pad_or_check_embeddings(embeddings)
        save_chunks_to_db(doc_id, doc_type, chunks, embeddings, source=source)

    vectors = [
        {
            "id": f"{doc_id}-{i}",
            "values": embeddings[i].tolist(),
            "metadata": {
                "text": chunks[i],
                "doc_id": doc_id,
                "doc_type": doc_type,
                "source": source or doc_id,
                "chunk_id": i,
            },
        }
        for i in range(len(chunks))
    ]

    try:
        _pc_index.upsert(vectors=vectors, namespace=doc_id)
    except Exception as e:
        print(f"[WARN] Pinecone upsert failed for doc {doc_id}: {e}")

    return chunks, embeddings

def retrieve_top_chunks(query: str, doc_id: str, top_k: int = 3) -> List[str]:
    """Query Pinecone, then rerank locally for better accuracy."""
    if not query or not doc_id:
        return []
    q_vec = embed_chunks([query])
    q_vec = _pad_or_check_embeddings(q_vec)
    if q_vec.size == 0:
        print(f"[warn] Query embedding is empty for query='{query}'")
        return []
    q_vec = q_vec[0]

    try:
        res = _pc_index.query(
            vector=q_vec.tolist(),
            top_k=max(top_k * 2, 6),
            include_metadata=True,
            namespace=doc_id
        )
    except Exception as e:
        print(f"[WARN] Pinecone query failed: {e}")
        return []

    matches = getattr(res, "matches", None) or res.get("matches", [])
    candidates = []
    for idx, m in enumerate(matches):
        txt = m.get("metadata", {}).get("text")
        vals = np.array(m.get("values") or [], dtype=np.float32)
        if not txt:
            continue
        if vals.size == 0:
            print(f"[warn] Skipping empty embedding from Pinecone match {idx}")
            continue
        candidates.append((txt, vals))

    if not candidates:
        return []

    def cosine(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

    ranked = sorted(
        candidates,
        key=lambda x: cosine(q_vec, _pad_or_check_embeddings(x[1])[0]),
        reverse=True
    )
    return [text for text, _ in ranked[:top_k]]
