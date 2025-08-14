# engine/pinecone_handler.py
from typing import List, Tuple
import numpy as np
from pinecone import Pinecone, ServerlessSpec
from config import Config
from engine.db import save_chunks_to_db, fetch_chunks_from_db, get_all_doc_ids
from engine.embedding_handler import chunk_text, embed_chunks

# Pinecone client helper
def _pc() -> Pinecone:
    return Pinecone(api_key=Config.PINECONE_API_KEY)

def _ensure_index():
    pc = _pc()
    indexes = {i["name"]: i for i in pc.list_indexes()}
    if Config.PINECONE_INDEX_NAME not in indexes:
        pc.create_index(
            name=Config.PINECONE_INDEX_NAME,
            dimension=Config.EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    return pc.Index(Config.PINECONE_INDEX_NAME)

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
    index = _ensure_index()

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
        index.upsert(vectors=vectors, namespace=doc_id)
    except Exception as e:
        print(f"[WARN] Pinecone upsert failed for doc {doc_id}: {e}")

    return chunks, embeddings

def retrieve_top_chunks(query: str, doc_id: str, top_k: int = 3) -> List[str]:
    """Query Pinecone for top-K similar chunks."""
    if not query or not doc_id:
        return []
    index = _ensure_index()
    q_vec = embed_chunks([query])
    q_vec = _pad_or_check_embeddings(q_vec)[0]
    try:
        res = index.query(vector=q_vec.tolist(), top_k=top_k, include_metadata=True, namespace=doc_id)
    except Exception as e:
        print(f"[WARN] Pinecone query failed: {e}")
        return []
    matches = getattr(res, "matches", None) or res.get("matches", [])
    return [m.get("metadata", {}).get("text", "") for m in matches if m.get("metadata", {}).get("text")]
