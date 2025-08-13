from typing import List, Tuple, Optional
import numpy as np

from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

from config import Config
from engine.db import save_chunks_to_db, fetch_chunks_from_db


# --- Embedding model (lazy) ---
_EMBEDDER: Optional[SentenceTransformer] = None


def get_embedding_model() -> SentenceTransformer:
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = SentenceTransformer(Config.EMBEDDING_MODEL_NAME)
    return _EMBEDDER


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Split text into overlapping chunks (word-based)."""
    tokens = text.split()
    if not tokens:
        return []
    chunks: List[str] = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(tokens), step):
        chunk = " ".join(tokens[i:i + chunk_size]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def embed_chunks(chunks: List[str]) -> np.ndarray:
    """Generate dense embeddings for chunks."""
    model = get_embedding_model()
    return model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)


# --- Pinecone init / index helper ---
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
            spec=ServerlessSpec(cloud="aws", region=Config.PINECONE_ENV.split("-")[0] if "-" in Config.PINECONE_ENV else "us-east-1")
        )
    return pc.Index(Config.PINECONE_INDEX_NAME)


# --- Public API ---
def process_and_index_document(doc_id: str, doc_type: str, text: str, source: str = None) -> Tuple[List[str], np.ndarray]:
    """
    Idempotent ingestion path:
    1) If chunks+embeddings exist in Postgres, reuse them.
    2) Else chunk+embed, persist to Postgres.
    3) Upsert all vectors to Pinecone (safe to repeat).
    """
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
        save_chunks_to_db(doc_id, doc_type, chunks, embeddings, source=source)

    # Upsert to Pinecone (namespace per doc)
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
    # Use namespace=doc_id to isolate per-document for faster filtered queries
    index.upsert(vectors=vectors, namespace=doc_id)
    return chunks, embeddings


def retrieve_top_chunks(query: str, doc_id: str, top_k: int = 3) -> List[str]:
    """
    Query Pinecone for top-K similar chunks within the document namespace.
    """
    if not query or not doc_id:
        return []

    index = _ensure_index()
    q_vec = embed_chunks([query])[0]
    res = index.query(
        vector=q_vec.tolist(),
        top_k=top_k,
        include_metadata=True,
        namespace=doc_id
    )

    # pinecone-client v3 returns a dict-like object with "matches"
    matches = getattr(res, "matches", None) or res.get("matches", [])
    texts: List[str] = []
    for m in matches:
        md = getattr(m, "metadata", None) or m.get("metadata", {})
        if md and "text" in md:
            texts.append(md["text"])
    return texts
