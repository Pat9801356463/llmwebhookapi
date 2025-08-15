# engine/embedding_handler.py
from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from config import Config

_EMBEDDER: Optional[SentenceTransformer] = None

def get_embedding_model() -> SentenceTransformer:
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = SentenceTransformer(Config.EMBEDDING_MODEL_NAME)
    return _EMBEDDER

def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
    chunk_size = chunk_size or Config.CHUNK_SIZE
    overlap = overlap or Config.OVERLAP_SIZE
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
    chunks = [c for c in chunks if isinstance(c, str) and c.strip()]
    if not chunks:
        print("[debug] Embedding Handler: No valid chunks to embed.")
        return np.zeros((0, Config.EMBEDDING_DIM), dtype=np.float32)
    model = get_embedding_model()
    print(f"[debug] Embedding {len(chunks)} chunks...")
    embs = model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    embs = np.asarray(embs, dtype=np.float32)
    if embs.ndim == 1:
        embs = np.expand_dims(embs, axis=0)
    cur_dim = embs.shape[1]
    target = Config.EMBEDDING_DIM
    if cur_dim == target:
        return embs
    if cur_dim < target:
        pad = np.zeros((embs.shape[0], target - cur_dim), dtype=np.float32)
        return np.concatenate([embs, pad], axis=1)
    return embs[:, :target]
