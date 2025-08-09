import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from config import Config

# Load embedding model
MODEL = SentenceTransformer(Config.EMBEDDING_MODEL_NAME)

# Load FAISS index
INDEX = faiss.read_index(Config.FAISS_INDEX_PATH)

# Load metadata file (assumes same order as index)
with open("data/embeddings/chunk_metadata.json", "r", encoding="utf-8") as f:
    CHUNK_METADATA = json.load(f)

def embed_query(text: str) -> np.ndarray:
    return MODEL.encode([text])[0]

def retrieve_clauses(parsed_query: dict, top_k: int = 5):
    """
    Takes structured query dict and returns top-k matched clause chunks
    from the FAISS index with metadata.
    """
    # Join structured fields to form semantic query
    query_parts = [str(v) for v in parsed_query.values() if v]
    full_query = " ".join(query_parts)
    query_vector = embed_query(full_query)

    # Search FAISS
    distances, indices = INDEX.search(np.array([query_vector]), top_k)
    matched_chunks = []

    for idx in indices[0]:
        if idx < len(CHUNK_METADATA):
            meta = CHUNK_METADATA[idx]
            matched_chunks.append({
                "text": meta["text"],
                "source": meta["source"],
                "doc_type": meta.get("doc_type", "unknown"),
                "score": float(distances[0][np.where(indices[0] == idx)[0][0]])
            })

    return matched_chunks

# 🧪 Example
if __name__ == "__main__":
    example = {
        "age": 46,
        "procedure": "knee surgery",
        "location": "Pune",
        "policy_duration": "3 months"
    }
    results = retrieve_clauses(example)
    print("🔍 Top Retrieved Chunks:")
    for i, chunk in enumerate(results):
        print(f"\n--- Chunk #{i+1} from {chunk['source']} ---")
        print(chunk["text"][:300], "...")
