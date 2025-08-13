# startup.py
import sys
from engine import db
from engine import faiss_handler

print("🚀 Starting initialization...")

# 1. Ensure Postgres tables exist
print("📦 Ensuring database tables...")
db.create_tables()
print("✅ Tables ready.")

# 2. Fetch all chunks from DB
print("📥 Fetching all chunks + embeddings from Postgres...")
chunks_data = db.get_all_chunks_with_embeddings()

if not chunks_data:
    print("⚠️ No chunks found in DB. FAISS index will be empty until you ingest documents.")
    faiss_handler._FAISS_INDEX = None
    faiss_handler._CHUNKS = []
else:
    # 3. Build FAISS index in memory
    print(f"🧠 Building FAISS index with {len(chunks_data)} chunks...")
    import faiss
    import numpy as np

    embeddings = np.array([c["embedding"] for c in chunks_data], dtype=np.float32)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss_handler._FAISS_INDEX = index
    faiss_handler._CHUNKS = [c["text"] for c in chunks_data]
    print("✅ FAISS index built and ready.")

print("🚀 Initialization complete. Starting API server...")

