# startup.py
import sys
from engine import db, faiss_handler
import numpy as np
import faiss

def main():
    print("🚀 Startup script starting...")

    # 1. Ensure database tables exist
    try:
        db.create_tables()
        print("✅ Database tables ensured.")
    except Exception as e:
        print(f"❌ Failed to ensure tables: {e}")
        sys.exit(1)

    # 2. Ensure FAISS index file exists
    try:
        faiss_handler.ensure_faiss_index()
    except Exception as e:
        print(f"❌ Failed to ensure FAISS index file: {e}")

    # 3. Load all chunks + embeddings from Postgres
    try:
        records = db.get_all_chunks_with_embeddings()
        if not records:
            print("⚠️ No chunks found in Postgres. FAISS index will be empty until a PDF is ingested.")
            return

        chunks = [r["chunk"] for r in records]
        embeddings = np.array([np.frombuffer(r["embedding"], dtype=np.float32) for r in records])

        # Build FAISS index in memory
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)

        # Save in global variables
        faiss_handler._FAISS_INDEX = index
        faiss_handler._CHUNKS = chunks

        print(f"✅ Loaded {len(chunks)} chunks from Postgres into FAISS index.")
    except Exception as e:
        print(f"❌ Failed to load chunks from Postgres: {e}")


if __name__ == "__main__":
    main()
