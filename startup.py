# startup.py
from engine import db
from engine import faiss_handler
from config import Config

def main():
    print("🚀 Running startup tasks...")

    # 1. Ensure database tables exist
    print("📦 Ensuring Postgres tables...")
    db.create_tables()
    print("✅ Tables ensured.")

    # 2. Try loading FAISS index from Postgres (if chunks exist)
    try:
        print("📂 Checking for stored chunks in Postgres...")
        chunks_data = db.fetch_chunks_from_db(doc_id="default")
        if chunks_data:
            print(f"📄 Found {len(chunks_data)} chunks — rebuilding FAISS index in memory...")
            texts = [c["text"] for c in chunks_data]
            embeddings = [c["embedding"] for c in chunks_data]
            faiss_handler._CHUNKS = texts
            dim = len(embeddings[0])
            index = faiss_handler.faiss.IndexFlatL2(dim)
            index.add(faiss_handler.np.array(embeddings))
            faiss_handler._FAISS_INDEX = index
            print("✅ FAISS index loaded into memory.")
        else:
            print("⚠️ No stored chunks found — FAISS will be empty until ingestion.")
    except Exception as e:
        print(f"❌ Failed to load FAISS from DB: {e}")

    print("🏁 Startup complete.")

if __name__ == "__main__":
    main()
