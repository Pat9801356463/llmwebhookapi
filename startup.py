# startup.py
import os
from engine import faiss_handler, db

def main():
    # 1. Ensure DB connection works
    try:
        db.test_connection()
        print("✅ Connected to Postgres successfully.")
    except Exception as e:
        print(f"❌ Failed to connect to Postgres: {e}")
        return

    # 2. Attempt to rebuild FAISS from DB
    try:
        if faiss_handler.rebuild_faiss_from_db("default"):
            print("✅ FAISS index loaded into memory at startup.")
        else:
            print("⚠️ No stored FAISS data found. You may need to ingest a PDF.")
    except Exception as e:
        print(f"❌ Error rebuilding FAISS index from DB: {e}")

if __name__ == "__main__":
    main()
