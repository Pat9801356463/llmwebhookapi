# startup.py
from engine import faiss_handler, db

def main():
    # 1) Ensure DB is reachable
    try:
        db.test_connection()
        print("✅ Connected to Postgres successfully.")
    except Exception as e:
        print(f"❌ Failed to connect to Postgres: {e}")
        return

    # 2) Ensure tables exist
    try:
        db.create_tables()
        print("✅ Tables ensured.")
    except Exception as e:
        print(f"❌ Failed to ensure tables: {e}")

    # 3) Attempt to rebuild FAISS from DB (doc_id='default')
    try:
        if faiss_handler.rebuild_faiss_from_db("default"):
            print("✅ FAISS index loaded into memory at startup.")
        else:
            print("⚠️ No stored FAISS data found for doc_id='default'. You may need to ingest a PDF once.")
    except Exception as e:
        print(f"❌ Error rebuilding FAISS index from DB: {e}")

if __name__ == "__main__":
    main()
