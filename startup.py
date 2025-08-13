# startup.py
from engine import faiss_handler, db
from engine.pinecone_handler import process_and_index_document
from engine.db import get_all_doc_ids, fetch_chunks_from_db

def main():
    try:
        db.test_connection()
        print("✅ Connected to Postgres successfully.")
    except Exception as e:
        print(f"❌ Failed to connect to Postgres: {e}")
        return

    try:
        db.create_tables()
        print("✅ Tables ensured.")
    except Exception as e:
        print(f"❌ Failed to ensure tables: {e}")

    # Rebuild FAISS for 'default' if present
    try:
        if faiss_handler.rebuild_faiss_from_db("default"):
            print("✅ FAISS index loaded into memory at startup.")
        else:
            print("⚠️ No stored FAISS data found for doc_id='default'.")
    except Exception as e:
        print(f"❌ Error rebuilding FAISS index from DB: {e}")

    # Optionally re-upsert all docs to Pinecone (safe, idempotent)
    try:
        doc_ids = get_all_doc_ids()
        for doc_id in doc_ids:
            chunks = fetch_chunks_from_db(doc_id)
            if not chunks:
                continue
            # Reconstruct text from chunks for idempotent process_and_index_document
            text = " ".join([c["text"] for c in chunks])
            process_and_index_document(doc_id, "policy", text, source=doc_id)
        print("✅ Re-upsert to Pinecone attempted for stored docs.")
    except Exception as e:
        print(f"⚠️ Skipping Pinecone re-upsert at startup: {e}")

if __name__ == "__main__":
    main()
