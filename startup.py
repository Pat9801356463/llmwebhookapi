# startup.py
from engine import db, faiss_handler

print("🚀 Starting initialization...")

print("📦 Ensuring database tables...")
db.create_tables()
print("✅ Tables ready.")

print("📥 Rebuilding FAISS index from DB...")
faiss_handler.rebuild_faiss_from_db()

print("🚀 Initialization complete. Starting API server...")
