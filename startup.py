# startup.py
from engine import db
from engine import faiss_handler

print("📦 Running startup tasks...")

# Create Postgres tables
db.create_tables()
print("✅ Postgres tables ensured.")

# Initialize FAISS index (create file if it doesn't exist)
faiss_handler.ensure_faiss_index()
print("✅ FAISS index ready.")

print("🚀 Startup tasks complete.")
