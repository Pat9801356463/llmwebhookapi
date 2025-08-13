import os
import sys
import numpy as np

# Ensure app root is on sys.path
sys.path.append(os.path.dirname(__file__))

from engine.embedding_handler import upsert_to_pinecone
from db import fetch_chunks_from_db

def rebuild_pinecone_from_db():
    print("🚀 Starting Pinecone rebuild from Postgres...")

    chunks = fetch_chunks_from_db()
    if not chunks:
        print("⚠️ No chunks found in Postgres. Skipping Pinecone rebuild.")
        return

    vectors = []
    for chunk in chunks:
        chunk_id = chunk["id"] if isinstance(chunk, dict) else chunk[0]
        text = chunk["chunk"] if isinstance(chunk, dict) else chunk[1]
        embedding = chunk["embedding"] if isinstance(chunk, dict) else chunk[2]

        # Ensure embedding is numpy array
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding, dtype=np.float32)

        vectors.append((str(chunk_id), embedding.tolist(), {"text": text}))

    print(f"📦 Preparing to upsert {len(vectors)} vectors into Pinecone...")
    upsert_to_pinecone(vectors, namespace="default")
    print("✅ Pinecone rebuild complete.")

if __name__ == "__main__":
    rebuild_pinecone_from_db()

        print(f"⚠️ Skipping Pinecone re-upsert at startup: {e}")

if __name__ == "__main__":
    main()
