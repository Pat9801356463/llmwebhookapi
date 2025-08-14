import os
import json
import numpy as np
from contextlib import contextmanager
import psycopg2
from pinecone import Pinecone
from config import Config

# ---------- POSTGRES CONNECTION ----------
DATABASE_URL = os.getenv("DATABASE_URL")

@contextmanager
def get_db_connection():
    """Context manager to connect to PostgreSQL."""
    conn = psycopg2.connect(DATABASE_URL)
    try:
        yield conn
    finally:
        conn.close()

# ---------- PINECONE CONNECTION ----------
pc = Pinecone(api_key=Config.PINECONE_API_KEY)
INDEX_NAME = "hack"  # Your new 384-d index
index = pc.Index(INDEX_NAME)

# Auto fetch the index dimension from Pinecone metadata
_index_info = pc.describe_index(INDEX_NAME)
PINECONE_DIM = _index_info.dimension

# ---------- SAVE CHUNKS TO POSTGRES ----------
def save_chunks_to_db(doc_id, chunks):
    """
    Save a list of chunks to PostgreSQL.
    Each chunk: (chunk_id, content, embedding).
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            for chunk_id, content, embedding in chunks:
                cur.execute("""
                    INSERT INTO chunks (doc_id, chunk_id, content, embedding)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (doc_id, chunk_id)
                    DO UPDATE SET content = EXCLUDED.content,
                                  embedding = EXCLUDED.embedding;
                """, (doc_id, chunk_id, content, json.dumps(embedding)))
        conn.commit()

# ---------- UPSERT TO PINECONE ----------
def upsert_to_pinecone(vectors):
    """
    Upsert a batch of vectors to Pinecone.
    Vectors format: [(id, embedding, metadata), ...]
    """
    for vid, emb, meta in vectors:
        if len(emb) != PINECONE_DIM:
            raise ValueError(
                f"Embedding dimension mismatch: expected {PINECONE_DIM}, got {len(emb)}"
            )
    index.upsert(vectors=vectors)

# ---------- FETCH CHUNKS FROM POSTGRES ----------
def fetch_chunks_from_db(doc_id):
    """Fetch all chunks for a given document ID."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT chunk_id, content, embedding
                FROM chunks
                WHERE doc_id = %s;
            """, (doc_id,))
            rows = cur.fetchall()

    # Convert embedding from stored JSON/text to list
    processed_rows = []
    for chunk_id, content, embedding in rows:
        if isinstance(embedding, str):
            try:
                embedding = json.loads(embedding)
            except json.JSONDecodeError:
                pass
        elif isinstance(embedding, memoryview):
            embedding = np.frombuffer(embedding.tobytes(), dtype=np.float32).tolist()
        processed_rows.append((chunk_id, content, embedding))
    return processed_rows

# ---------- PINECONE QUERY ----------
def query_pinecone(vector, top_k=5):
    """Query Pinecone with an embedding vector."""
    if len(vector) != PINECONE_DIM:
        raise ValueError(
            f"Query vector dimension mismatch: expected {PINECONE_DIM}, got {len(vector)}"
        )
    return index.query(vector=vector, top_k=top_k, include_metadata=True)

# ---------- GET ALL DOC IDS ----------
def get_all_doc_ids():
    """Fetch all distinct doc_ids from PostgreSQL."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT doc_id FROM chunks;")
            rows = cur.fetchall()
            return [row[0] for row in rows]

# ---------- DELETE DOCUMENT ----------
def delete_document(doc_id):
    """Delete all chunks for a given document ID from Postgres."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM chunks WHERE doc_id = %s;", (doc_id,))
        conn.commit()
