import os
import psycopg2
from contextlib import contextmanager

# Get database URL from environment variable
DATABASE_URL = os.getenv("DATABASE_URL")

# ---------- CONNECTION HANDLER ----------
@contextmanager
def get_db_connection():
    """
    Context manager to connect to PostgreSQL.
    Auto-closes the connection after use.
    """
    conn = psycopg2.connect(DATABASE_URL)
    try:
        yield conn
    finally:
        conn.close()

# ---------- SAVE CHUNKS ----------
def save_chunks_to_db(doc_id, chunks):
    """
    Save a list of chunks to the database.
    Each chunk is a tuple: (chunk_id, content, embedding).
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
                """, (doc_id, chunk_id, content, embedding))
        conn.commit()

# ---------- FETCH CHUNKS ----------
def fetch_chunks_from_db(doc_id):
    """
    Fetch all chunks for a given document ID.
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT chunk_id, content, embedding
                FROM chunks
                WHERE doc_id = %s;
            """, (doc_id,))
            return cur.fetchall()

# ---------- GET ALL DOCUMENT IDS ----------
def get_all_doc_ids():
    """
    Fetch all distinct document IDs from the chunks table.
    Useful for Pinecone sync or debugging.
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT doc_id FROM chunks;")
            rows = cur.fetchall()
            return [row[0] for row in rows]

# ---------- DELETE DOCUMENT ----------
def delete_document(doc_id):
    """
    Delete all chunks for a given document ID.
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM chunks WHERE doc_id = %s;", (doc_id,))
        conn.commit()

