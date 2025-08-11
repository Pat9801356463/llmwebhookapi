# engine/db.py
import psycopg2
from psycopg2.extras import Json
from typing import List, Dict, Any
from config import Config
import numpy as np

def get_connection():
    return psycopg2.connect(
        dbname=Config.DB_NAME,
        user=Config.DB_USER,
        password=Config.DB_PASSWORD,
        host=Config.DB_HOST,
        port=Config.DB_PORT
    )

def create_tables():
    with get_connection() as conn:
        with conn.cursor() as cur:
            # Enable pgvector extension
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            # Table for queries and results
            cur.execute("""
                CREATE TABLE IF NOT EXISTS queries (
                    id SERIAL PRIMARY KEY,
                    session_id TEXT,
                    user_query TEXT,
                    parsed_query JSONB,
                    decision TEXT,
                    amount NUMERIC,
                    justification TEXT,
                    matched_clauses JSONB,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # Table for document chunks and embeddings
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS indexed_chunks (
                    id SERIAL PRIMARY KEY,
                    doc_id TEXT,
                    source TEXT,
                    doc_type TEXT,
                    chunk_id INT,
                    text TEXT,
                    embedding VECTOR({get_embedding_dim()})
                );
            """)
        conn.commit()

def get_embedding_dim():
    """Return embedding dimension for configured model."""
    # MiniLM-L6-v2 => 384 dims
    # You can change here if using another model
    return 384

def log_user_query(session_id: str, user_query: str, reasoning_result: Dict[str, Any]):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO queries (
                    session_id, user_query, parsed_query,
                    decision, amount, justification, matched_clauses
                ) VALUES (%s, %s, %s, %s, %s, %s, %s);
            """, (
                session_id,
                user_query,
                Json(reasoning_result.get("parsed", {})),
                reasoning_result.get("decision", "unknown"),
                reasoning_result.get("amount"),
                reasoning_result.get("justification"),
                Json(reasoning_result.get("matched_clauses", []))
            ))
        conn.commit()

def save_chunks_to_db(doc_id: str, doc_type: str, chunks: List[str], embeddings: np.ndarray):
    with get_connection() as conn:
        with conn.cursor() as cur:
            for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                cur.execute("""
                    INSERT INTO indexed_chunks (doc_id, source, doc_type, chunk_id, text, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s);
                """, (
                    doc_id,
                    f"{doc_id}.source",  # source can be set as doc_id or actual filename
                    doc_type,
                    i,
                    chunk,
                    emb.tolist()  # pgvector accepts Python lists for VECTOR columns
                ))
        conn.commit()

def fetch_chunks_from_db(doc_id: str) -> List[Dict[str, Any]]:
    """Retrieve chunks and embeddings for a given document ID."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT text, embedding
                FROM indexed_chunks
                WHERE doc_id = %s
                ORDER BY chunk_id ASC;
            """, (doc_id,))
            rows = cur.fetchall()

    return [
        {"text": row[0], "embedding": np.array(row[1], dtype=np.float32)}
        for row in rows
    ] if rows else []

# 🧪 Manual Run
if __name__ == "__main__":
    create_tables()
    print("✅ Tables ensured.")
