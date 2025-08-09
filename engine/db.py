import psycopg2
from psycopg2.extras import Json
from typing import List, Dict, Any
from config import Config

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
            cur.execute("""
                CREATE TABLE IF NOT EXISTS indexed_chunks (
                    id SERIAL PRIMARY KEY,
                    source TEXT,
                    doc_type TEXT,
                    chunk_id INT,
                    text TEXT,
                    embedding VECTOR(384)
                );
            """)
        conn.commit()

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

def save_chunks_to_db(metadata_list: List[Dict[str, Any]], embeddings: Any):
    with get_connection() as conn:
        with conn.cursor() as cur:
            for meta, emb in zip(metadata_list, embeddings):
                cur.execute("""
                    INSERT INTO indexed_chunks (source, doc_type, chunk_id, text, embedding)
                    VALUES (%s, %s, %s, %s, %s);
                """, (
                    meta["source"],
                    meta.get("doc_type", "unknown"),
                    meta.get("chunk_id", 0),
                    meta["text"],
                    emb.tolist()
                ))
        conn.commit()

# 🧪 Manual Run
if __name__ == "__main__":
    create_tables()
    print("✅ Tables ensured.")
