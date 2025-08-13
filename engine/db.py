# engine/db.py
import psycopg2
from psycopg2.extras import Json, DictCursor
from typing import List, Dict, Any, Optional
from config import Config
import numpy as np

def get_connection():
    return psycopg2.connect(
        dbname=Config.DB_NAME,
        user=Config.DB_USER,
        password=Config.DB_PASSWORD,
        host=Config.DB_HOST,
        port=Config.DB_PORT,
    )

def test_connection():
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1;")
            cur.fetchone()

def get_embedding_dim():
    return Config.EMBEDDING_DIM

def create_tables():
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
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
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS indexed_chunks (
                    id SERIAL PRIMARY KEY,
                    doc_id TEXT,
                    source TEXT,
                    doc_type TEXT,
                    chunk_id INT,
                    text TEXT,
                    embedding BYTEA,
                    UNIQUE (doc_id, chunk_id)
                );
                """
            )
        conn.commit()

def log_user_query(session_id: str, user_query: str, reasoning_result: Dict[str, Any]):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO queries (
                    session_id, user_query, parsed_query,
                    decision, amount, justification, matched_clauses
                ) VALUES (%s, %s, %s, %s, %s, %s, %s);
                """,
                (
                    session_id,
                    user_query,
                    Json(reasoning_result.get("parsed", {})),
                    reasoning_result.get("decision", "unknown"),
                    reasoning_result.get("amount", reasoning_result.get("payout_amount")),
                    reasoning_result.get("justification"),
                    Json(reasoning_result.get("matched_clauses", [])),
                ),
            )
        conn.commit()

def save_chunks_to_db(
    doc_id: str, doc_type: str, chunks: List[str], embeddings: np.ndarray, source: Optional[str] = None
):
    with get_connection() as conn:
        with conn.cursor() as cur:
            for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                emb_bytes = np.asarray(emb, dtype=np.float32).tobytes()
                cur.execute(
                    """
                    INSERT INTO indexed_chunks (doc_id, source, doc_type, chunk_id, text, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (doc_id, chunk_id) DO UPDATE SET
                      text = EXCLUDED.text,
                      embedding = EXCLUDED.embedding,
                      source = EXCLUDED.source;
                    """,
                    (doc_id, source or doc_id, doc_type, i, chunk, psycopg2.Binary(emb_bytes)),
                )
        conn.commit()

def fetch_chunks_from_db(doc_id: str) -> List[Dict[str, Any]]:
    with get_connection() as conn:
        with conn.cursor(cursor_factory=DictCursor) as cur:
            cur.execute(
                """
                SELECT text, embedding
                FROM indexed_chunks
                WHERE doc_id = %s
                ORDER BY chunk_id ASC;
                """,
                (doc_id,),
            )
            rows = cur.fetchall()
    if not rows:
        return []
    return [
        {"text": row["text"], "embedding": np.frombuffer(bytes(row["embedding"]), dtype=np.float32)}
        for row in rows
    ]

def get_all_doc_ids() -> List[str]:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT doc_id FROM indexed_chunks;")
            rows = cur.fetchall()
    return [r[0] for r in rows if r]
