# engine/db.py
import os
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import psycopg2
import psycopg2.extras
from psycopg2.extras import Json, execute_values

from config import Config


# ------------------------------
# Connection helpers
# ------------------------------
def _dsn_from_config() -> str:
    """
    Build a connection DSN from Config, unless DATABASE_URL is present.
    Railway often provides DATABASE_URL; we prefer that if available.
    """
    if os.getenv("DATABASE_URL"):
        return os.getenv("DATABASE_URL")  # Full URL, e.g., postgres://user:pass@host:port/db
    return f"postgresql://{Config.DB_USER}:{Config.DB_PASSWORD}@{Config.DB_HOST}:{Config.DB_PORT}/{Config.DB_NAME}"


def get_connection():
    """
    Returns a new psycopg2 connection. Caller is responsible for closing it.
    """
    dsn = _dsn_from_config()
    return psycopg2.connect(dsn)


def test_connection():
    """Lightweight connectivity check."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1;")
            cur.fetchone()


def get_embedding_dim() -> int:
    """Return embedding dimension for configured model (MiniLM-L6-v2 = 384)."""
    return int(getattr(Config, "EMBEDDING_DIM", 384))


# ------------------------------
# Table creation
# ------------------------------
def create_tables():
    """
    Creates required tables if not present:
      - queries
      - indexed_chunks (with unique constraint on (doc_id, chunk_id))
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            # Query logs
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

            # Indexed chunks + embeddings (stored as BYTEA)
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS indexed_chunks (
                    id SERIAL PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    source TEXT,
                    doc_type TEXT,
                    chunk_id INT NOT NULL,
                    text TEXT,
                    embedding BYTEA,
                    UNIQUE (doc_id, chunk_id)
                );
                """
            )

            # Helpful index for queries on doc_id
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_indexed_chunks_docid ON indexed_chunks(doc_id);"
            )
        conn.commit()


# ------------------------------
# Logging user queries
# ------------------------------
def log_user_query(session_id: str, user_query: str, reasoning_result: Dict[str, Any]):
    """
    Insert a query + reasoning record into 'queries'.
    Safe to call even if some fields are missing in reasoning_result.
    """
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


# ------------------------------
# Chunk storage + retrieval
# ------------------------------
def save_chunks_to_db(
    doc_id: str,
    doc_type: str,
    chunks: List[str],
    embeddings: np.ndarray,
    source: Optional[str] = None,
):
    """
    Store chunks + embeddings in Postgres (BYTEA format). Idempotent:
    - Uses UNIQUE(doc_id, chunk_id) constraint.
    - Upserts on conflict, updating text/source/doc_type/embedding.
    """
    if len(chunks) == 0:
        return
    if len(chunks) != embeddings.shape[0]:
        raise ValueError("chunks and embeddings length mismatch")

    # Build rows for efficient batch upsert
    values: List[Tuple] = []
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        emb_bytes = np.asarray(emb, dtype=np.float32).tobytes()
        values.append((doc_id, source or doc_id, doc_type, i, chunk, emb_bytes))

    with get_connection() as conn:
        with conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO indexed_chunks (
                    doc_id, source, doc_type, chunk_id, text, embedding
                ) VALUES %s
                ON CONFLICT (doc_id, chunk_id) DO UPDATE SET
                    source = EXCLUDED.source,
                    doc_type = EXCLUDED.doc_type,
                    text    = EXCLUDED.text,
                    embedding = EXCLUDED.embedding;
                """,
                values,
                page_size=1000,
            )
        conn.commit()


def fetch_chunks_from_db(doc_id: str) -> List[Dict[str, Any]]:
    """
    Fetch chunks + embeddings for a specific doc_id, ordered by chunk_id ASC.
    Returns a list of dicts:
        [{ "text": str, "embedding": np.ndarray }, ...]
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
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

    out: List[Dict[str, Any]] = []
    for text, emb in rows:
        # psycopg2 returns memoryview for BYTEA -> convert to bytes -> np.frombuffer
        if isinstance(emb, memoryview):
            emb = emb.tobytes()
        vec = np.frombuffer(emb, dtype=np.float32)
        out.append({"text": text, "embedding": vec})
    return out


# Optional: if you ever want to bulk fetch everything (not used by current flow)
def fetch_all_chunks() -> List[Dict[str, Any]]:
    """
    Fetch every chunk from the table (for debugging/ops).
    Returns: [{ "doc_id": str, "chunk_id": int, "text": str, "embedding": np.ndarray }, ...]
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT doc_id, chunk_id, text, embedding
                FROM indexed_chunks
                ORDER BY doc_id ASC, chunk_id ASC;
                """
            )
            rows = cur.fetchall()

    results: List[Dict[str, Any]] = []
    for doc_id, chunk_id, text, emb in rows:
        if isinstance(emb, memoryview):
            emb = emb.tobytes()
        vec = np.frombuffer(emb, dtype=np.float32)
        results.append(
            {"doc_id": doc_id, "chunk_id": chunk_id, "text": text, "embedding": vec}
        )
    return results


if __name__ == "__main__":
    create_tables()
    print("✅ Tables ensured.")

