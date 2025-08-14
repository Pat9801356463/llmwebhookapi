# engine/db.py
import os
import json
import numpy as np
import psycopg2
import psycopg2.extras
from contextlib import contextmanager


# ---------- CONNECTION ----------
DATABASE_URL = os.getenv("DATABASE_URL") or os.getenv("PG_CONN")


@contextmanager
def get_db_connection():
    """
    Context manager for PostgreSQL connection.
    Ensures connection is closed after use.
    """
    conn = psycopg2.connect(DATABASE_URL)
    try:
        yield conn
    finally:
        conn.close()


# ---------- SETUP ----------
def create_tables():
    """
    Create required tables if not exist:
      - chunks: stores per-document chunks + embeddings (BYTEA)
      - queries: optional; for logging user requests (safe no-op if unused)
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            # Chunks table (compatible with your older schema but extended)
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    doc_id   TEXT       NOT NULL,
                    chunk_id INTEGER    NOT NULL,
                    content  TEXT       NOT NULL,
                    embedding BYTEA     NOT NULL,
                    doc_type TEXT,
                    source   TEXT,
                    PRIMARY KEY (doc_id, chunk_id)
                );
                """
            )

            # Optional query logs (so app.reasoner/log_user_query don't crash)
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
        conn.commit()


def test_connection():
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1;")


# ---------- UPSERT CHUNKS ----------
def save_chunks_to_db(doc_id, doc_type, chunks, embeddings, source=None):
    """
    Save a list of text chunks + embeddings to the DB in the 'chunks' table.

    Args:
      doc_id (str): document identifier
      doc_type (str): e.g., 'policy'
      chunks (List[str]): text chunks
      embeddings (np.ndarray): shape (n, d) float32
      source (str|None): optional source URL/path
    """
    if embeddings is None or len(chunks) == 0:
        return

    # Ensure numpy array float32
    if not isinstance(embeddings, np.ndarray):
        embeddings = np.array(embeddings, dtype=np.float32)
    else:
        embeddings = embeddings.astype(np.float32)

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            for i, (content, emb) in enumerate(zip(chunks, embeddings)):
                emb_bytes = emb.tobytes()  # store as BYTEA
                cur.execute(
                    """
                    INSERT INTO chunks (doc_id, chunk_id, content, embedding, doc_type, source)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (doc_id, chunk_id)
                    DO UPDATE
                    SET content = EXCLUDED.content,
                        embedding = EXCLUDED.embedding,
                        doc_type = EXCLUDED.doc_type,
                        source   = EXCLUDED.source;
                    """,
                    (doc_id, i, content, psycopg2.Binary(emb_bytes), doc_type, source),
                )
        conn.commit()


# ---------- FETCH CHUNKS ----------
def fetch_chunks_from_db(doc_id):
    """
    Fetch all chunks for a given doc_id.
    Returns: List[Dict] with keys: 'text', 'embedding' (as list[float])
    """
    results = []
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT chunk_id, content, embedding
                FROM chunks
                WHERE doc_id = %s
                ORDER BY chunk_id ASC;
                """,
                (doc_id,),
            )
            rows = cur.fetchall()

    for row in rows:
        emb = row["embedding"]
        # psycopg2 returns memoryview for BYTEA
        if isinstance(emb, memoryview):
            emb = np.frombuffer(emb.tobytes(), dtype=np.float32)
        elif isinstance(emb, (bytes, bytearray)):
            emb = np.frombuffer(emb, dtype=np.float32)
        elif isinstance(emb, list):
            emb = np.array(emb, dtype=np.float32)
        else:
            # As a fallback, try JSON
            try:
                emb = np.array(json.loads(emb), dtype=np.float32)
            except Exception:
                emb = np.array([], dtype=np.float32)

        results.append(
            {
                "text": row["content"],            # <-- what pinecone_handler expects
                "embedding": emb.tolist(),         # make JSON/pinecone friendly
            }
        )
    return results


# ---------- UTILS ----------
def get_all_doc_ids():
    """
    Return all distinct doc_id values in the chunks table.
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT doc_id FROM chunks;")
            rows = cur.fetchall()
            return [r[0] for r in rows]


def delete_document(doc_id):
    """
    Remove all chunks for the given document from DB (does NOT touch Pinecone).
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM chunks WHERE doc_id = %s;", (doc_id,))
        conn.commit()


# ---------- OPTIONAL: LOG QUERIES ----------
def log_user_query(session_id, user_query, reasoning_result: dict):
    """
    Safe logger; won't crash if not used.
    """
    parsed = reasoning_result.get("parsed", {})
    decision = reasoning_result.get("decision", "unknown")
    amount = reasoning_result.get("amount", reasoning_result.get("payout_amount"))
    justification = reasoning_result.get("justification")
    matched = reasoning_result.get("matched_clauses", [])

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO queries (
                    session_id, user_query, parsed_query,
                    decision, amount, justification, matched_clauses
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s);
                """,
                (
                    session_id,
                    user_query,
                    json.dumps(parsed),
                    decision,
                    amount,
                    justification,
                    json.dumps(matched),
                ),
            )
        conn.commit()
