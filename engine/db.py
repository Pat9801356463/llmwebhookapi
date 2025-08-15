import os
import datetime
import psycopg2
from psycopg2.extras import RealDictCursor
from pinecone import Pinecone, ServerlessSpec

PG_CONN = None

def get_pg_conn():
    """Lazy connection to Postgres. Returns None if DB is unreachable."""
    global PG_CONN
    if PG_CONN is not None:
        return PG_CONN
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        print("⚠ No DATABASE_URL set — skipping Postgres connection.")
        return None
    try:
        PG_CONN = psycopg2.connect(dsn, cursor_factory=RealDictCursor)
        print("✅ Connected to Postgres.")
        return PG_CONN
    except psycopg2.OperationalError as e:
        print(f"❌ Postgres connection failed: {e}")
        PG_CONN = None
        return None

def create_tables():
    """Create tables if DB is reachable."""
    conn = get_pg_conn()
    if not conn:
        print("⚠ Skipping table creation — DB not reachable.")
        return
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS user_queries (
                    id SERIAL PRIMARY KEY,
                    user_id TEXT,
                    query TEXT,
                    timestamp TIMESTAMP DEFAULT NOW()
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id SERIAL PRIMARY KEY,
                    doc_id TEXT,
                    doc_type TEXT,
                    chunk TEXT,
                    embedding FLOAT8[],
                    source TEXT
                )
            """)
        conn.commit()
        print("✅ Tables ensured in Postgres.")
    except Exception as e:
        print(f"❌ Failed to create tables: {e}")

def log_user_query(user_id: str, query: str):
    conn = get_pg_conn()
    if not conn:
        print(f"⚠ Query not logged — DB not reachable: {query}")
        return
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO user_queries (user_id, query, timestamp) VALUES (%s, %s, %s)",
                (user_id, query, datetime.datetime.utcnow())
            )
        conn.commit()
    except Exception as e:
        print(f"❌ Failed to log query: {e}")

def save_chunks_to_db(doc_id: str, doc_type: str, chunks, embeddings, source: str = None):
    conn = get_pg_conn()
    if not conn:
        print("⚠ Skipping chunk save — DB not reachable.")
        return
    try:
        with conn.cursor() as cur:
            for chunk, emb in zip(chunks, embeddings):
                cur.execute("""
                    INSERT INTO document_chunks (doc_id, doc_type, chunk, embedding, source)
                    VALUES (%s, %s, %s, %s, %s)
                """, (doc_id, doc_type, chunk, emb.tolist(), source))
        conn.commit()
    except Exception as e:
        print(f"❌ Failed to save chunks: {e}")

def fetch_chunks_from_db(doc_id: str):
    conn = get_pg_conn()
    if not conn:
        return []
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT chunk AS text, embedding
                FROM document_chunks
                WHERE doc_id = %s
                ORDER BY id ASC
            """, (doc_id,))
            return cur.fetchall()
    except Exception as e:
        print(f"❌ Failed to fetch chunks: {e}")
        return []

def get_all_doc_ids():
    conn = get_pg_conn()
    if not conn:
        return []
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT doc_id FROM document_chunks")
            rows = cur.fetchall()
        return [row["doc_id"] for row in rows]
    except Exception as e:
        print(f"❌ Failed to get doc IDs: {e}")
        return []

# Pinecone setup
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "hack")

pc = None
pinecone_index = None

def init_pinecone():
    """Initialize Pinecone safely."""
    global pc, pinecone_index
    if not PINECONE_API_KEY:
        print("⚠ No Pinecone API key set.")
        return
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        if PINECONE_INDEX_NAME not in [idx["name"] for idx in pc.list_indexes()]:
            print(f"📌 Creating Pinecone index '{PINECONE_INDEX_NAME}' (384-dim)")
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        pinecone_index = pc.Index(PINECONE_INDEX_NAME)
        print(f"✅ Connected to Pinecone index '{PINECONE_INDEX_NAME}'.")
    except Exception as e:
        print(f"❌ Pinecone init failed: {e}")


