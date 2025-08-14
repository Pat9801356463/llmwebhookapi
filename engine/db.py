# engine/db.py
import os
import datetime
import psycopg2
from psycopg2.extras import RealDictCursor
from pinecone import Pinecone, ServerlessSpec

# -----------------------------
# PostgreSQL Connection (Optional Logging)
# -----------------------------
PG_CONN = None

def get_pg_conn():
    """Connect to Postgres if DATABASE_URL is set."""
    global PG_CONN
    if PG_CONN is None and os.getenv("DATABASE_URL"):
        PG_CONN = psycopg2.connect(os.getenv("DATABASE_URL"), cursor_factory=RealDictCursor)
    return PG_CONN

def create_tables():
    """Create tables for logging queries and storing chunks."""
    conn = get_pg_conn()
    if not conn:
        print("⚠ No Postgres connection — skipping table creation")
        return
    with conn.cursor() as cur:
        # For query logs
        cur.execute("""
            CREATE TABLE IF NOT EXISTS user_queries (
                id SERIAL PRIMARY KEY,
                user_id TEXT,
                query TEXT,
                timestamp TIMESTAMP DEFAULT NOW()
            )
        """)
        # For chunk storage
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
    print("✅ Tables ensured in Postgres")

def log_user_query(user_id: str, query: str):
    """Log a query to Postgres."""
    conn = get_pg_conn()
    if not conn:
        print(f"⚠ No Postgres connection — not logging query: {query}")
        return
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO user_queries (user_id, query, timestamp) VALUES (%s, %s, %s)",
            (user_id, query, datetime.datetime.utcnow())
        )
        conn.commit()
    print(f"📝 Logged query for user {user_id}")

# -----------------------------
# Chunk Storage Helpers
# -----------------------------
def save_chunks_to_db(doc_id: str, doc_type: str, chunks, embeddings, source: str = None):
    """Persist chunks + embeddings to Postgres."""
    conn = get_pg_conn()
    if not conn:
        print("⚠ No Postgres connection — skipping chunk save")
        return
    with conn.cursor() as cur:
        for chunk, emb in zip(chunks, embeddings):
            cur.execute("""
                INSERT INTO document_chunks (doc_id, doc_type, chunk, embedding, source)
                VALUES (%s, %s, %s, %s, %s)
            """, (doc_id, doc_type, chunk, emb.tolist(), source))
        conn.commit()

def fetch_chunks_from_db(doc_id: str):
    """Fetch all chunks for a given document ID."""
    conn = get_pg_conn()
    if not conn:
        return []
    with conn.cursor() as cur:
        cur.execute("""
            SELECT chunk AS text, embedding
            FROM document_chunks
            WHERE doc_id = %s
            ORDER BY id ASC
        """, (doc_id,))
        return cur.fetchall()

def get_all_doc_ids():
    """Return all distinct document IDs."""
    conn = get_pg_conn()
    if not conn:
        return []
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT doc_id FROM document_chunks")
        rows = cur.fetchall()
        return [row["doc_id"] for row in rows]

# -----------------------------
# Pinecone Setup
# -----------------------------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "hack")  # default to your 'hack' index

pc = None
pinecone_index = None

def init_pinecone():
    """Initialize Pinecone client and connect to index."""
    global pc, pinecone_index
    if not PINECONE_API_KEY:
        print("⚠ No Pinecone API key set")
        return
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)

        # Ensure index exists
        if PINECONE_INDEX_NAME not in [idx["name"] for idx in pc.list_indexes()]:
            print(f"📌 Creating Pinecone index '{PINECONE_INDEX_NAME}' (384-dim)")
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )

        pinecone_index = pc.Index(PINECONE_INDEX_NAME)
        print(f"✅ Connected to Pinecone index '{PINECONE_INDEX_NAME}'")
    except Exception as e:
        print(f"❌ Pinecone init failed: {e}")

# Initialize on import
create_tables()
init_pinecone()
