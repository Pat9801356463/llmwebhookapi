import os
import pinecone
import numpy as np
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
from config import Config
from engine.db import save_chunks_to_db, fetch_chunks_from_db
from utils.pdf_loader import extract_text_from_pdf  # We'll create this helper if not already present

# === Lazy load embedding model ===
EMBEDDING_MODEL = None

def get_embedding_model():
    """Load the embedding model only once."""
    global EMBEDDING_MODEL
    if EMBEDDING_MODEL is None:
        EMBEDDING_MODEL = SentenceTransformer(Config.EMBEDDING_MODEL_NAME)
    return EMBEDDING_MODEL

# === Initialize Pinecone ===
pinecone.init(api_key=os.environ["PINECONE_API_KEY"], environment=os.environ["PINECONE_ENV"])
INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "policy-llm-index")

if INDEX_NAME not in pinecone.list_indexes():
    pinecone.create_index(
        name=INDEX_NAME,
        dimension=768,  # adjust if using different embedding model
        metric="cosine"
    )
index = pinecone.Index(INDEX_NAME)


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """Split text into overlapping chunks."""
    tokens = text.split()
    chunks = []
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk = " ".join(tokens[i:i + chunk_size]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def embed_chunks(chunks: List[str]) -> np.ndarray:
    """Generate embeddings for chunks."""
    model = get_embedding_model()
    return model.encode(chunks, convert_to_numpy=True)


def process_pdf_and_store(doc_id: str, pdf_path: str, source: str = None) -> Tuple[List[str], np.ndarray]:
    """Extract text from PDF → chunk → embed → save to DB + Pinecone."""
    existing = fetch_chunks_from_db(doc_id)
    if existing:
        chunks = [c["text"] for c in existing]
        embeddings = np.vstack([c["embedding"] for c in existing])
        return chunks, embeddings

    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text, Config.CHUNK_SIZE, Config.OVERLAP_SIZE)
    embeddings = embed_chunks(chunks)

    # Save to Postgres for persistence
    save_chunks_to_db(doc_id, "pdf", chunks, embeddings, source=source)

    # Save to Pinecone for fast vector search
    pinecone_vectors = [(f"{doc_id}_{i}", emb.tolist(), {"text": chunk}) for i, (chunk, emb) in enumerate(zip(chunks, embeddings))]
    index.upsert(vectors=pinecone_vectors)

    return chunks, embeddings


def retrieve_top_chunks(query: str, doc_id: str = None, top_k: int = 3) -> List[str]:
    """Retrieve top-K chunks using Pinecone."""
    query_vec = embed_chunks([query])[0]  # shape: (768,)
    filter_dict = {"doc_id": {"$eq": doc_id}} if doc_id else None

    results = index.query(vector=query_vec.tolist(), top_k=top_k, include_metadata=True, filter=filter_dict)
    return [match["metadata"]["text"] for match in results["matches"]]


def process_pdf_and_queries(doc_id: str, pdf_path: str, queries: List[str]) -> dict:
    """
    Process PDF once, then answer multiple queries.
    Returns a dictionary: {query: [top_chunks]}
    """
    process_pdf_and_store(doc_id, pdf_path)
    answers = {}
    for q in queries:
        answers[q] = retrieve_top_chunks(q, doc_id=doc_id, top_k=3)
    return answers
