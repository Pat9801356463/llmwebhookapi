import os
import json
import faiss
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
import pdfplumber
from docx import Document
from config import Config
from engine.db import save_chunks_to_db

EMBEDDING_MODEL = SentenceTransformer(Config.EMBEDDING_MODEL_NAME)

def extract_text_from_pdf(file_path):
    try:
        with pdfplumber.open(file_path) as pdf:
            return "\n".join([page.extract_text() or "" for page in pdf.pages]).strip()
    except Exception as e:
        print(f"⚠️ Error reading PDF {file_path}: {e}")
        return ""

def extract_text_from_docx(file_path):
    try:
        doc = Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs if para.text.strip()]).strip()
    except Exception as e:
        print(f"⚠️ Error reading DOCX {file_path}: {e}")
        return ""

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    tokens = text.split()
    chunks = []
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk = " ".join(tokens[i:i + chunk_size]).strip()
        if chunk:  # only keep non-empty chunks
            chunks.append(chunk)
    return chunks

def embed_chunks(chunks: List[str]) -> np.ndarray:
    return EMBEDDING_MODEL.encode(chunks, convert_to_numpy=True)

def process_file(file_path, doc_type):
    filename = os.path.basename(file_path)
    print(f"📄 Processing: {filename}")

    # Extract text
    if file_path.endswith(".pdf"):
        text = extract_text_from_pdf(file_path)
    elif file_path.endswith(".docx"):
        text = extract_text_from_docx(file_path)
    else:
        print(f"⚠️ Skipped unsupported file type: {file_path}")
        return [], []

    if not text:
        print(f"⚠️ Skipped empty document: {file_path}")
        return [], []

    # Chunk and embed
    chunks = chunk_text(text, Config.CHUNK_SIZE, Config.OVERLAP_SIZE)
    if not chunks:
        print(f"⚠️ No valid chunks generated for: {file_path}")
        return [], []

    embeddings = embed_chunks(chunks)
    metadata = [
        {"text": chunk, "source": filename, "doc_type": doc_type, "chunk_id": i}
        for i, chunk in enumerate(chunks)
    ]
    return embeddings, metadata

def build_faiss_index(all_vectors: np.ndarray, save_path: str):
    print("🔧 Building FAISS index...")
    dimension = all_vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(all_vectors)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    faiss.write_index(index, save_path)
    print(f"✅ FAISS index saved at: {save_path}")

def save_metadata(metadata, save_path="data/embeddings/chunk_metadata.json"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"✅ Metadata saved to: {save_path}")

def run_indexing():
    root_dirs = {
        "data/policies/": "policy",
        "data/contracts/": "contract",
        "data/emails/": "email"
    }

    all_embeddings = []
    all_metadata = []

    for folder, doc_type in root_dirs.items():
        if not os.path.exists(folder):
            print(f"⚠️ Skipping missing folder: {folder}")
            continue

        for file in os.listdir(folder):
            if file.lower().endswith((".pdf", ".docx")):
                path = os.path.join(folder, file)
                embeddings, metadata = process_file(path, doc_type)
                if embeddings.size > 0:
                    all_embeddings.append(embeddings)
                    all_metadata.extend(metadata)

    if all_embeddings:
        all_vectors = np.vstack(all_embeddings)  # flatten to single array
        build_faiss_index(all_vectors, Config.FAISS_INDEX_PATH)
        save_metadata(all_metadata)
        save_chunks_to_db(all_metadata, all_vectors)
        print("✅ All embeddings and metadata saved.")
    else:
        print("❌ No valid documents found. Indexing aborted.")

if __name__ == "__main__":
    run_indexing()
