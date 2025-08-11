from flask import Flask, request, jsonify
import requests
import io
import hashlib
import pdfplumber
import docx
from config import Config
from engine.gemini_runner import GeminiLLM
from engine.cohere_runner import CohereLLM
from engine.faiss_handler import process_and_store_document, retrieve_top_chunks
from engine.db import fetch_chunks_from_db
import numpy as np

app = Flask(__name__)

# === LLM Clients ===
gemini_llm = GeminiLLM()
cohere_llm = CohereLLM()


def extract_text_from_url(url: str) -> str:
    """Download and extract text from PDF or DOCX URL."""
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        file_bytes = io.BytesIO(resp.content)

        if url.lower().endswith(".pdf"):
            text = ""
            with pdfplumber.open(file_bytes) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text

        elif url.lower().endswith(".docx"):
            doc = docx.Document(file_bytes)
            return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

        return ""
    except Exception as e:
        print(f"❌ Document extraction failed: {e}")
        return ""


def get_doc_id(url: str) -> str:
    """Generate a unique document ID from its URL."""
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]


def get_llm_answer(prompt: str) -> str:
    """Try Gemini first, fallback to Cohere if it fails or returns error/empty."""
    try:
        ans = gemini_llm.generate(prompt)
        if ans and not ans.strip().startswith("❌") and ans.strip():
            return ans.strip()
        print(f"⚠️ Gemini returned error/empty — falling back to Cohere: {ans}")
    except Exception as e:
        print(f"⚠️ Gemini call failed: {e}")

    try:
        coh_ans = cohere_llm.generate(prompt)
        if coh_ans and not coh_ans.strip().startswith("❌"):
            return coh_ans.strip()
        return f"❌ Cohere returned error: {coh_ans}"
    except Exception as e2:
        return f"❌ Error generating answer with Cohere: {e2}"


@app.route("/hackrx/run", methods=["POST"])
def hackrx_run():
    # === 1. Auth check ===
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer ") or auth_header.split(" ")[1] != Config.API_KEY:
        return jsonify({"error": "Unauthorized"}), 401

    # === 2. Parse request ===
    data = request.get_json(force=True)
    doc_url = data.get("documents")
    questions = data.get("questions", [])

    if not doc_url or not questions:
        return jsonify({"error": "Missing 'documents' or 'questions'"}), 400

    doc_id = get_doc_id(doc_url)

    # === 3. Retrieve or process document ===
    stored_chunks = fetch_chunks_from_db(doc_id)
    if stored_chunks:
        print(f"📂 Found {len(stored_chunks)} chunks in DB for {doc_id}")
        chunks = [c["text"] for c in stored_chunks]
        embeddings = np.vstack([c["embedding"] for c in stored_chunks])
    else:
        print(f"🆕 Processing new document {doc_id}")
        policy_text = extract_text_from_url(doc_url)
        if not policy_text:
            return jsonify({"error": "Failed to extract document text"}), 500
        chunks, embeddings = process_and_store_document(doc_id, "policy", policy_text)

    # === 4. Answer each question ===
    answers = []
    for q in questions:
        top_chunks = retrieve_top_chunks(q, chunks, embeddings, top_k=3)
        context = "\n".join(top_chunks)

        prompt = (
            "You are an insurance policy assistant.\n"
            f"Context from policy:\n{context}\n\n"
            f"Question: {q}\n"
            f"Answer concisely and ONLY based on the policy content."
        )

        answers.append(get_llm_answer(prompt))

    # === 5. Return HackRx format ===
    return jsonify({"answers": answers})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
