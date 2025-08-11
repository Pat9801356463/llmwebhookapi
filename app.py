from flask import Flask, request, jsonify
import requests
import io
import hashlib
import pdfplumber
import docx
from config import Config
from engine.faiss_handler import process_and_store_document
from engine.db import log_user_query
from engine.formatter import format_decision_response
from engine.reasoner import reason_over_query

app = Flask(__name__)

# --- Health & Root ---
@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "status": "ok",
        "message": "Policy LLM API is running",
        "endpoints": ["/hackrx/run", "/health"]
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"}), 200


# --- Utils ---
def extract_text_from_url(url: str) -> str:
    """Download and extract text from PDF or DOCX."""
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
    """Generate a short SHA-256 doc_id from URL."""
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]


# --- Main API ---
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
    session_id = data.get("session_id", "anonymous")

    if not doc_url or not questions:
        return jsonify({"error": "Missing 'documents' or 'questions'"}), 400

    # === 3. Ensure document is processed & stored ===
    doc_id = get_doc_id(doc_url)
    policy_text = extract_text_from_url(doc_url)
    if not policy_text:
        return jsonify({"error": "Failed to extract document text"}), 500

    process_and_store_document(doc_id, "policy", policy_text, source=doc_url)

    # === 4. Run reasoning for each question ===
    answers = []
    for q in questions:
        reasoning_result = reason_over_query(q, doc_id, top_k=3)

        # 📝 Store in DB
        log_user_query(session_id, q, reasoning_result)

        # 📦 Format final API output
        answers.append(format_decision_response(reasoning_result))

    return jsonify({"answers": answers})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)

