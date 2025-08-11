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
from engine.db import log_user_query
from engine.formatter import format_decision_response
from engine.query_parser import parse_query

app = Flask(__name__)

gemini_llm = GeminiLLM()
cohere_llm = CohereLLM()

def extract_text_from_url(url: str) -> str:
    """Extract text from PDF/DOCX."""
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
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]

def get_llm_answer(prompt: str) -> str:
    try:
        ans = gemini_llm.generate(prompt)
        if ans and ans.strip() and not ans.strip().startswith("❌"):
            return ans.strip()
    except Exception as e:
        print(f"⚠️ Gemini failed: {e}")

    try:
        coh_ans = cohere_llm.generate(prompt)
        if coh_ans and not coh_ans.strip().startswith("❌"):
            return coh_ans.strip()
        return f"❌ Cohere returned error: {coh_ans}"
    except Exception as e2:
        return f"❌ Cohere failed: {e2}"

@app.route("/hackrx/run", methods=["POST"])
def hackrx_run():
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer ") or auth_header.split(" ")[1] != Config.API_KEY:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json(force=True)
    doc_url = data.get("documents")
    questions = data.get("questions", [])
    session_id = data.get("session_id", "anonymous")

    if not doc_url or not questions:
        return jsonify({"error": "Missing 'documents' or 'questions'"}), 400

    doc_id = get_doc_id(doc_url)
    chunks, embeddings = process_and_store_document(doc_id, "policy", extract_text_from_url(doc_url))

    answers = []
    for q in questions:
        parsed_meta = parse_query(q)
        top_chunks = retrieve_top_chunks(q, doc_id, top_k=3)
        context = "\n".join(top_chunks)

        prompt = (
            "You are an insurance policy assistant.\n"
            f"Context from policy:\n{context}\n\n"
            f"Question: {q}\n"
            f"Answer concisely and ONLY based on the policy content."
        )

        raw_answer = get_llm_answer(prompt)
        reasoning_result = {
            "parsed": {**parsed_meta, "question": q},
            "decision": "info",
            "amount": None,
            "justification": raw_answer,
            "matched_clauses": [
                {"source": doc_id, "doc_type": "policy", "text": c}
                for c in top_chunks
            ]
        }

        log_user_query(session_id, q, reasoning_result)
        answers.append(format_decision_response(reasoning_result))

    return jsonify({"answers": answers})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
