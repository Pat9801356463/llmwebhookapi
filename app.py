from flask import Flask, request, jsonify
import requests
import io
import pdfplumber
import docx
from config import Config
from engine.gemini_runner import GeminiLLM
from engine.cohere_runner import CohereLLM
from engine.faiss_handler import build_faiss_index, retrieve_top_chunks
from engine.text_utils import chunk_text

app = Flask(__name__)

# === LLM Clients ===
gemini_llm = GeminiLLM()
cohere_llm = CohereLLM()

def extract_text_from_url(url):
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

def get_llm_answer(prompt):
    """Try Gemini first, fallback to Cohere if it fails."""
    try:
        return gemini_llm.generate(prompt)
    except Exception as e:
        print(f"⚠️ Gemini failed: {e} — falling back to Cohere")
        try:
            return cohere_llm.generate(prompt)
        except Exception as e2:
            return f"Error generating answer: {e2}"

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

    # === 3. Extract text ===
    policy_text = extract_text_from_url(doc_url)
    if not policy_text:
        return jsonify({"error": "Failed to extract document text"}), 500

    # === 4. Chunk + embed in FAISS (in-memory) ===
    chunks = chunk_text(policy_text, chunk_size=500)
    index, chunk_map = build_faiss_index(chunks)

    # === 5. Answer each question ===
    answers = []
    for q in questions:
        # Retrieve relevant text
        top_chunks = retrieve_top_chunks(q, index, chunk_map, top_k=3)
        context = "\n".join(top_chunks)

        prompt = (
            "You are an insurance policy assistant.\n"
            f"Context from policy:\n{context}\n\n"
            f"Question: {q}\n"
            f"Answer concisely and ONLY based on the policy content."
        )

        answer = get_llm_answer(prompt)
        answers.append(answer.strip())

    # === 6. Return HackRx format ===
    return jsonify({"answers": answers})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
