from flask import Flask, request, jsonify, render_template
from engine.query_parser import parse_query
from engine.retriever import retrieve_clauses
from engine.reasoner import reason_over_query
from engine.formatter import format_decision_response
from engine.session_manager import get_session_context, update_session, start_session
from engine.db import log_user_query
from engine.gemini_runner import GeminiLLM
from config import Config
import requests
import pdfplumber
import io

app = Flask(__name__)

# === Load Gemini client ===
gemini_llm = GeminiLLM()

# === External alternate policy API URL ===
ALT_API_URL = "https://bajaj-alianz-health-insurance-lznkjuhdddi3v9weyxgshc.streamlit.app/api/alternatives"

def fetch_alternates_from_external(parsed):
    try:
        payload = {
            "age": parsed.get("age"),
            "state": parsed.get("location"),
            "coverage": parsed.get("coverage", 30000),
            "plan_type": parsed.get("plan_type", "Any")
        }
        res = requests.post(ALT_API_URL, json=payload, timeout=10)
        if res.status_code == 200:
            return res.json().get("plans", [])
        else:
            print(f"⚠️ Failed to fetch alternates. Status: {res.status_code}")
            return []
    except Exception as e:
        print("❌ Error fetching external alternatives:", e)
        return []

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_query = request.form.get("query", "")
        session_id = request.remote_addr

        result = reason_over_query(user_query)
        response_json = format_decision_response(result)

        log_user_query(session_id, user_query, response_json)
        update_session(session_id, user_query, response_json)

        alt_suggestions = []
        if result.get("decision", "").lower() == "rejected":
            alt_suggestions = fetch_alternates_from_external(result.get("parsed", {}))

        return render_template("index.html", response=response_json, suggestions=alt_suggestions)

    return render_template("index.html")

@app.route("/api/query", methods=["POST"])
def api_query():
    data = request.json
    user_query = data.get("query", "")
    session_id = request.remote_addr

    result = reason_over_query(user_query)
    response_json = format_decision_response(result)
    log_user_query(session_id, user_query, response_json)
    update_session(session_id, user_query, response_json)

    alt_suggestions = []
    if result.get("decision", "").lower() == "rejected":
        alt_suggestions = fetch_alternates_from_external(result.get("parsed", {}))

    return jsonify({
        "response": response_json,
        "suggestions": alt_suggestions
    })

@app.route("/api/context", methods=["GET"])
def api_context():
    session_id = request.remote_addr
    return jsonify(get_session_context(session_id))

@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_json(force=True)
    if not data or "query" not in data:
        return jsonify({"error": "Missing 'query' in request"}), 400

    user_query = data["query"]
    session_id = data.get("session_id") or start_session()

    result = reason_over_query(user_query)
    response_json = format_decision_response(result)

    log_user_query(session_id, user_query, response_json)
    update_session(session_id, user_query, response_json)

    alt_suggestions = []
    if result.get("decision", "").lower() == "rejected":
        alt_suggestions = fetch_alternates_from_external(result.get("parsed", {}))

    return jsonify({
        "session_id": session_id,
        "response": response_json,
        "suggestions": alt_suggestions
    })

# === HackRx Required Endpoint ===
@app.route("/hackrx/run", methods=["POST"])
def hackrx_run():
    # 1. Auth check
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer ") or auth_header.split(" ")[1] != Config.API_KEY:
        return jsonify({"error": "Unauthorized"}), 401

    # 2. Parse request
    data = request.get_json(force=True)
    pdf_url = data.get("documents")
    questions = data.get("questions", [])

    if not pdf_url or not questions:
        return jsonify({"error": "Missing 'documents' or 'questions'"}), 400

    # 3. Download PDF
    try:
        pdf_resp = requests.get(pdf_url, timeout=15)
        pdf_resp.raise_for_status()
        pdf_bytes = io.BytesIO(pdf_resp.content)
    except Exception as e:
        return jsonify({"error": f"Failed to download PDF: {e}"}), 500

    # 4. Extract text
    try:
        text = ""
        with pdfplumber.open(pdf_bytes) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        return jsonify({"error": f"Failed to extract PDF text: {e}"}), 500

    # Limit text to avoid LLM token overflow
    truncated_text = text[:15000]

    # 5. Get answers
    answers = []
    for q in questions:
        prompt = (
            f"You are an insurance policy assistant.\n"
            f"Policy document content:\n{truncated_text}\n\n"
            f"Question: {q}\n"
            f"Answer concisely based ONLY on the above policy content."
        )
        try:
            answer = gemini_llm.generate(prompt)
            answers.append(answer.strip())
        except Exception as e:
            answers.append(f"Error generating answer: {e}")

    # 6. Return HackRx format
    return jsonify({"answers": answers})

if __name__ == "__main__":
    app.run(debug=True)

