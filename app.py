import hashlib
from typing import List, Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from config import Config
from engine.db import log_user_query  # safe: will no-op if PG isn't configured
from engine.pdf_loader import extract_text_from_url
from engine.pinecone_handler import process_and_index_document
from engine.reasoner import reason_over_query
from engine.formatter import format_decision_response

# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI(
    title="HackRx Policy LLM API",
    description="RAG-first policy QA with Pinecone + Postgres logging",
    version="1.0.0",
)

# -----------------------------
# Models
# -----------------------------
class RunRequest(BaseModel):
    documents: str = Field(..., description="Public PDF/DOCX URL (blob link)")
    questions: List[str] = Field(..., description="List of questions to answer")
    session_id: Optional[str] = Field(default="anonymous", description="Session/user id for logging")

class RunResponse(BaseModel):
    answers: List[str]

# -----------------------------
# Helpers
# -----------------------------
def _doc_id_from_url(url: str) -> str:
    # short stable namespace for Pinecone
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]

def _require_bearer(auth_header: Optional[str]):
    """Strict Bearer auth for HackRx runner."""
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    token = auth_header.split(" ", 1)[1].strip()
    if not Config.API_KEY or token != Config.API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

# -----------------------------
# GET version of /hackrx/run (informational only)
# -----------------------------
@app.get("/hackrx/run")
def hackrx_run_get():
    return JSONResponse(
        content={
            "status": "ok",
            "message": "This is the HackRx Policy LLM runner. Use POST with an Authorization Bearer token to run processing.",
            "expected_payload": {
                "documents": "https://example.com/file.pdf",
                "questions": ["Question 1", "Question 2"],
                "session_id": "optional-session-id"
            }
        }
    )

# -----------------------------
# POST version of /hackrx/run (full processing)
# -----------------------------
@app.post("/hackrx/run", response_model=RunResponse)
def hackrx_run(payload: RunRequest, Authorization: Optional[str] = Header(None)):
    # 1) Auth
    _require_bearer(Authorization)

    # 2) Validate input
    if not payload.documents or not payload.questions:
        raise HTTPException(status_code=400, detail="Missing 'documents' or 'questions'")

    # 3) Generate doc namespace & fetch text
    doc_url = payload.documents
    doc_id = _doc_id_from_url(doc_url)

    policy_text = extract_text_from_url(doc_url)
    if not policy_text:
        raise HTTPException(status_code=500, detail="Failed to fetch/parse document text")

    # 4) Ingest (idempotent): chunk, embed, persist, upsert to Pinecone (namespace=doc_id)
    process_and_index_document(doc_id, doc_type="policy", text=policy_text, source=doc_url)

    # 5) Answer all questions via pipeline (parse → retrieve → reason)
    answers: List[str] = []
    for q in payload.questions:
        result = reason_over_query(q, doc_id=doc_id, top_k=5)
        # optional logging to Postgres (no-op if DB not configured)
        try:
            log_user_query(payload.session_id or "anonymous", q)
        except Exception as e:
            # don't fail user flow for logging errors
            print(f"[warn] failed to log query: {e}")
        # Convert rich JSON decision into a simple answer string as expected by HackRx
        answers.append(format_decision_response(result))

    return RunResponse(answers=answers)

# -----------------------------
# Health / Details / Root
# -----------------------------
@app.get("/health")
def health():
    return JSONResponse(content={"status": "ok"})

@app.get("/details")
def details():
    return JSONResponse(content={"status": "ok", "info": "Details endpoint reachable"})

@app.get("/")
def root():
    return JSONResponse(
        content={
            "message": "HackRx Policy LLM API is running",
            "endpoints": ["/hackrx/run (GET|POST)", "/health", "/details"],
        }
    )
