# app.py
import hashlib
from typing import List, Optional
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from config import Config
from engine.db import create_tables, log_user_query
from engine.formatter import format_decision_response
from engine.reasoner import reason_over_query
from engine.pinecone_handler import process_and_index_document
from engine.pdf_loader import extract_text_from_url

app = FastAPI(title="HackRx Policy LLM API", version="1.0.0")


# ---------- Models ----------
class RunRequest(BaseModel):
    documents: str = Field(..., description="PDF/DOCX blob URL")
    questions: List[str] = Field(..., description="List of questions")
    session_id: Optional[str] = Field(default="anonymous")


# ---------- Startup ----------
@app.on_event("startup")
def on_startup():
    try:
        create_tables()
    except Exception as e:
        print(f"⚠️ Failed to ensure tables on startup: {e}")


# ---------- Health ----------
@app.get("/")
def index():
    return {
        "status": "ok",
        "message": "HackRx Policy LLM API is running",
        "endpoints": ["/hackrx/run", "/health"],
    }

@app.get("/health")
def health():
    return {"status": "healthy"}


# ---------- Utils ----------
def get_doc_id(url: str) -> str:
    """Generate a short SHA-256 doc_id from URL."""
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]

def _auth_check(authorization: Optional[str]):
    """
    Checks API key if set; skips if Config.API_KEY is missing/empty.
    """
    if not Config.API_KEY:
        print("⚠ AUTH CHECK SKIPPED — No API_KEY set in environment")
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    token = authorization.split(" ", 1)[1]
    if token != Config.API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


# ---------- Main HackRx Endpoint ----------
@app.post("/hackrx/run")
def hackrx_run(payload: RunRequest, Authorization: Optional[str] = Header(None)):
    # 1) Auth (skips if no API_KEY in env)
    _auth_check(Authorization)

    doc_url = payload.documents
    questions = payload.questions or []
    session_id = payload.session_id or "anonymous"

    if not doc_url or not questions:
        raise HTTPException(status_code=400, detail="Missing 'documents' or 'questions'")

    # 2) Ingest PDF into Pinecone
    doc_id = get_doc_id(doc_url)
    policy_text = extract_text_from_url(doc_url)
    if not policy_text:
        raise HTTPException(status_code=500, detail="Failed to extract document text")

    process_and_index_document(doc_id, "policy", policy_text, source=doc_url)

    # 3) Batch answer all questions
    answers = []
    for q in questions:
        reasoning_result = reason_over_query(q, doc_id, top_k=3)
        try:
            log_user_query(session_id, q, reasoning_result)
        except Exception as e:
            print(f"⚠️ Failed to log query: {e}")
        answers.append(format_decision_response(reasoning_result))

    return JSONResponse(content={"answers": answers})
