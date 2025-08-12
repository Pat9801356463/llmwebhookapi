# app.py
import io
import hashlib
from typing import List, Optional

import requests
import pdfplumber
import docx

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from config import Config
from engine.db import create_tables, log_user_query
from engine.formatter import format_decision_response
from engine.reasoner import reason_over_query
from engine.pinecone_handler import process_and_index_document  # << Pinecone
# If you want to keep a local fallback, you can dual-import and feature flag.

app = FastAPI(title="Policy LLM API", version="1.0.0")


# ---------- Models ----------
class RunRequest(BaseModel):
    documents: str = Field(..., description="PDF/DOCX blob URL")
    questions: List[str] = Field(..., description="List of questions")
    session_id: Optional[str] = Field(default="anonymous")


class RunResponse(BaseModel):
    answers: List[dict]


# ---------- Startup ----------
@app.on_event("startup")
def on_startup():
    # Ensure DB tables exist
    try:
        create_tables()
    except Exception as e:
        print(f"⚠️ Failed to ensure tables on startup: {e}")


# ---------- Health / Root ----------
@app.get("/")
def index():
    return {
        "status": "ok",
        "message": "Policy LLM API is running",
        "endpoints": ["/hackrx/run", "/health"],
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


# ---------- Utils ----------
def extract_text_from_url(url: str) -> str:
    """Download and extract text from PDF or DOCX."""
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        file_bytes = io.BytesIO(resp.content)

        if url.lower().endswith(".pdf"):
            text = ""
            with pdfplumber.open(file_bytes) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text.strip()

        if url.lower().endswith(".docx"):
            doc = docx.Document(file_bytes)
            return "\n".join([p.text for p in doc.paragraphs if p.text.strip()]).strip()

        return ""
    except Exception as e:
        print(f"❌ Document extraction failed: {e}")
        return ""


def get_doc_id(url: str) -> str:
    """Generate a short SHA-256 doc_id from URL."""
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]


def _auth_check(authorization: Optional[str]):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    token = authorization.split(" ", 1)[1]
    if token != Config.API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


# ---------- Main API (Batch) ----------
@app.post("/hackrx/run", response_model=RunResponse)
def hackrx_run(payload: RunRequest, Authorization: Optional[str] = Header(None)):
    # 1) Auth
    _auth_check(Authorization)

    doc_url = payload.documents
    questions = payload.questions or []
    session_id = payload.session_id or "anonymous"

    if not doc_url or not questions:
        raise HTTPException(status_code=400, detail="Missing 'documents' or 'questions'")

    # 2) Ingest once → Pinecone index (doc_id namespace)
    doc_id = get_doc_id(doc_url)
    policy_text = extract_text_from_url(doc_url)
    if not policy_text:
        raise HTTPException(status_code=500, detail="Failed to extract document text")

    # Build chunks + embeddings + upsert to Pinecone (idempotent)
    # Also warms up an in-process cache inside pinecone_handler for this doc_id
    process_and_index_document(doc_id, "policy", policy_text, source=doc_url)

    # 3) Answer all questions against the same indexed doc
    answers = []
    for q in questions:
        reasoning_result = reason_over_query(q, doc_id, top_k=3)

        # Store in DB for audit/explainability
        try:
            log_user_query(session_id, q, reasoning_result)
        except Exception as e:
            print(f"⚠️ Failed to log query: {e}")

        # Format final API output
        answers.append(format_decision_response(reasoning_result))

    return JSONResponse(content={"answers": answers})
