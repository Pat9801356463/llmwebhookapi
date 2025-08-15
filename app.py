import hashlib
import json
import traceback
from typing import List, Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from config import Config
from engine.db import log_user_query
from engine.pdf_loader import extract_text_from_url
from engine.pinecone_handler import process_and_index_document
from engine.reasoner import reason_over_query
from engine.formatter import format_decision_response, format_pretty_print
from engine.embedding_handler import embed_chunks  # force load model at startup

# OCR fallback imports
try:
    import pytesseract
    from pdf2image import convert_from_bytes
    from PIL import Image
except ImportError:
    pytesseract = None

# ✅ Cache to avoid reprocessing the same document repeatedly
_doc_cache = {}

# Pre-load embedder at startup (avoids first request delay)
_ = embed_chunks(["warmup model load"])

app = FastAPI(
    title="HackRx Policy LLM API",
    description="RAG-first policy QA with Pinecone + Postgres logging",
    version="1.1.0",
)

class RunRequest(BaseModel):
    documents: str
    questions: List[str]
    session_id: Optional[str] = "anonymous"

class RunResponse(BaseModel):
    answers: List[str]

def _doc_id_from_url(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]

def _require_bearer(auth_header: Optional[str]):
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    token = auth_header.split(" ", 1)[1].strip()
    if not Config.API_KEY or token != Config.API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

def _to_plain_answer(result) -> str:
    """Ensure we always return a clean string answer."""
    try:
        if isinstance(result, dict):
            if result.get("justification"):
                return str(result["justification"]).strip()
            pp = format_pretty_print(format_decision_response(result))
            if pp and isinstance(pp, str):
                return pp.strip()
            return json.dumps(result, ensure_ascii=False)
        return str(result).strip()
    except Exception:
        return "No answer available"

def _ocr_extract_from_pdf(pdf_bytes: bytes) -> str:
    """Fallback OCR extraction for scanned PDFs."""
    if not pytesseract:
        print("[error] OCR dependencies not installed. Run: pip install pytesseract pillow pdf2image")
        return ""
    try:
        images = convert_from_bytes(pdf_bytes)
        text_parts = []
        for img in images:
            text_parts.append(pytesseract.image_to_string(img))
        return "\n".join(text_parts)
    except Exception as e:
        print(f"[error] OCR extraction failed: {e}")
        return ""

@app.post("/hackrx/run", response_model=RunResponse)
def hackrx_run(payload: RunRequest, Authorization: Optional[str] = Header(None)):
    _require_bearer(Authorization)

    if not payload.documents or not payload.questions:
        raise HTTPException(status_code=400, detail="Missing 'documents' or 'questions'")

    doc_url = payload.documents
    doc_id = _doc_id_from_url(doc_url)

    try:
        if doc_id not in _doc_cache:
            # Step 1: Extract text
            policy_text = extract_text_from_url(doc_url)
            print(f"[debug] Extracted text length: {len(policy_text) if policy_text else 0}")

            # Step 2: OCR fallback if empty
            if not policy_text or len(policy_text.strip()) == 0:
                print("[info] PDF extraction returned empty text! Attempting OCR...")
                try:
                    import requests
                    pdf_bytes = requests.get(doc_url, timeout=30).content
                    policy_text = _ocr_extract_from_pdf(pdf_bytes)
                    print(f"[debug] OCR extracted text length: {len(policy_text)}")
                except Exception as e:
                    print(f"[error] Failed to download or OCR PDF: {e}")

            if not policy_text or len(policy_text.strip()) == 0:
                raise HTTPException(status_code=500, detail="Failed to fetch/parse document text (even with OCR)")

            # Step 3: Index document
            chunks = process_and_index_document(doc_id, doc_type="policy", text=policy_text, source=doc_url)
            print(f"[debug] Processing {doc_id}, chunks before embedding: {len(chunks) if chunks else 0}")
            if not chunks or all(len(c.strip()) == 0 for c in chunks):
                print("[error] No valid chunks found to index.")

            _doc_cache[doc_id] = True

        answers: List[str] = []
        for q in payload.questions:
            try:
                result = reason_over_query(q, doc_id=doc_id, top_k=5)
                if not result or (isinstance(result, dict) and not result.get("matches")):
                    print(f"[warn] No matches found for: {q}")
                    result = {"justification": "No relevant information found in document"}
            except Exception as e:
                result = f"Error processing question: {str(e)}"
                print(f"[error] reason_over_query failed: {traceback.format_exc()}")
            try:
                log_user_query(payload.session_id or "anonymous", q)
            except Exception as e:
                print(f"[warn] failed to log query: {e}")
            answers.append(_to_plain_answer(result))

        return RunResponse(answers=answers)

    except HTTPException:
        raise
    except Exception as e:
        print(f"[fatal] Internal server error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.get("/health")
def health():
    return JSONResponse(content={"status": "ok"})

@app.get("/details")
def details():
    return JSONResponse(content={"status": "ok", "info": "Details endpoint reachable"})

@app.get("/hackrx/run")
def hackrx_run_info():
    return JSONResponse(content={
        "status": "ok",
        "message": "This is the HackRx Policy LLM runner. Use POST with an Authorization Bearer token to run processing.",
        "expected_payload": {
            "documents": "https://example.com/file.pdf",
            "questions": ["Question 1", "Question 2"],
            "session_id": "optional-session-id",
        },
    })

@app.get("/")
def root():
    return JSONResponse(content={
        "message": "HackRx Policy LLM API is running",
        "endpoints": ["/hackrx/run (GET|POST)", "/health", "/details"],
    })
