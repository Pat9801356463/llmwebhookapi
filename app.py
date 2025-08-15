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
except ImportError:
    pytesseract = None

_doc_cache = {}
_ = embed_chunks(["warmup model load"])  # warmup embedder

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
    if not pytesseract:
        print("[error] OCR dependencies not installed.")
        return ""
    try:
        images = convert_from_bytes(pdf_bytes)
        return "\n".join(pytesseract.image_to_string(img) for img in images)
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
            policy_text = extract_text_from_url(doc_url)
            if not policy_text.strip():
                print("[info] Empty text from PDF — trying OCR fallback.")
                try:
                    import requests
                    pdf_bytes = requests.get(doc_url, timeout=30).content
                    policy_text = _ocr_extract_from_pdf(pdf_bytes)
                except Exception as e:
                    print(f"[error] OCR download failed: {e}")

            if not policy_text.strip():
                raise HTTPException(status_code=500, detail="Failed to extract text from document.")

            chunks = process_and_index_document(doc_id, "policy", policy_text, doc_url)
            if not chunks or all(not c.strip() for c in chunks):
                print("[warn] No chunks to index.")
            _doc_cache[doc_id] = True

        answers: List[str] = []
        for q in payload.questions:
            try:
                result = reason_over_query(q, doc_id=doc_id, top_k=5)
                if not result or (isinstance(result, dict) and not result.get("matches")):
                    result = {"justification": "No relevant information found in document"}
            except Exception:
                result = {"justification": "Error processing question"}
                print(f"[error] reason_over_query failed:\n{traceback.format_exc()}")
            try:
                log_user_query(payload.session_id or "anonymous", q)
            except Exception as e:
                print(f"[warn] Failed to log query: {e}")
            answers.append(_to_plain_answer(result))

        return RunResponse(answers=answers)

    except HTTPException:
        raise
    except Exception:
        print(f"[fatal] Internal server error:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/details")
def details():
    return {"status": "ok", "info": "Details endpoint reachable"}

@app.post("/hackrx/run", response_model=RunResponse)
def hackrx_run(
    payload: RunRequest,
    Authorization: Optional[str] = Header(None),
    force_reload: Optional[bool] = False
):
    _require_bearer(Authorization)

    if not payload.documents or not payload.questions:
        raise HTTPException(status_code=400, detail="Missing 'documents' or 'questions'")

    doc_url = payload.documents
    doc_id = _doc_id_from_url(doc_url)

    # ✅ Clear cache if force_reload is requested
    if force_reload:
        print(f"[info] force_reload=True → Clearing cache for doc {doc_id}")
        _doc_cache.pop(doc_id, None)

    try:
        if doc_id not in _doc_cache:
            policy_text = extract_text_from_url(doc_url)
            if not policy_text.strip():
                print("[info] Empty text from PDF — trying OCR fallback.")
                try:
                    import requests
                    pdf_bytes = requests.get(doc_url, timeout=30).content
                    policy_text = _ocr_extract_from_pdf(pdf_bytes)
                except Exception as e:
                    print(f"[error] OCR download failed: {e}")

            if not policy_text.strip():
                raise HTTPException(status_code=500, detail="Failed to extract text from document.")

            chunks = process_and_index_document(doc_id, "policy", policy_text, doc_url)
            if not chunks or all(not c.strip() for c in chunks):
                print("[warn] No chunks to index.")
            _doc_cache[doc_id] = True

        answers: List[str] = []
        for q in payload.questions:
            try:
                result = reason_over_query(q, doc_id=doc_id, top_k=5)
                if not result or (isinstance(result, dict) and not result.get("matches")):
                    result = {"justification": "No relevant information found in document"}
            except Exception:
                result = {"justification": "Error processing question"}
                print(f"[error] reason_over_query failed:\n{traceback.format_exc()}")
            try:
                log_user_query(payload.session_id or "anonymous", q)
            except Exception as e:
                print(f"[warn] Failed to log query: {e}")
            answers.append(_to_plain_answer(result))

        return RunResponse(answers=answers)

    except HTTPException:
        raise
    except Exception:
        print(f"[fatal] Internal server error:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.get("/")
def root():
    return {
        "message": "HackRx Policy LLM API is running",
        "endpoints": ["/hackrx/run (GET|POST)", "/health", "/details"],
    }
