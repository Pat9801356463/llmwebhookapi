# app.py
import hashlib
import json
import traceback
from typing import List, Optional

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from config import Config
from engine.db import log_user_query
from engine.pdf_loader import extract_text_from_url
from engine.pinecone_handler import process_and_index_document, namespace_exists
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
_ = embed_chunks(["warmup model load"])  # warmup

app = FastAPI(
    title="HackRx Policy LLM API",
    description="RAG-first policy QA with Pinecone + Postgres logging",
    version="1.2.1",  # bumped version
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
            just = (result.get("justification") or "").strip()
            if just:
                return just
            try:
                pp = format_pretty_print(format_decision_response(result))
                if isinstance(pp, str) and pp.strip():
                    return pp.strip()
            except Exception:
                pass
            return json.dumps(result, ensure_ascii=False)
        s = str(result or "").strip()
        return s if s else "No answer available"
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

    # No more unconditional cache clearing — keeps in-proc skip working
    global _doc_cache

    if force_reload:
        print(f"[info] force_reload=True for doc {doc_id}")

    try:
        # 1) Extract text
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
            raise HTTPException(status_code=422, detail="No text extracted from document.")

        chunks = []
        # 2) Only embed/index if not already present or force_reload
        if force_reload or not namespace_exists(doc_id):
            print(f"[info] Processing and indexing doc {doc_id}...")
            chunks = process_and_index_document(doc_id, "policy", policy_text, doc_url)
        else:
            print(f"[info] Doc {doc_id} already exists in Pinecone — skipping embedding.")

        # Flatten chunks only if we processed them
        if chunks:
            flat_chunks = []
            for c in chunks:
                if isinstance(c, list):
                    flat_chunks.extend([str(x) for x in c if str(x).strip()])
                elif isinstance(c, str) and c.strip():
                    flat_chunks.append(c)
            chunks = flat_chunks
            if not chunks:
                print("[warn] No non-empty chunks to index.")
                raise HTTPException(status_code=422, detail="No valid chunks extracted.")
            _doc_cache[doc_id] = True

        # 3) Answer questions
        answers: List[str] = []
        for q in payload.questions:
            try:
                result = reason_over_query(q, doc_id=doc_id, top_k=5)  # reduced top_k for speed
                if not result or (isinstance(result, dict) and not result.get("matched_clauses")):
                    result = {
                        "justification": "No explicit exclusion matched. If the waiting period is already met, approve unless another exclusion applies."
                    }
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

@app.get("/")
def root():
    return {
        "message": "HackRx Policy LLM API is running",
        "endpoints": ["/hackrx/run (POST)", "/health", "/details"],
    }
