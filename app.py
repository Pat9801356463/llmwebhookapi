from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from typing import List
import requests
import tempfile
import os
import fitz  # PyMuPDF for PDF reading
import uvicorn

# If you use a vector DB (e.g., Pinecone), import your retrieval logic here
# from your_retrieval_module import retrieve_answers

app = FastAPI()

# Replace with your expected Bearer token for auth
API_KEY = os.getenv("HACKRX_API_KEY", "your_api_key_here")

# -------- Data Models -------- #
class HackRxRequest(BaseModel):
    documents: str  # URL to PDF
    questions: List[str]

class HackRxResponse(BaseModel):
    answers: List[str]

# -------- Helpers -------- #
def verify_auth(request: Request):
    """Check Bearer token auth."""
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    token = auth_header.split(" ")[1]
    if token != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")

def download_pdf(url: str) -> str:
    """Download PDF from URL to a temp file."""
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp_file.write(resp.content)
    tmp_file.close()
    return tmp_file.name

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF."""
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def simple_answer_lookup(pdf_text: str, question: str) -> str:
    """
    A placeholder answer generator.
    Replace with retrieval-augmented generation logic (e.g., vector DB + GPT).
    """
    return f"(Stub answer) For question '{question}', please check policy document."

# -------- API Endpoints -------- #
@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/hackrx/run", response_model=HackRxResponse)
async def hackrx_run(request: Request, payload: HackRxRequest):
    verify_auth(request)

    try:
        # Step 1: Download and read PDF
        pdf_path = download_pdf(payload.documents)
        pdf_text = extract_text_from_pdf(pdf_path)

        # Step 2: Generate answers
        answers = [simple_answer_lookup(pdf_text, q) for q in payload.questions]

        # Step 3: Clean up
        os.remove(pdf_path)

        # Step 4: Return in correct format
        return {"answers": answers}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------- Local Dev Run -------- #
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
