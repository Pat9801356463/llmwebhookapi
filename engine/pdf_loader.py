import io
import requests
import pdfplumber
import docx

def extract_text_from_bytes(file_bytes: bytes, filename_hint: str = "") -> str:
    name = (filename_hint or "").lower()
    bio = io.BytesIO(file_bytes)

    if name.endswith(".pdf"):
        return _extract_pdf_text(bio)
    if name.endswith(".docx") or name.endswith(".doc"):
        return _extract_docx_text(bio)

    text = _extract_pdf_text(io.BytesIO(file_bytes))
    if text.strip():
        return text
    return _extract_docx_text(io.BytesIO(file_bytes))

def extract_text_from_url(url: str, timeout: int = 20) -> str:
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    text = extract_text_from_bytes(resp.content, filename_hint=url)
    print(f"[debug] PDF Loader: Extracted {len(text.split())} words from {url}")
    if text:
        print(f"[debug] Sample text: {text[:120]}...")
    return text

def _extract_pdf_text(bio: io.BytesIO) -> str:
    text = ""
    try:
        with pdfplumber.open(bio) as pdf:
            for page in pdf.pages:
                t = page.extract_text() or ""
                if t.strip():
                    text += t + "\n"
    except Exception as e:
        print(f"[error] PDF extraction failed: {e}")
        return ""
    return text.strip()

def _extract_docx_text(bio: io.BytesIO) -> str:
    try:
        document = docx.Document(bio)
        return "\n".join([p.text for p in document.paragraphs if p.text.strip()]).strip()
    except Exception as e:
        print(f"[error] DOCX extraction failed: {e}")
        return ""

