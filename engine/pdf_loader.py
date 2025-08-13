import io
import requests
import pdfplumber
import docx


def extract_text_from_bytes(file_bytes: bytes, filename_hint: str = "") -> str:
    """
    Extract text from PDF or DOCX given raw bytes.
    Uses filename hint to choose parser if possible; otherwise tries PDF then DOCX.
    """
    name = (filename_hint or "").lower()
    bio = io.BytesIO(file_bytes)

    if name.endswith(".pdf"):
        return _extract_pdf_text(bio)
    if name.endswith(".docx") or name.endswith(".doc"):
        return _extract_docx_text(bio)

    # Fallback: try PDF, then DOCX
    text = _extract_pdf_text(io.BytesIO(file_bytes))
    if text.strip():
        return text
    return _extract_docx_text(io.BytesIO(file_bytes))


def extract_text_from_url(url: str, timeout: int = 20) -> str:
    """
    Download a PDF/DOCX from URL and extract text.
    """
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return extract_text_from_bytes(resp.content, filename_hint=url)


def _extract_pdf_text(bio: io.BytesIO) -> str:
    text = ""
    try:
        with pdfplumber.open(bio) as pdf:
            for page in pdf.pages:
                t = page.extract_text() or ""
                if t:
                    text += t + "\n"
    except Exception:
        return ""
    return text.strip()


def _extract_docx_text(bio: io.BytesIO) -> str:
    try:
        document = docx.Document(bio)
        return "\n".join([p.text for p in document.paragraphs if p.text.strip()]).strip()
    except Exception:
        return ""
