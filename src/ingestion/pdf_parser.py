import io
from typing import Optional

import pdfplumber


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """Extract raw text from a PDF file.

    Notes:
        - Used for inference-time resume uploads.
        - For the Kaggle CSV dataset, we ingest `Resume_str` directly.
    """

    if not pdf_bytes:
        return ""

    text_parts = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                text_parts.append(page_text)

    return "\n".join(text_parts)


def extract_text_from_pdf_path(pdf_path: str) -> str:
    with open(pdf_path, "rb") as f:
        return extract_text_from_pdf_bytes(f.read())


def safe_decode_text_file(file_bytes: bytes, encoding: Optional[str] = None) -> str:
    if not file_bytes:
        return ""

    if encoding:
        return file_bytes.decode(encoding, errors="replace")

    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            return file_bytes.decode(enc)
        except UnicodeDecodeError:
            continue

    return file_bytes.decode("utf-8", errors="replace")
