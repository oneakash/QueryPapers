"""Synchronous PDF text extraction with strict validation."""

from typing import Any

import pymupdf

from app.api.exceptions import InvalidDocumentError

_pymupdf: Any = pymupdf


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract plain text from raw PDF bytes or raise InvalidDocumentError."""
    if not file_bytes:
        raise InvalidDocumentError("Uploaded file is empty (0 bytes).")

    try:
        doc: Any = _pymupdf.open(stream=file_bytes, filetype="pdf")
    except Exception as exc:
        raise InvalidDocumentError("Corrupted or invalid PDF file format.") from exc

    try:
        if doc.is_encrypted:
            raise InvalidDocumentError("Password-protected PDFs are not supported.")

        page_count = int(doc.page_count)
        if page_count == 0:
            raise InvalidDocumentError("PDF contains zero pages.")

        pages_text: list[str] = []
        for idx in range(page_count):
            page = doc.load_page(idx)
            page_str = str(page.get_text("text")).strip()
            if page_str:
                pages_text.append(page_str)
    finally:
        doc.close()

    full_text = "\n\n".join(pages_text).strip()
    if not full_text:
        raise InvalidDocumentError(
            "No extractable text found (scanned or image-only PDF)."
        )

    return full_text
