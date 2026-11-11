"""Deterministic text chunking utilities for document ingestion."""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.api.exceptions import InvalidDocumentError


def chunk_text(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> list[str]:
    """Split extracted document text into overlapping semantic chunks."""
    cleaned = text.strip()
    if not cleaned:
        raise InvalidDocumentError("Cannot chunk empty text.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = [chunk.strip() for chunk in splitter.split_text(cleaned) if chunk.strip()]
    if not chunks:
        raise InvalidDocumentError("Document produced zero valid chunks.")
    return chunks
