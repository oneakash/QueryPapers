"""Asynchronous document ingestion pipeline with threadpool offloading and rollback."""

from fastapi.concurrency import run_in_threadpool

from app.api.exceptions import InvalidDocumentError, VectorStoreOperationError
from app.core.interfaces import VectorStoreProtocol
from app.ingestion.chunker import chunk_text
from app.ingestion.parsers.pdf_parser import extract_text_from_pdf


def _parse_and_chunk_sync(file_bytes: bytes, filename: str) -> list[str]:
    """Run CPU-bound parsing and splitting inside a worker thread."""
    lower_name = filename.lower()
    if lower_name.endswith(".pdf"):
        raw_text = extract_text_from_pdf(file_bytes)
    elif lower_name.endswith(".txt"):
        if not file_bytes:
            raise InvalidDocumentError("Uploaded .txt file is empty (0 bytes).")
        try:
            raw_text = file_bytes.decode("utf-8").strip()
        except UnicodeDecodeError as exc:
            raise InvalidDocumentError(
                "Text file must be valid UTF-8 encoded."
            ) from exc
        if not raw_text:
            raise InvalidDocumentError("Uploaded .txt file contains only whitespace.")
    else:
        raise InvalidDocumentError(
            "Unsupported file type. Only .pdf and .txt files are allowed."
        )

    return chunk_text(raw_text)


async def ingest_document(
    file_bytes: bytes,
    filename: str,
    user_id: str,
    doc_id: str,
    vector_store: VectorStoreProtocol,
) -> int:
    """Parse, chunk, and upsert a document with rollback on failure."""
    chunks = await run_in_threadpool(_parse_and_chunk_sync, file_bytes, filename)

    try:
        inserted_count = await vector_store.upsert_chunks(
            user_id=user_id,
            doc_id=doc_id,
            chunks=chunks,
        )
        return inserted_count
    except Exception as exc:
        # Rollback any partially inserted chunks for this (user_id, doc_id)
        try:
            await vector_store.delete_document(user_id=user_id, doc_id=doc_id)
        except Exception:
            pass
        raise VectorStoreOperationError(
            f"Failed to index document '{doc_id}'; partial points rolled back."
        ) from exc
