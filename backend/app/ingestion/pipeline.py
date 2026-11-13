"""Asynchronous document ingestion pipeline with threadpool offloading and rollback."""

import hashlib
import uuid
from dataclasses import dataclass
from typing import Literal

from fastapi.concurrency import run_in_threadpool

from app.api.exceptions import InvalidDocumentError, VectorStoreOperationError
from app.core.interfaces import VectorStoreProtocol
from app.ingestion.chunker import chunk_text
from app.ingestion.parsers.pdf_parser import extract_text_from_pdf

IngestStatus = Literal["indexed", "duplicate", "replaced"]


@dataclass(frozen=True, slots=True)
class IngestResult:
    """Outcome of an idempotent ingestion request."""

    doc_id: str
    filename: str
    chunks_indexed: int
    status: IngestStatus
    replaced_doc_ids: tuple[str, ...] = ()


_DOC_ID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_DNS, "querypapers.documents")


def derive_doc_id(user_id: str, content_hash: str) -> str:
    """Return a stable per-tenant document ID: uuid5(user_id + content_hash)."""
    return str(uuid.uuid5(_DOC_ID_NAMESPACE, f"{user_id}:{content_hash}"))


def compute_content_hash(file_bytes: bytes) -> str:
    """Return the SHA-256 hex digest that identifies a file's exact contents."""
    return hashlib.sha256(file_bytes).hexdigest()


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
    content_hash: str = "",
) -> int:
    """Parse, chunk, and upsert a document with rollback on failure."""
    chunks = await run_in_threadpool(_parse_and_chunk_sync, file_bytes, filename)

    try:
        inserted_count = await vector_store.upsert_chunks(
            user_id=user_id,
            doc_id=doc_id,
            chunks=chunks,
            filename=filename,
            content_hash=content_hash,
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


async def ingest_document_idempotent(
    file_bytes: bytes,
    filename: str,
    user_id: str,
    vector_store: VectorStoreProtocol,
    replace_same_filename: bool = False,
) -> IngestResult:
    """Ingest a file once per user, skipping duplicates and optionally replacing.

    - Byte-identical content already indexed for this user is skipped, with no
      parsing or embedding, and the existing document is returned.
    - With ``replace_same_filename``, older documents that share this filename
      are removed only after the new version is fully indexed.
    - ``doc_id`` is ``uuid5(user_id, content_hash)``, so two concurrent identical
      uploads write the same point IDs instead of creating duplicates.
    """
    content_hash = compute_content_hash(file_bytes)

    duplicates = await vector_store.find_documents(user_id, content_hash=content_hash)
    if duplicates:
        existing = duplicates[0]
        return IngestResult(
            doc_id=existing.doc_id,
            filename=existing.filename or filename,
            chunks_indexed=existing.chunk_count,
            status="duplicate",
        )

    doc_id = derive_doc_id(user_id, content_hash)
    stale_ids: list[str] = []
    if replace_same_filename:
        same_name = await vector_store.find_documents(user_id, filename=filename)
        stale_ids = [d.doc_id for d in same_name if d.doc_id != doc_id]

    chunks_count = await ingest_document(
        file_bytes=file_bytes,
        filename=filename,
        user_id=user_id,
        doc_id=doc_id,
        vector_store=vector_store,
        content_hash=content_hash,
    )
    if not stale_ids:
        return IngestResult(doc_id, filename, chunks_count, "indexed")

    try:
        for stale_id in stale_ids:
            await vector_store.delete_document(user_id=user_id, doc_id=stale_id)
    except Exception as exc:
        # Roll back the new version so a retry redoes the whole replacement.
        try:
            await vector_store.delete_document(user_id=user_id, doc_id=doc_id)
        except Exception:
            pass
        raise VectorStoreOperationError(
            f"Indexed '{filename}' but could not remove its previous version."
        ) from exc

    return IngestResult(doc_id, filename, chunks_count, "replaced", tuple(stale_ids))