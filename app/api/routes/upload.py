"""HTTP endpoints for document upload and deletion."""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel

from app.api.dependencies import get_current_user_id, get_vector_store
from app.api.exceptions import InvalidDocumentError, VectorStoreOperationError
from app.core.interfaces import VectorStoreProtocol
from app.ingestion.pipeline import ingest_document

router = APIRouter(tags=["ingestion"])


class UploadResponse(BaseModel):
    """Response schema for successful document ingestion."""

    doc_id: str
    filename: str
    chunks_indexed: int


class DeleteDocumentResponse(BaseModel):
    """Response schema for document deletion."""

    doc_id: str
    deleted: bool


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_document(
    file: Annotated[UploadFile, File(...)],
    user_id: Annotated[str, Depends(get_current_user_id)],
    vector_store: Annotated[VectorStoreProtocol, Depends(get_vector_store)],
) -> UploadResponse:
    """Ingest a PDF or TXT document scoped to the authenticated user_id."""
    filename = file.filename or "uploaded_document.pdf"
    file_bytes = await file.read()
    doc_id = f"doc_{uuid.uuid4().hex[:12]}"

    try:
        chunks_count = await ingest_document(
            file_bytes=file_bytes,
            filename=filename,
            user_id=user_id,
            doc_id=doc_id,
            vector_store=vector_store,
        )
    except InvalidDocumentError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except VectorStoreOperationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    return UploadResponse(
        doc_id=doc_id,
        filename=filename,
        chunks_indexed=chunks_count,
    )


@router.delete(
    "/documents/{doc_id}",
    response_model=DeleteDocumentResponse,
    status_code=status.HTTP_200_OK,
)
async def delete_document(
    doc_id: str,
    user_id: Annotated[str, Depends(get_current_user_id)],
    vector_store: Annotated[VectorStoreProtocol, Depends(get_vector_store)],
) -> DeleteDocumentResponse:
    """Delete all vectors for a specific document owned by the current user."""
    await vector_store.delete_document(user_id=user_id, doc_id=doc_id)
    return DeleteDocumentResponse(doc_id=doc_id, deleted=True)
