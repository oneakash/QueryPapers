"""HTTP endpoints for document upload and deletion."""

from typing import Annotated, Literal

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Response,
    UploadFile,
    status,
)
from pydantic import BaseModel, Field

from app.api.dependencies import get_current_user_id, get_vector_store
from app.api.exceptions import InvalidDocumentError, VectorStoreOperationError
from app.core.config import Settings, get_settings
from app.core.interfaces import VectorStoreProtocol
from app.ingestion.pipeline import ingest_document_idempotent

router = APIRouter(tags=["ingestion"])


class UploadResponse(BaseModel):
    """Response schema for successful document ingestion."""

    doc_id: str
    filename: str
    chunks_indexed: int
    status: Literal["indexed", "duplicate", "replaced"] = "indexed"
    replaced_doc_ids: list[str] = Field(default_factory=list)


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
    response: Response,
    file: Annotated[UploadFile, File(...)],
    user_id: Annotated[str, Depends(get_current_user_id)],
    vector_store: Annotated[VectorStoreProtocol, Depends(get_vector_store)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> UploadResponse:
    """Ingest a PDF or TXT document scoped to the authenticated user_id.

    Returns 201 for a newly indexed file (or a replacement) and 200 when the
    exact same file was already indexed and nothing was re-embedded.
    """
    filename = file.filename or "uploaded_document.pdf"
    file_bytes = await file.read()

    try:
        result = await ingest_document_idempotent(
            file_bytes=file_bytes,
            filename=filename,
            user_id=user_id,
            vector_store=vector_store,
            replace_same_filename=settings.replace_on_filename_conflict,
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

    if result.status == "duplicate":
        response.status_code = status.HTTP_200_OK

    return UploadResponse(
        doc_id=result.doc_id,
        filename=result.filename,
        chunks_indexed=result.chunks_indexed,
        status=result.status,
        replaced_doc_ids=list(result.replaced_doc_ids),
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
