"""Unit tests for Step 3: Ingestion pipeline, rollback, and Qdrant filter isolation."""

from typing import Any

import pymupdf
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.api.routes.upload import router as upload_router
from app.services.qdrant_store import QdrantVectorStore
from tests.unit.fakes import FakeVectorStore

_pymupdf: Any = pymupdf


def _create_sample_pdf_bytes(text: str | None = "Hello Paris RAG world.") -> bytes:
    """Create an in-memory PDF with optional text for deterministic testing."""
    doc: Any = _pymupdf.open()
    page = doc.new_page()
    if text:
        page.insert_text((72, 72), text)
    pdf_bytes = bytes(doc.tobytes())
    doc.close()
    return pdf_bytes


@pytest.fixture
def test_app() -> tuple[FastAPI, FakeVectorStore]:
    """Create an isolated FastAPI app wired with FakeVectorStore."""
    app = FastAPI()
    store = FakeVectorStore()
    app.state.vector_store = store
    app.include_router(upload_router)
    return app, store


@pytest.mark.asyncio
async def test_upload_valid_pdf_indexes_chunks(
    test_app: tuple[FastAPI, FakeVectorStore],
) -> None:
    """Verify valid PDF upload extracts text and stores chunks under X-User-ID."""
    app, store = test_app
    pdf_bytes = _create_sample_pdf_bytes("Eiffel Tower is located in Paris.")

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/upload",
            headers={"X-User-ID": "user_a"},
            files={"file": ("paris.pdf", pdf_bytes, "application/pdf")},
        )

    assert response.status_code == 201
    data = response.json()
    assert data["chunks_indexed"] == 1

    # Confirm chunk is searchable by user_a but invisible to user_b
    assert len(await store.search("Eiffel Tower", user_id="user_a")) == 1
    assert await store.search("Eiffel Tower", user_id="user_b") == []


@pytest.mark.asyncio
async def test_upload_rejects_empty_and_corrupted_files(
    test_app: tuple[FastAPI, FakeVectorStore],
) -> None:
    """Verify 0-byte, image-only PDF, and fake binary files return 400 Bad Request."""
    app, _ = test_app
    empty_page_pdf = _create_sample_pdf_bytes(text=None)

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        # 1. 0-byte PDF
        res_zero = await client.post(
            "/upload",
            headers={"X-User-ID": "user_a"},
            files={"file": ("empty.pdf", b"", "application/pdf")},
        )
        # 2. Fake .exe renamed to .pdf
        res_corrupt = await client.post(
            "/upload",
            headers={"X-User-ID": "user_a"},
            files={"file": ("malware.pdf", b"MZ\x90\x00not_a_pdf", "application/pdf")},
        )
        # 3. Blank/scanned PDF with 0 text
        res_blank = await client.post(
            "/upload",
            headers={"X-User-ID": "user_a"},
            files={"file": ("scanned.pdf", empty_page_pdf, "application/pdf")},
        )

    assert res_zero.status_code == 400
    assert res_corrupt.status_code == 400
    assert res_blank.status_code == 400


@pytest.mark.asyncio
async def test_upload_rolls_back_partial_chunks_on_failure(
    test_app: tuple[FastAPI, FakeVectorStore],
) -> None:
    """Verify failed upsert rolls back partially inserted chunks for that doc_id."""
    app, store = test_app
    store.fail_on_upsert = True
    pdf_bytes = _create_sample_pdf_bytes("Rollback test content.")

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/upload",
            headers={"X-User-ID": "user_a"},
            files={"file": ("fail.pdf", pdf_bytes, "application/pdf")},
        )

    assert response.status_code == 503
    # Confirm partial chunk inserted before failure was rolled back
    assert await store.search("Rollback test", user_id="user_a") == []


def test_qdrant_tenant_filter_builder_enforces_user_id() -> None:
    """Verify QdrantVectorStore.build_tenant_filter always sets mandatory user_id."""
    q_filter = QdrantVectorStore.build_tenant_filter(
        user_id="tenant_42",
        doc_id="doc_99",
    )
    assert q_filter.must is not None
    keys_and_vals = {
        cond.key: cond.match.value for cond in q_filter.must  # type: ignore[union-attr]
    }
    assert keys_and_vals["user_id"] == "tenant_42"
    assert keys_and_vals["doc_id"] == "doc_99"

    with pytest.raises(ValueError, match="user_id is required"):
        QdrantVectorStore.build_tenant_filter(user_id="   ")
