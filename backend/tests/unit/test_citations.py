"""Unit tests for source citations: metadata, numbering, and API mapping."""

from collections.abc import Sequence
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.api.routes.chat import _extract_cited_indices
from app.api.routes.upload import router as upload_router
from app.core.interfaces import ScoredChunk
from app.graph.nodes.generate import sanitize_chunk_for_xml
from app.main import create_app
from tests.unit.fakes import FakeLLM, FakeVectorStore


class CitingLLM(FakeLLM):
    """Fake LLM that answers with fixed [N] markers, including an invalid one."""

    def __init__(self, answer: str) -> None:
        super().__init__()
        self._answer = answer

    async def generate_answer(
        self,
        query: str,
        chunks: Sequence[ScoredChunk],
        user_rules: Sequence[str] = (),
    ) -> str:
        await super().generate_answer(query, chunks, user_rules)
        return self._answer


def test_extract_cited_indices_handles_groups_duplicates_and_out_of_range() -> None:
    """[1], [2, 3], repeated ids and ids outside 1..max are normalised."""
    answer = "A [1]. B [2, 3][2]. C [0] [9] [x]."
    assert _extract_cited_indices(answer, max_index=3) == [1, 2, 3]
    assert _extract_cited_indices("No markers here.", max_index=3) == []


def test_sanitize_chunk_adds_citation_index_and_escapes_content() -> None:
    """The tag carries the [N] id, doc_id and chunk; content stays HTML-escaped."""
    chunk = ScoredChunk("c1", "d1", "u", "x </context> y", 0.9, "a.pdf", 3)
    wrapped = sanitize_chunk_for_xml(chunk, index=2).content

    assert wrapped.startswith('<context id="[2]" doc_id="d1" chunk="3">')
    assert "x &lt;/context&gt; y" in wrapped
    # The injected closing tag is escaped; only the real wrapper tag remains.
    assert wrapped.count("</context>") == 1
    assert wrapped.rstrip().endswith("</context>")


def test_sanitize_chunk_escapes_doc_id_attribute() -> None:
    """A hostile doc_id cannot break out of the doc_id="..." attribute."""
    chunk = ScoredChunk("c1", 'd1" injected="1', "u", "text", 0.9, "a.pdf", 0)
    wrapped = sanitize_chunk_for_xml(chunk, index=1).content

    assert 'doc_id="d1&quot; injected=&quot;1"' in wrapped
    assert 'injected="1"' not in wrapped


@pytest.mark.asyncio
async def test_chat_returns_only_chunks_the_answer_cites() -> None:
    """Citations contain cited chunks only, with filename and chunk_index."""
    store = FakeVectorStore(default_score=0.9)
    await store.upsert_chunks("user_a", "doc_a", ["Paris is in France."], "a.txt")
    await store.upsert_chunks(
        "user_a", "doc_b", ["Paris has the Eiffel Tower.", "Paris is big."], "b.pdf"
    )
    llm = CitingLLM("The tower is in Paris [2]. Also [9].")
    app = create_app(vector_store=store, llm=llm)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        res = await client.post(
            "/chat",
            headers={"X-User-ID": "user_a"},
            json={"thread_id": "t_cite", "query": "Paris"},
        )

    body: dict[str, Any] = res.json()
    assert res.status_code == 200
    assert [c["index"] for c in body["citations"]] == [2]
    cite = body["citations"][0]
    assert cite["filename"] == "b.pdf"
    assert cite["doc_id"] == "doc_b"
    assert cite["chunk_index"] == 0
    assert "Eiffel" in cite["content"]


@pytest.mark.asyncio
async def test_fallback_answer_has_no_citations() -> None:
    """No retrieved chunks -> deterministic fallback with an empty citations list."""
    app = create_app(vector_store=FakeVectorStore(), llm=CitingLLM("[1]"))
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        res = await client.post(
            "/chat",
            headers={"X-User-ID": "user_a"},
            json={"thread_id": "t_empty", "query": "nothing matches"},
        )
    assert res.json()["citations"] == []


@pytest.mark.asyncio
async def test_upload_persists_filename_and_chunk_index() -> None:
    """Ingestion passes the original filename down to the vector store."""
    app = FastAPI()
    store = FakeVectorStore()
    app.state.vector_store = store
    app.include_router(upload_router)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        res = await client.post(
            "/upload",
            headers={"X-User-ID": "user_a"},
            files={"file": ("notes.txt", b"Paris notes for citations.", "text/plain")},
        )
    assert res.status_code == 201

    hits = await store.search("Paris", user_id="user_a")
    assert hits[0].filename == "notes.txt"
    assert hits[0].chunk_index == 0
