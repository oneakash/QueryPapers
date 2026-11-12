"""Unit tests for Core Configuration and Dependency Inversion Protocols."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.core.config import Settings, get_settings
from app.core.interfaces import LLMProtocol, VectorStoreProtocol
from tests.unit.fakes import FakeLLM, FakeVectorStore


def test_settings_defaults_and_caching() -> None:
    """Verify Settings loads defaults cleanly and singleton cache works."""
    settings_a = get_settings()
    settings_b = get_settings()
    assert settings_a is settings_b
    assert isinstance(settings_a, Settings)
    assert settings_a.hitl_confidence_threshold == 0.60


def test_fakes_satisfy_runtime_protocols() -> None:
    """Verify FakeVectorStore and FakeLLM satisfy structural typing protocols."""
    store = FakeVectorStore()
    llm = FakeLLM()
    assert isinstance(store, VectorStoreProtocol)
    assert isinstance(llm, LLMProtocol)


@pytest.mark.asyncio
async def test_fake_vector_store_enforces_tenant_isolation() -> None:
    """Ensure user_b cannot retrieve chunks uploaded by user_a."""
    store = FakeVectorStore(default_score=0.92)
    await store.upsert_chunks(
        user_id="user_a",
        doc_id="doc_1",
        chunks=["Confidential Q4 revenue report for Project Apollo."],
    )

    # Query as user_b with identical keywords -> must return 0 chunks
    results_b = await store.search(query="Project Apollo revenue", user_id="user_b")
    assert results_b == []

    # Query as user_a -> must return the matching chunk with score
    results_a = await store.search(query="Project Apollo revenue", user_id="user_a")
    assert len(results_a) == 1
    assert results_a[0].user_id == "user_a"
    assert results_a[0].score == 0.92


@pytest.mark.asyncio
async def test_fake_vector_store_delete_document_is_tenant_scoped() -> None:
    """Ensure deleting a document only removes chunks for the owning user_id."""
    store = FakeVectorStore()
    await store.upsert_chunks("user_a", "doc_shared_id", ["Paris travel guide."])
    await store.upsert_chunks("user_b", "doc_shared_id", ["Paris restaurant list."])

    # user_a deletes their document
    await store.delete_document(user_id="user_a", doc_id="doc_shared_id")

    assert await store.search("Paris", user_id="user_a") == []
    assert len(await store.search("Paris", user_id="user_b")) == 1


@pytest.mark.asyncio
async def test_fake_llm_coreference_rewrite_and_generation() -> None:
    """Verify FakeLLM resolves coreferences from history and formats answers."""
    llm = FakeLLM()
    history = [
        HumanMessage(content="What is the capital of France?"),
        AIMessage(content="The capital of France is Paris."),
    ]

    rewritten = await llm.rewrite_query(
        history=history,
        query="What are famous tourist destinations here?",
    )
    assert "in paris" in rewritten.lower()
    assert llm.rewrite_calls == 1
