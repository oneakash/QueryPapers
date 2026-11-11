"""Unit tests for GeminiClient (gemini-embedding-2 and gemini-3.6-flash)."""

import json
from typing import Any

import httpx
import pytest

from app.core.config import Settings
from app.core.interfaces import LLMProtocol, ScoredChunk
from app.services.llm_client import (
    GeminiClient,
    OpenAICompatibleClient,
    create_embedding_client,
    create_llm_client,
)
from app.services.qdrant_store import EmbeddingClientProtocol


def test_gemini_client_satisfies_protocols() -> None:
    """Verify GeminiClient structurally satisfies LLM and Embedding protocols."""
    client = GeminiClient(api_key="test-key")
    assert isinstance(client, LLMProtocol)
    assert isinstance(client, EmbeddingClientProtocol)


@pytest.mark.asyncio
async def test_gemini_embedding_2_batch_and_query_task_types() -> None:
    """Verify batchEmbedContents and embedContent pass taskType and dimensions."""
    captured_requests: list[tuple[str, dict[str, Any]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        captured_requests.append((request.url.path, body))
        if "batchEmbedContents" in request.url.path:
            return httpx.Response(
                200,
                json={
                    "embeddings": [
                        {"values": [0.1, 0.2, 0.3]},
                        {"values": [0.4, 0.5, 0.6]},
                    ]
                },
            )
        return httpx.Response(200, json={"embedding": {"values": [0.9, 0.8, 0.7]}})

    client = GeminiClient(
        api_key="free-tier-key",
        embedding_model="gemini-embedding-2",
        embedding_dimensions=768,
    )
    client._http = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="https://generativelanguage.googleapis.com/v1beta",
    )

    doc_vecs = await client.embed_documents(["Chunk 1", "Chunk 2"])
    query_vec = await client.embed_query("Search query")
    await client.close()

    assert len(doc_vecs) == 2
    assert query_vec == [0.9, 0.8, 0.7]

    batch_path, batch_body = captured_requests[0]
    assert batch_path.endswith("/models/gemini-embedding-2:batchEmbedContents")
    reqs = batch_body["requests"]
    assert reqs[0]["taskType"] == "RETRIEVAL_DOCUMENT"
    assert reqs[0]["outputDimensionality"] == 768

    query_path, query_body = captured_requests[1]
    assert query_path.endswith("/models/gemini-embedding-2:embedContent")
    assert query_body["taskType"] == "RETRIEVAL_QUERY"
    assert query_body["outputDimensionality"] == 768


@pytest.mark.asyncio
async def test_gemini_client_retries_on_429_rate_limit() -> None:
    """Verify automatic retry when Google AI Studio Free Tier returns 429."""
    attempts = 0

    def flaky_handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, json={"error": "RESOURCE_EXHAUSTED"})
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {"content": {"parts": [{"text": "Grounded Gemini answer."}]}}
                ]
            },
        )

    client = GeminiClient(api_key="free-tier-key", max_retries=2)
    client._http = httpx.AsyncClient(
        transport=httpx.MockTransport(flaky_handler),
        base_url="https://generativelanguage.googleapis.com/v1beta",
    )

    chunk = ScoredChunk(
        chunk_id="c1",
        doc_id="d1",
        user_id="user_a",
        content="<context>Paris info</context>",
        score=0.91,
    )
    answer = await client.generate_answer("Where is Paris?", [chunk])
    await client.close()

    assert attempts == 2
    assert answer == "Grounded Gemini answer."


def test_model_factories_switch_providers_via_settings() -> None:
    """Verify factories return GeminiClient or OpenAICompatibleClient from Settings."""
    gemini_cfg = Settings(
        llm_provider="gemini",
        llm_model="gemini-3.6-flash",
        embedding_provider="gemini",
        embedding_model="gemini-embedding-2",
    )
    assert isinstance(create_llm_client(gemini_cfg), GeminiClient)
    assert isinstance(create_embedding_client(gemini_cfg), GeminiClient)

    groq_cfg = Settings(
        llm_provider="openai_compatible",
        llm_model="llama-3.3-70b-versatile",
        llm_base_url="https://api.groq.com/openai/v1",
        embedding_provider="openai_compatible",
        embedding_model="text-embedding-3-small",
    )
    assert isinstance(create_llm_client(groq_cfg), OpenAICompatibleClient)
    assert isinstance(create_embedding_client(groq_cfg), OpenAICompatibleClient)
