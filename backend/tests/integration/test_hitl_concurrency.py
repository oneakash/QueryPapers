"""Integration tests for HITL concurrency, thread ownership, and health."""

import asyncio

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import create_app
from tests.unit.fakes import FakeLLM, FakeVectorStore


@pytest.mark.asyncio
async def test_hitl_double_resume_and_abandoned_interrupt_guards() -> None:
    """Verify 409 on abandoned interrupt and [200, 409] on concurrent /resume."""
    # Score 0.50 (< 0.75) forces the graph to pause at human_review
    store = FakeVectorStore(default_score=0.50)
    await store.upsert_chunks(
        user_id="user_a",
        doc_id="doc_sec",
        chunks=["Confidential deployment runbook for production servers."],
    )
    llm = FakeLLM()
    app = create_app(vector_store=store, llm=llm)

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        # 1. Trigger low-confidence query -> pauses at awaiting_approval
        chat_res = await client.post(
            "/chat",
            headers={"X-User-ID": "user_a"},
            json={
                "thread_id": "thread_race_1",
                "query": "Show deployment runbook",
            },
        )
        assert chat_res.status_code == 200
        assert chat_res.json()["status"] == "awaiting_approval"

        # 2. Abandoned Interrupt: sending another /chat on same thread -> 409 Conflict
        abandoned_res = await client.post(
            "/chat",
            headers={"X-User-ID": "user_a"},
            json={
                "thread_id": "thread_race_1",
                "query": "Another question while paused",
            },
        )
        assert abandoned_res.status_code == 409

        # 3. Cross-Tenant Thread Hijack: user_b tries to resume user_a's thread -> 403
        hijack_res = await client.post(
            "/resume",
            headers={"X-User-ID": "user_b"},
            json={
                "thread_id": "thread_race_1",
                "approved": True,
                "feedback": "Trying to hijack.",
            },
        )
        assert hijack_res.status_code == 403

        # 4. Concurrent Double-Resume Race: fire two /resume calls simultaneously
        resume_payload = {
            "thread_id": "thread_race_1",
            "approved": True,
            "feedback": "Approved by human reviewer.",
        }
        res1, res2 = await asyncio.gather(
            client.post(
                "/resume",
                headers={"X-User-ID": "user_a"},
                json=resume_payload,
            ),
            client.post(
                "/resume",
                headers={"X-User-ID": "user_a"},
                json=resume_payload,
            ),
        )
        statuses = sorted([res1.status_code, res2.status_code])
        assert statuses == [200, 409]


@pytest.mark.asyncio
async def test_thread_ownership_blocks_cross_tenant_chat() -> None:
    """Verify user_b cannot send POST /chat to a thread_id owned by user_a."""
    store = FakeVectorStore(default_score=0.90)
    await store.upsert_chunks("user_a", "doc_1", ["Paris travel notes."])
    app = create_app(vector_store=store, llm=FakeLLM())

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        res_a = await client.post(
            "/chat",
            headers={"X-User-ID": "user_a"},
            json={"thread_id": "thread_owned_by_a", "query": "Paris travel"},
        )
        assert res_a.status_code == 200

        res_b = await client.post(
            "/chat",
            headers={"X-User-ID": "user_b"},
            json={"thread_id": "thread_owned_by_a", "query": "Paris travel"},
        )
        assert res_b.status_code == 403


@pytest.mark.asyncio
async def test_deep_health_endpoint_reports_503_when_degraded() -> None:
    """Verify GET /health returns 200 when healthy and 503 when Qdrant is down."""
    app = create_app(vector_store=FakeVectorStore(), llm=FakeLLM())

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        ok_res = await client.get("/health")
        assert ok_res.status_code == 200
        assert ok_res.json()["status"] == "ok"

        # Simulate Qdrant outage
        async def _degraded_probe() -> tuple[bool, bool]:
            return True, False

        app.state.health_probe = _degraded_probe
        deg_res = await client.get("/health")
        assert deg_res.status_code == 503
        assert deg_res.json()["qdrant"] is False
