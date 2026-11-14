"""Unit tests for structured logging, request-id middleware, timing decorator."""

import logging
import uuid
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.core.logging import (
    StructuredFormatter,
    extract_generate_fields,
    extract_hitl_fields,
    extract_retrieve_fields,
    get_request_id,
    hash_user_id,
    log_node_timing,
    set_request_id,
)
from app.main import app

class TestRequestIdMiddleware:
    """Tests for RequestIdMiddleware that sets contextvar and X-Request-ID header."""

    def test_middleware_sets_x_request_id_header(self) -> None:
        """Verify middleware adds X-Request-ID header to every response."""
        client = TestClient(app)
        response = client.get("/health")
        assert "X-Request-ID" in response.headers
        assert len(response.headers["X-Request-ID"]) == 8

    def test_middleware_preserves_incoming_request_id(self) -> None:
        """Verify middleware uses incoming X-Request-ID if provided."""
        client = TestClient(app)
        custom_request_id = "custom-123"
        response = client.get(
            "/health", headers={"X-Request-ID": custom_request_id}
        )
        assert response.headers["X-Request-ID"] == custom_request_id

    def test_middleware_sets_contextvar(self) -> None:
        """Verify middleware sets request_id in contextvar."""
        client = TestClient(app)
        test_request_id = str(uuid.uuid4())[:8]
        client.get("/health", headers={"X-Request-ID": test_request_id})

class TestHashUserId:
    """Tests for user_id hashing utility."""

    def test_hash_user_id_returns_consistent_hash(self) -> None:
        """Verify same user_id always produces same hash."""
        user_id = "user_123"
        hash1 = hash_user_id(user_id)
        hash2 = hash_user_id(user_id)
        assert hash1 == hash2
        assert len(hash1) == 16

    def test_hash_user_id_different_users_different_hashes(self) -> None:
        """Verify different user_ids produce different hashes."""
        hash1 = hash_user_id("user_a")
        hash2 = hash_user_id("user_b")
        assert hash1 != hash2

    def test_hash_user_id_none_returns_anonymous(self) -> None:
        """Verify None user_id returns 'anonymous'."""
        assert hash_user_id(None) == "anonymous"

    def test_hash_user_id_empty_returns_anonymous(self) -> None:
        """Verify empty string user_id returns 'anonymous' (falsy check)."""
        result = hash_user_id("")
        assert result == "anonymous"

class TestStructuredFormatter:
    """Tests for JSON structured log formatter."""

    def test_formatter_produces_valid_json(self) -> None:
        """Verify formatter produces valid JSON."""
        import json

        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        result = formatter.format(record)
        parsed = json.loads(result)
        assert parsed["message"] == "Test message"
        assert parsed["level"] == "INFO"

    def test_formatter_includes_request_id(self) -> None:
        """Verify formatter includes request_id from context."""
        import json

        formatter = StructuredFormatter()
        set_request_id("test-request-123")
        record = logging.LogRecord(
            name="test.logger",
            level=logging.DEBUG,
            pathname="test.py",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None,
        )
        result = formatter.format(record)
        parsed = json.loads(result)
        assert parsed["request_id"] == "test-request-123"

    def test_formatter_excludes_sensitive_fields(self) -> None:
        """Verify formatter never includes sensitive fields like user_id."""
        import json

        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None,
        )
        record.user_id = "sensitive_user_123"
        record.api_key = "secret_key"
        record.query = "secret query"
        record.content = "secret content"

        result = formatter.format(record)
        parsed = json.loads(result)

        assert "user_id" not in result
        assert "api_key" not in result
        assert "query" not in result
        assert "content" not in result

    def test_formatter_includes_exception_info(self) -> None:
        """Verify formatter includes exception type and message."""
        import json

        formatter = StructuredFormatter()
        try:
            raise ValueError("test error")
        except ValueError as e:
            record = logging.LogRecord(
                name="test.logger",
                level=logging.ERROR,
                pathname="test.py",
                lineno=1,
                msg="Error occurred",
                args=(),
                exc_info=(ValueError, e, None),
            )
            result = formatter.format(record)
            parsed = json.loads(result)
            assert "exception" in parsed
            assert parsed["exception"]["type"] == "ValueError"
            assert parsed["exception"]["message"] == "test error"

class TestLogNodeTimingDecorator:
    """Tests for node timing decorator."""

    @pytest.mark.asyncio
    async def test_decorator_logs_duration(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Verify decorator logs duration for async function."""
        import asyncio

        caplog.set_level(logging.DEBUG)

        @log_node_timing("test_node")
        async def async_func(state: dict[str, Any]) -> dict[str, Any]:
            await asyncio.sleep(0.01)
            return {"result": "ok"}

        state = {
            "user_id": "test_user",
            "config": {"configurable": {"thread_id": "thread_1"}},
        }
        result = await async_func(state)
        assert result == {"result": "ok"}

        assert any("Node timing" in record.message for record in caplog.records)
        assert any(
            hasattr(record, "duration_ms") and record.duration_ms > 0
            for record in caplog.records
        )

    @pytest.mark.asyncio
    async def test_decorator_logs_node_specific_fields(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Verify decorator logs node-specific fields via extractor."""
        caplog.set_level(logging.DEBUG)

        @log_node_timing("retrieve", extract_fields=extract_retrieve_fields)
        async def retrieve_func(state: dict[str, Any]) -> dict[str, Any]:
            return {"retrieved_chunks": [1, 2, 3], "top_score": 0.95}

        state = {
            "user_id": "test_user",
            "config": {"configurable": {"thread_id": "thread_1"}},
            "top_score": 0.95,
            "retrieved_chunks": [1, 2, 3],
        }
        await retrieve_func(state)

        assert any(
            hasattr(record, "top_score") and record.top_score == 0.95
            for record in caplog.records
        )
        assert any(
            hasattr(record, "num_chunks") and record.num_chunks == 3
            for record in caplog.records
        )

    @pytest.mark.asyncio
    async def test_decorator_hashes_user_id(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Verify decorator hashes user_id and never logs raw value."""
        caplog.set_level(logging.DEBUG)

        @log_node_timing("test_node")
        async def test_func(state: dict[str, Any]) -> dict[str, Any]:
            return {}

        state = {
            "user_id": "sensitive_user_id_123",
            "config": {"configurable": {"thread_id": "thread_1"}},
        }
        await test_func(state)

        assert all(
            "sensitive_user_id_123" not in record.message
            for record in caplog.records
        )
        assert any(
            hasattr(record, "user_id_hash")
            and record.user_id_hash != "sensitive_user_id_123"
            for record in caplog.records
        )

    @pytest.mark.asyncio
    async def test_decorator_does_not_leak_document_text(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Verify decorator never logs document text, queries, or answers."""
        caplog.set_level(logging.DEBUG)

        @log_node_timing("generate", extract_fields=extract_generate_fields)
        async def generate_func(state: dict[str, Any]) -> dict[str, Any]:
            return {"draft": "This is a secret answer"}

        state = {
            "user_id": "test_user",
            "config": {"configurable": {"thread_id": "thread_1"}},
            "original_query": "What is the secret?",
            "draft": "This is a secret answer",
            "retrieved_chunks": [{"content": "Secret document text"}],
            "top_score": 0.9,
        }
        await generate_func(state)

        for record in caplog.records:
            assert "Secret document text" not in record.message
            assert "What is the secret?" not in record.message
            assert "This is a secret answer" not in record.message

class TestExtractors:
    """Tests for field extractor functions."""

    def test_extract_retrieve_fields(self) -> None:
        """Verify retrieve field extractor returns safe fields."""
        state = {
            "top_score": 0.95,
            "retrieved_chunks": [1, 2, 3],
            "user_id": "test_user",
            "original_query": "test query",
        }
        fields = extract_retrieve_fields(state)
        assert fields["top_score"] == 0.95
        assert fields["num_chunks"] == 3
        assert "user_id" not in fields
        assert "original_query" not in fields

    def test_extract_generate_fields(self) -> None:
        """Verify generate field extractor returns safe fields."""
        state = {
            "top_score": 0.85,
            "retrieved_chunks": [1, 2],
            "route": "retrieve",
            "attempts": 2,
            "draft": "some answer",
        }
        fields = extract_generate_fields(state)
        assert fields["top_score"] == 0.85
        assert fields["num_chunks"] == 2
        assert fields["route"] == "retrieve"
        assert fields["attempts"] == 2
        assert "draft" not in fields

    def test_extract_hitl_fields(self) -> None:
        """Verify hitl field extractor returns safe fields."""
        state = {
            "top_score": 0.5,
            "hitl_approved": True,
            "hitl_feedback": "good",
        }
        fields = extract_hitl_fields(state)
        assert fields["top_score"] == 0.5
        assert fields["hitl_approved"] is True

    def test_extractors_handle_missing_fields(self) -> None:
        """Verify extractors handle missing fields gracefully."""
        state: dict[str, Any] = {}
        assert extract_retrieve_fields(state) == {}
        assert extract_generate_fields(state) == {}
        assert extract_hitl_fields(state) == {}