"""Pydantic request and response schemas for chat, HITL resume, and health."""

from typing import Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request payload for POST /chat (user_id comes from X-User-ID header)."""

    thread_id: str = Field(..., min_length=1, max_length=128)
    query: str = Field(..., min_length=1, max_length=4000)
    stream: bool = False


class ResumeRequest(BaseModel):
    """Request payload for POST /resume to resolve a pending HITL interrupt."""

    thread_id: str = Field(..., min_length=1, max_length=128)
    approved: bool
    feedback: str | None = Field(default=None, max_length=2000)


class InterruptPreview(BaseModel):
    """Details returned to the human reviewer when a thread pauses at HITL."""

    question: str
    top_score: float
    chunks_preview: list[str]


class ChatResponse(BaseModel):
    """Unified response for both POST /chat and POST /resume."""

    thread_id: str
    status: Literal["completed", "awaiting_approval"]
    answer: str | None = None
    rewritten_query: str | None = None
    stream: bool = False
    top_retrieval_score: float = 0.0
    interrupt_payload: InterruptPreview | None = None


class HealthResponse(BaseModel):
    """Response schema for GET /health deep readiness probe."""

    status: Literal["ok", "degraded"]
    postgres: bool
    qdrant: bool
