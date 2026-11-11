"""LangGraph state schema definition for the stateful HITL RAG workflow."""

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from app.core.interfaces import ScoredChunk


class AgentState(TypedDict, total=False):
    """Typed state dictionary flowing through the LangGraph pipeline."""

    messages: Annotated[list[BaseMessage], add_messages]
    user_id: str
    original_query: str
    rewritten_query: str
    retrieved_chunks: list[ScoredChunk]
    top_score: float
    hitl_approved: bool | None
    hitl_feedback: str | None
    answer: str
