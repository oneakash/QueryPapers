"""Deterministic conditional routing logic for the LangGraph workflow."""

from collections.abc import Callable
from typing import Literal

from app.graph.state import AgentState


def make_grade_and_route_edge(
    confidence_threshold: float = 0.60,
) -> Callable[[AgentState], Literal["human_review", "generate"]]:
    """Create the conditional router comparing top_score to confidence_threshold."""

    def grade_and_route(
        state: AgentState,
    ) -> Literal["human_review", "generate"]:
        """Route low-confidence non-empty retrievals to human_review."""
        chunks = state.get("retrieved_chunks", [])
        if not chunks:
            return "generate"

        top_score = state.get("top_score", 0.0)
        if top_score < confidence_threshold:
            return "human_review"

        return "generate"

    return grade_and_route
