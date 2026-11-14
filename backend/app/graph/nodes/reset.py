"""Per-turn state reset node to clear transient fields."""

from typing import Any

from app.core.logging import log_node_timing
from app.graph.state import AgentState

def make_reset_node() -> Any:
    """Create the reset node that clears per-turn state fields."""

    @log_node_timing("reset")
    async def reset(state: AgentState) -> dict[str, Any]:
        """Reset all per-turn fields at the start of each turn."""
        return {
            "hitl_approved": None,
            "hitl_feedback": None,
            "route": None,
            "attempts": 0,
            "critique": None,
            "draft": None,
            "sources": [],
            "retrieved_chunks": [],
            "top_score": 0.0,
        }

    return reset