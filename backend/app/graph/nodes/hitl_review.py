"""Human-in-the-Loop (HITL) interrupt node and Long-Term Memory rule saver."""

import uuid
from typing import Any

from langgraph.store.base import BaseStore
from langgraph.types import interrupt

from app.graph.state import AgentState


def make_hitl_review_node(max_user_rules: int = 10) -> Any:
    """Create the human_review node that triggers interrupt() and stores LTM rules."""

    async def human_review(
        state: AgentState,
        *,
        store: BaseStore | None = None,
    ) -> dict[str, Any]:
        """Pause execution for human approval and persist correction rules."""
        chunks = state.get("retrieved_chunks", [])
        payload = {
            "question": state.get("rewritten_query") or state.get("original_query"),
            "top_score": state.get("top_score", 0.0),
            "chunks_preview": [c.content[:200] for c in chunks],
        }

        review_decision: dict[str, Any] = interrupt(payload)
        approved = bool(review_decision.get("approved", False))
        feedback_raw = review_decision.get("feedback")
        feedback = str(feedback_raw).strip() if feedback_raw else None

        # Persist reusable reviewer feedback in Long-Term Memory Store
        if feedback and store is not None:
            namespace = ("user_rules", state["user_id"])
            existing = await store.asearch(namespace, limit=max_user_rules + 5)
            if len(existing) >= max_user_rules:
                for old_item in existing[: len(existing) - max_user_rules + 1]:
                    await store.adelete(namespace, old_item.key)

            rule_key = f"rule_{uuid.uuid4().hex[:8]}"
            await store.aput(namespace, rule_key, {"rule": feedback})

        return {
            "hitl_approved": approved,
            "hitl_feedback": feedback,
        }

    return human_review
