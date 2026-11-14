"""Human-in-the-Loop (HITL) interrupt node and Long-Term Memory rule saver."""

import time
import uuid
from typing import Any

from langgraph.store.base import BaseStore
from langgraph.types import interrupt

from app.core.logging import extract_hitl_fields, log_node_timing
from app.graph.state import AgentState

def make_hitl_review_node(max_user_rules: int = 10) -> Any:
    """Create the human_review node that triggers interrupt() and stores LTM rules."""

    @log_node_timing("human_review", extract_fields=extract_hitl_fields)
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

            # Build list with created_at timestamps
            items_with_ts = []
            for item in existing:
                if isinstance(item.value, dict):
                    items_with_ts.append(
                        {
                            "key": item.key,
                            "created_at": item.value.get("created_at", 0),
                        }
                    )

            # Sort by created_at (oldest first) and evict oldest if at capacity
            if len(existing) >= max_user_rules:
                items_with_ts.sort(key=lambda x: x["created_at"])
                for old_item in items_with_ts[: len(existing) - max_user_rules + 1]:
                    await store.adelete(namespace, old_item["key"])

            rule_key = f"rule_{uuid.uuid4().hex[:8]}"
            await store.aput(
                namespace,
                rule_key,
                {"rule": feedback, "created_at": time.time()},
            )

        return {
            "hitl_approved": approved,
            "hitl_feedback": feedback,
        }

    return human_review