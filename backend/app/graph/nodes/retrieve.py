"""Tenant-isolated vector retrieval node."""

from typing import Any

from app.core.interfaces import VectorStoreProtocol
from app.graph.state import AgentState


def make_retrieve_node(
    vector_store: VectorStoreProtocol,
    top_k: int = 5,
) -> Any:
    """Create the retrieve node bound to the injected VectorStoreProtocol."""

    async def retrieve(state: AgentState) -> dict[str, Any]:
        """Retrieve top-k chunks for user_id and record highest similarity score."""
        user_id = state["user_id"]
        query = state.get("rewritten_query") or state["original_query"]

        chunks = await vector_store.search(
            query=query,
            user_id=user_id,
            limit=top_k,
        )
        top_score = max((chunk.score for chunk in chunks), default=0.0)
        return {
            "retrieved_chunks": chunks,
            "top_score": top_score,
        }

    return retrieve
