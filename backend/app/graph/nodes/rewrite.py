"""Query contextualization node with first-turn short-circuit and trimming."""

from typing import Any

from langchain_core.messages import BaseMessage, trim_messages

from app.core.interfaces import LLMProtocol
from app.core.logging import log_node_timing
from app.graph.state import AgentState

def make_rewrite_node(
    llm: LLMProtocol,
    max_history_messages: int = 5,
) -> Any:
    """Create the contextualize_query node bound to the injected LLMProtocol."""

    @log_node_timing("rewrite")
    async def contextualize_query(state: AgentState) -> dict[str, Any]:
        """Rewrite follow-up queries using recent conversation history."""
        original_query = state.get("original_query", "").strip()
        messages: list[BaseMessage] = list(state.get("messages", []))

        # Short-circuit on first turn: no prior history to resolve coreferences
        if len(messages) <= 1:
            return {"rewritten_query": original_query}

        trimmed_history = trim_messages(
            messages[:-1],
            max_tokens=max_history_messages,
            token_counter=len,
            strategy="last",
            allow_partial=False,
        )
        rewritten = await llm.rewrite_query(
            history=trimmed_history,
            query=original_query,
        )
        return {"rewritten_query": rewritten.strip() or original_query}

    return contextualize_query