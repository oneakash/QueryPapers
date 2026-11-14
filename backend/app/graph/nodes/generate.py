"""RAG answer generation node with XML sanitization, citations, and LTM rule."""

import html
from dataclasses import replace
from typing import Any

from langchain_core.messages import AIMessage
from langgraph.store.base import BaseStore

from app.core.interfaces import LLMProtocol, ScoredChunk
from app.core.logging import extract_generate_fields, log_node_timing
from app.graph.state import AgentState

EMPTY_RETRIEVAL_FALLBACK = (
    "I couldn't find relevant information in your uploaded documents "
    "to answer that question."
)
REJECTED_BY_REVIEWER_FALLBACK = (
    "The retrieved context was rejected during human review."
)

def sanitize_chunk_for_xml(chunk: ScoredChunk, index: int = 1) -> ScoredChunk:
    """Escape XML/HTML brackets in chunk content and attach 1-based citation index."""
    safe_text = html.escape(chunk.content, quote=False)
    doc_id = html.escape(str(getattr(chunk, "doc_id", "unknown")), quote=True)
    chunk_idx = getattr(chunk, "chunk_index", 0)
    return replace(
        chunk,
        content=f"""<context id="[{index}]" doc_id="{doc_id}" chunk="{chunk_idx}">
            {safe_text}
        </context>""",
    )

def make_generate_node(
    llm: LLMProtocol,
    max_user_rules: int = 10,
) -> Any:
    """Create the generate node bound to LLMProtocol and optional BaseStore."""

    @log_node_timing("generate", extract_fields=extract_generate_fields)
    async def generate(
        state: AgentState,
        *,
        store: BaseStore | None = None,
    ) -> dict[str, Any]:
        """Generate grounded answer or return deterministic fallback."""
        chunks = state.get("retrieved_chunks", [])

        # 1. Zero-chunk short-circuit: never call generation LLM
        if not chunks:
            return {
                "draft": EMPTY_RETRIEVAL_FALLBACK,
            }

        # 2. Human reviewer rejected the low-confidence retrieval
        if state.get("hitl_approved") is False:
            feedback = state.get("hitl_feedback")
            msg = (
                f"Reviewer override: {feedback}"
                if feedback
                else REJECTED_BY_REVIEWER_FALLBACK
            )
            return {
                "draft": msg,
            }

        # 3. Sanitize chunks against XML prompt injection & number them [1], [2]...
        safe_chunks = [
            sanitize_chunk_for_xml(c, index=idx)
            for idx, c in enumerate(chunks, start=1)
        ]

        # 4. Load bounded tenant rules from Long-Term Memory Store
        #    Sort by created_at to ensure stable ordering across stores
        user_rules: list[str] = []
        if store is not None:
            namespace = ("user_rules", state["user_id"])
            items = await store.asearch(namespace, limit=max_user_rules + 5)
            # Extract rules with their timestamps and sort by created_at (oldest first)
            items_with_ts = []
            for item in items:
                if isinstance(item.value, dict) and "rule" in item.value:
                    items_with_ts.append(
                        {
                            "rule": str(item.value["rule"]),
                            "created_at": item.value.get("created_at", 0),
                        }
                    )
            # Sort by created_at for stable ordering
            items_with_ts.sort(key=lambda x: x["created_at"])
            user_rules = [item["rule"] for item in items_with_ts[:max_user_rules]]

        feedback = state.get("hitl_feedback")
        if feedback and feedback not in user_rules:
            user_rules.append(feedback)

        query = state.get("rewritten_query") or state["original_query"]
        answer = await llm.generate_answer(
            query=query,
            chunks=safe_chunks,
            user_rules=user_rules,
        )
        return {
            "draft": answer,
        }

    return generate