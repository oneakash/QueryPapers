"""Finalize node that appends AIMessage, sets answer, and builds sources from chunks."""

import re
from typing import Any

from langchain_core.messages import AIMessage

from app.core.interfaces import ScoredChunk, Source
from app.core.logging import log_node_timing
from app.graph.state import AgentState

_CITATION_PATTERN = re.compile(r"\[(\d+(?:\\s*,\\s*\\d+)*)\]")

def _extract_cited_indices(answer: str, max_index: int) -> list[int]:
    """Return sorted 1-based chunk ids referenced as [N] or [N, M] in the answer."""
    cited: set[int] = set()
    for match in _CITATION_PATTERN.finditer(answer):
        for part in match.group(1).split(","):
            number = int(part)
            if 1 <= number <= max_index:
                cited.add(number)
    return sorted(cited)

def _build_sources(
    answer: str,
    chunks: list[ScoredChunk],
) -> list[Source]:
    """Build sources list from retrieved chunks using [N] markers in the answer."""
    if not answer or not chunks:
        return []

    cited_indices = _extract_cited_indices(answer, len(chunks))
    sources: list[Source] = []
    for number in cited_indices:
        chunk = chunks[number - 1]
        sources.append(
            Source(
                kind="document",
                index=number,
                title=chunk.filename or chunk.doc_id,
                doc_id=chunk.doc_id,
                filename=chunk.filename or chunk.doc_id,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                score=chunk.score,
            )
        )
    return sources

def make_finalize_node() -> Any:
    """Create the finalize node that processes draft and builds citations."""

    @log_node_timing("finalize")
    async def finalize(state: AgentState) -> dict[str, Any]:
        """Append AIMessage, set answer, and build sources from retrieved chunks."""
        draft = state.get("draft")
        chunks = state.get("retrieved_chunks", [])

        # Build sources from chunks and citation markers in draft
        sources = _build_sources(draft or "", chunks)

        # Append AIMessage to messages
        messages = list(state.get("messages", []))
        if draft:
            messages.append(AIMessage(content=draft))

        return {
            "answer": draft,
            "messages": messages,
            "sources": sources,
        }

    return finalize