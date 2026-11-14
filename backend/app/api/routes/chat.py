"""Chat and HITL resume endpoints supporting both standard JSON and SSE streams."""

import json
import re
from collections.abc import AsyncGenerator
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from app.api.dependencies import ThreadLockProtocol, get_lock_manager
from app.api.schemas import (
    ChatRequest,
    ChatResponse,
    Citation,
    InterruptPreview,
    ResumeRequest,
)
from app.core.interfaces import ScoredChunk, Source

router = APIRouter(tags=["chat"])

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


def _build_citations(
    answer: str | None,
    chunks: list[ScoredChunk],
) -> list[Citation]:
    """Map [N] markers in the answer back to the chunks the generate node numbered."""
    if not answer or not chunks:
        return []
    citations: list[Citation] = []
    for number in _extract_cited_indices(answer, len(chunks)):
        chunk = chunks[number - 1]
        citations.append(
            Citation(
                index=number,
                doc_id=chunk.doc_id,
                filename=chunk.filename or chunk.doc_id,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                score=round(float(chunk.score), 4),
            )
        )
    return citations


def _source_to_citation(source: Source) -> Citation:
    """Convert a Source to a Citation for the API response."""
    return Citation(
        index=source.index,
        doc_id=source.doc_id,
        filename=source.filename or source.doc_id,
        chunk_index=source.chunk_index,
        content=source.content,
        score=round(float(source.score), 4) if source.score is not None else 0.0,
    )


def _format_chat_response(
    thread_id: str,
    state_values: dict[str, Any],
    next_nodes: tuple[str, ...],
) -> ChatResponse:
    """Map LangGraph state snapshot into the strict frontend API contract."""
    messages = state_values.get("messages", [])

    # 1. ALWAYS prioritize the latest AI message over stale state keys
    answer = None
    if messages and getattr(messages[-1], "type", None) == "ai":
        answer = str(messages[-1].content)
    # 2. Only fallback to state["answer"] if no AI message exists
    elif state_values.get("answer"):
        answer = str(state_values["answer"])

    retrieved_chunks = state_values.get("retrieved_chunks", [])
    top_score = float(
        state_values.get(
            "top_score",
            retrieved_chunks[0].score if retrieved_chunks else 0.0,
        )
    )

    is_paused = "human_review" in next_nodes
    interrupt_payload = None

    if is_paused:
        preview_texts = [str(c.content)[:200] for c in retrieved_chunks[:5]]
        interrupt_payload = InterruptPreview(
            question=str(
                state_values.get("rewritten_query")
                or state_values.get("original_query", "")
            ),
            top_score=top_score,
            chunks_preview=preview_texts,
        )
        answer = None

    # Build citations from state sources if available, otherwise from chunks
    sources = state_values.get("sources", [])
    if sources and not is_paused:
        citations = [_source_to_citation(s) for s in sources]
    else:
        citations = [] if is_paused else _build_citations(answer, retrieved_chunks)

    return ChatResponse(
        thread_id=thread_id,
        status="awaiting_approval" if is_paused else "completed",
        answer=answer,
        rewritten_query=state_values.get("rewritten_query")
        or state_values.get("original_query"),
        top_retrieval_score=top_score,
        citations=citations,
        interrupt_payload=interrupt_payload,
    )


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: Request,
    payload: ChatRequest,
    x_user_id: Annotated[str, Header()],
    lock_manager: Annotated[ThreadLockProtocol, Depends(get_lock_manager)],
) -> Any:
    """Execute a RAG turn via blocking JSON or Server-Sent Events (SSE)."""
    graph: Any = request.app.state.rag_graph
    config: dict[str, Any] = {"configurable": {"thread_id": payload.thread_id}}

    state = await graph.aget_state(config)
    if state.values:
        owner = state.values.get("user_id")
        if owner and owner != x_user_id:
            raise HTTPException(
                status_code=403,
                detail="Thread belongs to another tenant.",
            )
        if "human_review" in state.next:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Thread is paused awaiting human review. "
                    "Call POST /resume before sending new messages."
                ),
            )

    graph_input: dict[str, Any] = {
        "user_id": x_user_id,
        "original_query": payload.query,
        "messages": [HumanMessage(content=payload.query)],
    }

    async def _sse_generator() -> AsyncGenerator[str, None]:
        async with lock_manager.try_lock(payload.thread_id) as acquired:
            if not acquired:
                err = json.dumps({"detail": "Thread is locked by another request."})
                yield f"event: error\ndata: {err}\n\n"
                return
            try:
                async for event in graph.astream(
                    graph_input,
                    config,
                    stream_mode="updates",
                ):
                    node_name = next(iter(event.keys()), "step")
                    step_payload = json.dumps(
                        {"node": node_name, "update": event[node_name]},
                        default=str,
                    )
                    yield f"event: step\ndata: {step_payload}\n\n"

                final_state = await graph.aget_state(config)
                final_resp = _format_chat_response(
                    payload.thread_id,
                    final_state.values,
                    tuple(final_state.next),
                )
                yield f"event: complete\ndata: {final_resp.model_dump_json()}\n\n"
            except Exception as exc:
                err_payload = json.dumps({"detail": str(exc)})
                yield f"event: error\ndata: {err_payload}\n\n"

    if payload.stream:
        return StreamingResponse(
            _sse_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    async with lock_manager.try_lock(payload.thread_id) as acquired:
        if not acquired:
            raise HTTPException(
                status_code=409,
                detail="Thread is currently locked by another active request.",
            )
        await graph.ainvoke(graph_input, config)
        final_state = await graph.aget_state(config)
        return _format_chat_response(
            payload.thread_id,
            final_state.values,
            tuple(final_state.next),
        )


@router.post("/resume", response_model=ChatResponse)
async def resume_endpoint(
    request: Request,
    payload: ResumeRequest,
    x_user_id: Annotated[str, Header()],
    lock_manager: Annotated[ThreadLockProtocol, Depends(get_lock_manager)],
) -> Any:
    """Resume a paused HITL thread with human approval or rejection."""
    graph: Any = request.app.state.rag_graph
    config: dict[str, Any] = {"configurable": {"thread_id": payload.thread_id}}

    async with lock_manager.try_lock(payload.thread_id) as acquired:
        if not acquired:
            raise HTTPException(
                status_code=409,
                detail="Thread is currently locked by another active request.",
            )
        state = await graph.aget_state(config)
        if not state.values or "human_review" not in state.next:
            raise HTTPException(
                status_code=409,
                detail="Thread is not paused for human review.",
            )
        owner = state.values.get("user_id")
        if owner and owner != x_user_id:
            raise HTTPException(
                status_code=403,
                detail="Thread belongs to another tenant.",
            )

        await graph.ainvoke(
            Command(
                resume={
                    "approved": payload.approved,
                    "feedback": payload.feedback,
                }
            ),
            config,
        )
        final_state = await graph.aget_state(config)
        return _format_chat_response(
            payload.thread_id,
            final_state.values,
            tuple(final_state.next),
        )
