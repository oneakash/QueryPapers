"""HTTP routes for stateful RAG chat and HITL interrupt resumption."""

import json
import logging
import time
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from app.api.dependencies import get_current_user_id, get_graph, get_lock_manager
from app.api.schemas import (
    ChatRequest,
    ChatResponse,
    InterruptPreview,
    ResumeRequest,
)

logger = logging.getLogger("hitl_rag.audit")
router = APIRouter(tags=["chat"])


def _build_chat_response(
    thread_id: str,
    snapshot: Any,
) -> ChatResponse:
    """Construct ChatResponse from a LangGraph StateSnapshot."""
    values: dict[str, Any] = snapshot.values or {}
    rewritten = values.get("rewritten_query")
    top_score = float(values.get("top_score", 0.0))

    if snapshot.next and "human_review" in snapshot.next:
        chunks = values.get("retrieved_chunks", [])
        preview = InterruptPreview(
            question=str(rewritten or values.get("original_query", "")),
            top_score=top_score,
            chunks_preview=[str(c.content)[:200] for c in chunks],
        )
        return ChatResponse(
            thread_id=thread_id,
            status="awaiting_approval",
            answer=None,
            rewritten_query=rewritten,
            top_retrieval_score=top_score,
            interrupt_payload=preview,
        )

    return ChatResponse(
        thread_id=thread_id,
        status="completed",
        answer=str(values.get("answer", "")),
        rewritten_query=rewritten,
        top_retrieval_score=top_score,
        interrupt_payload=None,
    )


@router.post(
    "/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
)
async def chat_endpoint(
    payload: ChatRequest,
    user_id: Annotated[str, Depends(get_current_user_id)],
    graph: Annotated[Any, Depends(get_graph)],
) -> ChatResponse:
    """Execute a stateful RAG turn or pause at human_review if low confidence."""
    start_time = time.perf_counter()
    config: RunnableConfig = {"configurable": {"thread_id": payload.thread_id}}

    existing_state = await graph.aget_state(config)
    if existing_state and existing_state.values:
        owner_id = existing_state.values.get("user_id")
        if owner_id and owner_id != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Thread belongs to a different user.",
            )
        if existing_state.next and "human_review" in existing_state.next:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    "Thread is paused awaiting human review. "
                    "Call POST /resume before sending new messages."
                ),
            )

    await graph.ainvoke(
        {
            "messages": [HumanMessage(content=payload.query)],
            "original_query": payload.query,
            "user_id": user_id,
            "hitl_approved": None,
            "hitl_feedback": None,
        },
        config=config,
    )

    final_snapshot = await graph.aget_state(config)
    response = _build_chat_response(payload.thread_id, final_snapshot)

    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
    logger.info(
        json.dumps(
            {
                "event": "chat_turn",
                "thread_id": payload.thread_id,
                "user_id": user_id,
                "rewritten_query": response.rewritten_query,
                "top_retrieval_score": response.top_retrieval_score,
                "status": response.status,
                "latency_ms": latency_ms,
            }
        )
    )
    return response


@router.post(
    "/resume",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
)
async def resume_endpoint(
    payload: ResumeRequest,
    user_id: Annotated[str, Depends(get_current_user_id)],
    graph: Annotated[Any, Depends(get_graph)],
    lock_manager: Annotated[Any, Depends(get_lock_manager)],
) -> ChatResponse:
    """Resume a paused HITL thread with atomic double-resume protection."""
    start_time = time.perf_counter()
    config: RunnableConfig = {"configurable": {"thread_id": payload.thread_id}}

    async with lock_manager.try_lock(payload.thread_id) as acquired:
        if not acquired:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Concurrent resume already in progress for this thread.",
            )

        snapshot = await graph.aget_state(config)
        if not snapshot or not snapshot.values:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Thread not found.",
            )

        owner_id = snapshot.values.get("user_id")
        if owner_id and owner_id != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Thread belongs to a different user.",
            )

        if not snapshot.next or "human_review" not in snapshot.next:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Thread is not currently awaiting human review.",
            )

        await graph.ainvoke(
            Command(
                resume={
                    "approved": payload.approved,
                    "feedback": payload.feedback,
                }
            ),
            config=config,
        )

        final_snapshot = await graph.aget_state(config)
        response = _build_chat_response(payload.thread_id, final_snapshot)

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        logger.info(
            json.dumps(
                {
                    "event": "hitl_audit",
                    "thread_id": payload.thread_id,
                    "reviewer_id": user_id,
                    "approved": payload.approved,
                    "feedback": payload.feedback,
                    "latency_ms": latency_ms,
                }
            )
        )
        return response
