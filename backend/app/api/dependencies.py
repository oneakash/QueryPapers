from contextlib import AbstractAsyncContextManager

"""FastAPI dependency injection providers for auth, graph, and distributed locks."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any, Protocol, cast

from fastapi import Header, HTTPException, Request, status

from app.core.interfaces import VectorStoreProtocol


class ThreadLockProtocol(Protocol):
    """Protocol for acquiring non-blocking per-thread locks."""

    def try_lock(self, thread_id: str) -> AbstractAsyncContextManager[bool]:
        """Context manager yielding True if lock acquired, False if already locked."""
        ...


class InMemoryThreadLockManager:
    """Process-safe non-blocking per-thread lock manager for dev and testing."""

    def __init__(self) -> None:
        self._active_threads: set[str] = set()
        self._guard = asyncio.Lock()

    @asynccontextmanager
    async def try_lock(self, thread_id: str) -> AsyncIterator[bool]:
        """Acquire non-blocking lock on thread_id or yield False immediately."""
        async with self._guard:
            if thread_id in self._active_threads:
                yield False
                return
            self._active_threads.add(thread_id)

        try:
            yield True
        finally:
            async with self._guard:
                self._active_threads.discard(thread_id)


class PostgresAdvisoryLockManager:
    """Multi-worker distributed lock manager using Postgres advisory locks."""

    def __init__(self, pool: Any) -> None:
        self._pool = pool

    @asynccontextmanager
    async def try_lock(self, thread_id: str) -> AsyncIterator[bool]:
        """Acquire a transaction-level Postgres advisory lock on thread_id."""
        async with self._pool.connection() as conn:
            async with conn.transaction():
                cursor = await conn.execute(
                    "SELECT pg_try_advisory_xact_lock(hashtext(%s))",
                    (thread_id,),
                )
                row = await cursor.fetchone()
                acquired = bool(row and row[0])
                yield acquired


async def get_current_user_id(
    x_user_id: Annotated[str | None, Header(alias="X-User-ID")] = None,
) -> str:
    """Extract and validate tenant user_id from request headers."""
    if x_user_id is None or not x_user_id.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or empty X-User-ID header.",
        )
    return x_user_id.strip()


def get_vector_store(request: Request) -> VectorStoreProtocol:
    """Retrieve the configured VectorStoreProtocol instance from app state."""
    return cast(VectorStoreProtocol, request.app.state.vector_store)


def get_graph(request: Request) -> Any:
    """Retrieve the compiled LangGraph instance from app state."""
    return request.app.state.rag_graph


def get_lock_manager(request: Request) -> Any:
    """Retrieve the per-thread lock manager from app state."""
    return request.app.state.lock_manager
