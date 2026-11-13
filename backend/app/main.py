"""FastAPI application entrypoint, CORS middleware, lifespan, and health probe."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.memory import InMemoryStore
from langgraph.store.postgres.aio import AsyncPostgresStore
from psycopg_pool import AsyncConnectionPool
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as rest

from app.api.dependencies import (
    InMemoryThreadLockManager,
    PostgresAdvisoryLockManager,
)
from app.api.routes.chat import router as chat_router
from app.api.routes.upload import router as upload_router
from app.api.schemas import HealthResponse
from app.core.config import Settings, get_settings
from app.core.interfaces import LLMProtocol, VectorStoreProtocol
from app.graph.builder import build_graph
from app.services.llm_client import create_embedding_client, create_llm_client
from app.services.qdrant_store import QdrantVectorStore


async def _default_health_check() -> tuple[bool, bool]:
    """Return default (postgres_ok, qdrant_ok) health status for tests."""
    return True, True


_PAYLOAD_INDEXES: dict[str, rest.PayloadSchemaType] = {
    "user_id": rest.PayloadSchemaType.KEYWORD,
    "doc_id": rest.PayloadSchemaType.KEYWORD,
    "content_hash": rest.PayloadSchemaType.KEYWORD,
    "filename": rest.PayloadSchemaType.KEYWORD,
    "chunk_index": rest.PayloadSchemaType.INTEGER,
}


async def _ensure_qdrant_collection(
    client: AsyncQdrantClient,
    collection_name: str,
    dimensions: int,
) -> None:
    """Create the Qdrant collection and its payload indexes if missing."""
    exists = await client.collection_exists(collection_name)
    if not exists:
        await client.create_collection(
            collection_name=collection_name,
            vectors_config=rest.VectorParams(
                size=dimensions,
                distance=rest.Distance.COSINE,
            ),
        )
    for field_name, field_schema in _PAYLOAD_INDEXES.items():
        await client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=field_schema,
        )


def create_app(
    settings: Settings | None = None,
    vector_store: VectorStoreProtocol | None = None,
    llm: LLMProtocol | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application instance."""
    cfg = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Initialize and cleanly tear down live Postgres, Qdrant, and LLM clients."""
        if vector_store is not None and llm is not None:
            yield
            return

        llm_client = create_llm_client(cfg)
        embed_client = create_embedding_client(cfg)
        qdrant_client = AsyncQdrantClient(
            url=cfg.qdrant_url,
            timeout=int(cfg.request_timeout_seconds),
        )
        await _ensure_qdrant_collection(
            client=qdrant_client,
            collection_name=cfg.qdrant_collection,
            dimensions=cfg.embedding_dimensions,
        )
        live_vector_store = QdrantVectorStore(
            client=qdrant_client,
            embedder=embed_client,
            collection_name=cfg.qdrant_collection,
        )

        pool = AsyncConnectionPool(
            conninfo=cfg.postgres_dsn,
            min_size=1,
            max_size=10,
            open=False,
        )
        await pool.open()

        async with (
            AsyncPostgresSaver.from_conn_string(cfg.postgres_dsn) as checkpointer,
            AsyncPostgresStore.from_conn_string(cfg.postgres_dsn) as ltm_store,
        ):
            await checkpointer.setup()
            await ltm_store.setup()

            app.state.vector_store = live_vector_store
            app.state.lock_manager = PostgresAdvisoryLockManager(pool)
            app.state.rag_graph = build_graph(
                vector_store=live_vector_store,
                llm=llm_client,
                checkpointer=checkpointer,
                store=ltm_store,
                confidence_threshold=cfg.hitl_confidence_threshold,
                retrieval_top_k=cfg.retrieval_top_k,
                max_user_rules=cfg.max_user_rules,
            )

            async def _live_health_probe() -> tuple[bool, bool]:
                pg_ok = False
                qdrant_ok = False
                try:
                    async with pool.connection() as conn:
                        await conn.execute("SELECT 1")
                    pg_ok = True
                except Exception:
                    pg_ok = False

                try:
                    await qdrant_client.get_collections()
                    qdrant_ok = True
                except Exception:
                    qdrant_ok = False

                return pg_ok, qdrant_ok

            app.state.health_probe = _live_health_probe
            try:
                yield
            finally:
                await pool.close()
                await qdrant_client.close()
                if hasattr(llm_client, "close"):
                    await llm_client.close()
                if hasattr(embed_client, "close"):
                    await embed_client.close()

    app = FastAPI(title=cfg.app_name, lifespan=lifespan)

    # Allow Next.js frontend to call FastAPI and send X-User-ID headers
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ],
        allow_credentials=True,
        allow_origin_regex=(
            r"^https?://(localhost|127\.0\.0\.1" r"|192\.168\.\d+\.\d+)(:\d+)?$"
        ),
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "X-User-ID", "Accept"],
    )

    app.state.lock_manager = InMemoryThreadLockManager()
    app.state.health_probe = _default_health_check

    if vector_store is not None and llm is not None:
        app.state.vector_store = vector_store
        app.state.rag_graph = build_graph(
            vector_store=vector_store,
            llm=llm,
            checkpointer=MemorySaver(),
            store=InMemoryStore(),
            confidence_threshold=cfg.hitl_confidence_threshold,
            retrieval_top_k=cfg.retrieval_top_k,
            max_user_rules=cfg.max_user_rules,
        )

    app.include_router(upload_router)
    app.include_router(chat_router)

    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    async def health_endpoint(request: Request) -> Any:
        """Verify live connectivity to PostgreSQL and Qdrant."""
        probe = request.app.state.health_probe
        pg_ok, qdrant_ok = await probe()
        overall: Literal["ok", "degraded"] = (
            "ok" if (pg_ok and qdrant_ok) else "degraded"
        )
        payload = HealthResponse(
            status=overall,
            postgres=pg_ok,
            qdrant=qdrant_ok,
        )
        http_status = (
            status.HTTP_200_OK
            if overall == "ok"
            else status.HTTP_503_SERVICE_UNAVAILABLE
        )
        return JSONResponse(status_code=http_status, content=payload.model_dump())

    return app


app = create_app()