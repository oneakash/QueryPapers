"""Qdrant vector store implementation with mandatory tenant isolation."""

import uuid
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as rest

from app.core.interfaces import ScoredChunk


@runtime_checkable
class EmbeddingClientProtocol(Protocol):
    """Protocol for asynchronous text embedding generation."""

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of document chunks."""
        ...

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single search query."""
        ...


class QdrantVectorStore:
    """Concrete Qdrant store enforcing strict per-user metadata filtering."""

    def __init__(
        self,
        client: AsyncQdrantClient,
        embedder: EmbeddingClientProtocol,
        collection_name: str,
    ) -> None:
        self._client = client
        self._embedder = embedder
        self._collection = collection_name

    @staticmethod
    def build_tenant_filter(
        user_id: str,
        doc_id: str | None = None,
    ) -> rest.Filter:
        """Construct a mandatory Qdrant filter scoped to user_id and doc_id."""
        if not user_id.strip():
            raise ValueError("user_id is required for all Qdrant operations.")

        conditions: list[rest.Condition] = [
            rest.FieldCondition(
                key="user_id",
                match=rest.MatchValue(value=user_id),
            )
        ]
        if doc_id is not None:
            conditions.append(
                rest.FieldCondition(
                    key="doc_id",
                    match=rest.MatchValue(value=doc_id),
                )
            )
        return rest.Filter(must=conditions)

    async def upsert_chunks(
        self,
        user_id: str,
        doc_id: str,
        chunks: Sequence[str],
        filename: str = "",
    ) -> int:
        """Embed and upsert chunks with tenant and document metadata."""
        if not chunks:
            return 0

        text_list = list(chunks)
        vectors = await self._embedder.embed_documents(text_list)

        points: list[rest.PointStruct] = []
        for idx, (chunk_text, vector) in enumerate(zip(text_list, vectors)):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{user_id}:{doc_id}:{idx}"))
            points.append(
                rest.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "user_id": user_id,
                        "doc_id": doc_id,
                        "filename": filename,
                        "chunk_index": idx,
                        "content": chunk_text,
                    },
                )
            )

        await self._client.upsert(
            collection_name=self._collection,
            points=points,
            wait=True,
        )
        return len(points)

    async def search(
        self,
        query: str,
        user_id: str,
        limit: int = 5,
    ) -> list[ScoredChunk]:
        """Query Qdrant with mandatory user_id filter and return ScoredChunks."""
        query_vector = await self._embedder.embed_query(query)
        tenant_filter = self.build_tenant_filter(user_id=user_id)

        response = await self._client.query_points(
            collection_name=self._collection,
            query=query_vector,
            query_filter=tenant_filter,
            limit=limit,
            with_payload=True,
        )

        results: list[ScoredChunk] = []
        for point in response.points:
            payload = point.payload or {}
            results.append(
                ScoredChunk(
                    chunk_id=str(point.id),
                    doc_id=str(payload.get("doc_id", "")),
                    user_id=str(payload.get("user_id", user_id)),
                    content=str(payload.get("content", "")),
                    score=float(point.score),
                    filename=str(payload.get("filename", "")),
                    chunk_index=int(payload.get("chunk_index", 0)),
                )
            )
        return results

    async def delete_document(
        self,
        user_id: str,
        doc_id: str,
    ) -> None:
        """Delete all points belonging to both user_id and doc_id."""
        doc_filter = self.build_tenant_filter(user_id=user_id, doc_id=doc_id)
        await self._client.delete(
            collection_name=self._collection,
            points_selector=rest.FilterSelector(filter=doc_filter),
            wait=True,
        )
