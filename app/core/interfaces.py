"""Abstract protocols and domain data structures for Dependency Inversion."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from langchain_core.messages import BaseMessage


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    """Immutable retrieved document chunk with its similarity score."""

    chunk_id: str
    doc_id: str
    user_id: str
    content: str
    score: float


@runtime_checkable
class VectorStoreProtocol(Protocol):
    """Abstract contract for tenant-isolated vector storage operations."""

    async def upsert_chunks(
        self,
        user_id: str,
        doc_id: str,
        chunks: Sequence[str],
    ) -> int:
        """Embed and upsert document chunks for a specific user and document."""
        ...

    async def search(
        self,
        query: str,
        user_id: str,
        limit: int = 5,
    ) -> list[ScoredChunk]:
        """Search vector store strictly filtered by user_id."""
        ...

    async def delete_document(
        self,
        user_id: str,
        doc_id: str,
    ) -> None:
        """Delete all vector points matching both user_id and doc_id."""
        ...


@runtime_checkable
class LLMProtocol(Protocol):
    """Abstract contract for LLM query contextualization and RAG generation."""

    async def rewrite_query(
        self,
        history: Sequence[BaseMessage],
        query: str,
    ) -> str:
        """Rewrite a follow-up query into a standalone search query."""
        ...

    async def generate_answer(
        self,
        query: str,
        chunks: Sequence[ScoredChunk],
        user_rules: Sequence[str] = (),
    ) -> str:
        """Generate a grounded answer from retrieved chunks and user LTM rules."""
        ...
