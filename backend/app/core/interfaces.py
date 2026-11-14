"""Abstract protocols and domain data structures for Dependency Inversion."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol, runtime_checkable

from langchain_core.messages import BaseMessage


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    """Immutable retrieved document chunk with its similarity score."""

    chunk_id: str
    doc_id: str
    user_id: str
    content: str
    score: float
    filename: str = ""
    chunk_index: int = 0
    content_hash: str = ""


@dataclass(frozen=True, slots=True)
class StoredDocument:
    """Document-level metadata recovered from a document's first chunk."""

    doc_id: str
    filename: str
    content_hash: str
    chunk_count: int


@dataclass(frozen=True, slots=True)
class Source:
    """Generic source reference for citations, supporting document and web sources."""

    kind: Literal["document", "web"]
    index: int
    title: str
    url: str | None = None
    doc_id: str = ""
    filename: str = ""
    chunk_index: int = 0
    content: str = ""
    score: float | None = None


@runtime_checkable
class VectorStoreProtocol(Protocol):
    """Abstract contract for tenant-isolated vector storage operations."""

    async def upsert_chunks(
        self,
        user_id: str,
        doc_id: str,
        chunks: Sequence[str],
        filename: str = "",
        content_hash: str = "",
    ) -> int:
        """Embed and upsert document chunks for a specific user and document."""
        ...

    async def find_documents(
        self,
        user_id: str,
        *,
        content_hash: str | None = None,
        filename: str | None = None,
    ) -> list[StoredDocument]:
        """List this user's documents, optionally matching a hash and/or filename."""
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
