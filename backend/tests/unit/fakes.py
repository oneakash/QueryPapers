"""In-memory fake implementations of core protocols for fast unit testing."""

from collections.abc import Sequence
from typing import Any

from langchain_core.messages import BaseMessage

from app.core.interfaces import ScoredChunk, StoredDocument


class FakeVectorStore:
    """In-memory tenant-isolated vector store for deterministic testing."""

    def __init__(self, default_score: float = 0.85) -> None:
        self._storage: list[ScoredChunk] = []
        self.default_score = default_score
        self.fail_on_upsert: bool = False

    async def upsert_chunks(
        self,
        user_id: str,
        doc_id: str,
        chunks: Sequence[str],
        filename: str = "",
        content_hash: str = "",
    ) -> int:
        """Store chunks in memory or simulate mid-upload failure."""
        if self.fail_on_upsert:
            # Simulate partial insert before raising error to test rollback
            if chunks:
                self._storage.append(
                    ScoredChunk(
                        chunk_id=f"{doc_id}_0",
                        doc_id=doc_id,
                        user_id=user_id,
                        content=chunks[0],
                        score=self.default_score,
                        filename=filename,
                        chunk_index=0,
                        content_hash=content_hash,
                    )
                )
            raise RuntimeError("Simulated Qdrant upsert failure.")

        for idx, text in enumerate(chunks):
            self._storage.append(
                ScoredChunk(
                    chunk_id=f"{doc_id}_{idx}",
                    doc_id=doc_id,
                    user_id=user_id,
                    content=text,
                    score=self.default_score,
                    filename=filename,
                    chunk_index=idx,
                    content_hash=content_hash,
                )
            )
        return len(chunks)

    async def search(
        self,
        query: str,
        user_id: str,
        limit: int = 5,
    ) -> list[ScoredChunk]:
        """Return chunks strictly belonging to user_id matching query terms."""
        query_words = {
            word.strip("?.,!").lower()
            for word in query.split()
            if len(word.strip("?.,!")) > 2
        }
        matches: list[ScoredChunk] = []
        for chunk in self._storage:
            if chunk.user_id != user_id:
                continue
            content_lower = chunk.content.lower()
            if any(word in content_lower for word in query_words):
                matches.append(chunk)
        return matches[:limit]

    async def find_documents(
        self,
        user_id: str,
        *,
        content_hash: str | None = None,
        filename: str | None = None,
    ) -> list[StoredDocument]:
        """Return documents matching the optional hash and filename."""
        results: list[StoredDocument] = []
        seen_doc_ids: set[str] = set()
        for chunk in self._storage:
            if chunk.user_id != user_id:
                continue
            if content_hash is not None and chunk.content_hash != content_hash:
                continue
            if filename is not None and chunk.filename != filename:
                continue
            if chunk.doc_id not in seen_doc_ids:
                seen_doc_ids.add(chunk.doc_id)
                # Count chunks for this doc_id
                doc_chunks = [
                    c
                    for c in self._storage
                    if c.user_id == user_id and c.doc_id == chunk.doc_id
                ]
                results.append(
                    StoredDocument(
                        doc_id=chunk.doc_id,
                        filename=chunk.filename,
                        content_hash=chunk.content_hash or "",
                        chunk_count=len(doc_chunks),
                    )
                )
        return results

    async def delete_document(
        self,
        user_id: str,
        doc_id: str,
    ) -> None:
        """Remove all chunks matching both user_id and doc_id."""
        self._storage = [
            chunk
            for chunk in self._storage
            if not (chunk.user_id == user_id and chunk.doc_id == doc_id)
        ]


class OutOfOrderFakeStore:
    """Fake store that returns items in reverse order (newest first) like Postgres."""

    def __init__(self) -> None:
        self._storage: dict[str, list[dict[str, Any]]] = {}

    async def asearch(self, namespace: tuple[str, ...], limit: int = 10) -> list[Any]:
        """Return items newest first (simulating Postgres behavior)."""
        key = ":".join(namespace)
        items = self._storage.get(key, [])
        # Return newest first
        return items[::-1][:limit]

    async def aput(
        self, namespace: tuple[str, ...], key: str, value: dict[str, Any]
    ) -> None:
        """Store value with created_at timestamp."""
        key_str = ":".join(namespace)
        if key_str not in self._storage:
            self._storage[key_str] = []
        self._storage[key_str].append({"key": key, "value": value})

    async def adelete(self, namespace: tuple[str, ...], key: str) -> None:
        """Delete item by key."""
        key_str = ":".join(namespace)
        if key_str in self._storage:
            self._storage[key_str] = [
                item for item in self._storage[key_str] if item["key"] != key
            ]


class FakeLLM:
    """Deterministic fake LLM for testing query rewriting and answer generation."""

    def __init__(self) -> None:
        self.rewrite_calls: int = 0
        self.generate_calls: int = 0
        self.last_prompt_chunks: list[str] = []
        self.last_user_rules: list[str] = []

    async def rewrite_query(
        self,
        history: Sequence[BaseMessage],
        query: str,
    ) -> str:
        """Resolve 'here' to 'Paris' if 'Paris' appears in conversation history."""
        self.rewrite_calls += 1
        history_text = " ".join(str(msg.content) for msg in history)
        if "here" in query.lower() and "paris" in history_text.lower():
            return query.lower().replace("here", "in Paris")
        return query

    async def generate_answer(
        self,
        query: str,
        chunks: Sequence[ScoredChunk],
        user_rules: Sequence[str] = (),
    ) -> str:
        """Format a deterministic answer from chunks and record call metadata."""
        self.generate_calls += 1
        self.last_prompt_chunks = [chunk.content for chunk in chunks]
        self.last_user_rules = list(user_rules)

        combined = " | ".join(chunk.content for chunk in chunks)
        rules_prefix = f"[Rules: {'; '.join(user_rules)}] " if user_rules else ""
        return f"{rules_prefix}Based on your documents: {combined}"
