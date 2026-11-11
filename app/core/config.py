"""Application configuration and environment validation."""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Fail-fast application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    app_name: str = Field(default="LangGraph HITL RAG API")
    environment: str = Field(default="development")

    # Required Database & Vector Store URLs
    postgres_dsn: str = Field(
        default="postgresql://rag_user:rag_password@localhost:5432/rag_db",
        description="PostgreSQL connection string for Checkpointer and Store.",
    )
    qdrant_url: str = Field(
        default="http://localhost:6333",
        description="Qdrant HTTP endpoint URL.",
    )
    qdrant_collection: str = Field(
        default="tenant_documents",
        description="Base Qdrant collection name for document chunks.",
    )

    # Dynamic Model & Provider Selection
    llm_provider: Literal["gemini", "openai_compatible"] = Field(
        default="gemini",
        description="Active LLM provider ('gemini' or 'openai_compatible').",
    )
    llm_model: str = Field(
        default="gemini-3.6-flash",
        description="Model name (e.g., gemini-3.6-flash, gpt-4o-mini, llama-3.3-70b).",
    )
    llm_api_key: str = Field(
        default="test-api-key",
        description="API key for the active LLM provider.",
    )
    llm_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="Base URL when using openai_compatible (Groq, OpenAI, Ollama).",
    )

    # Embedding Provider Selection
    embedding_provider: Literal["gemini", "openai_compatible"] = Field(
        default="gemini",
        description="Active Embedding provider ('gemini' or 'openai_compatible').",
    )
    embedding_model: str = Field(
        default="gemini-embedding-2",
        description="Embedding model name (e.g., gemini-embedding-2).",
    )
    embedding_api_key: str = Field(
        default="test-api-key",
        description="API key for the embedding provider.",
    )
    embedding_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="Base URL when using openai_compatible embeddings.",
    )
    embedding_dimensions: int = Field(
        default=768,
        gt=0,
        description="Vector output dimensions (768, 1536, or 3072).",
    )

    # RAG & HITL Thresholds
    retrieval_top_k: int = Field(default=5, ge=1, le=20)
    hitl_confidence_threshold: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Minimum top similarity score before triggering HITL interrupt.",
    )
    max_user_rules: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum Long-Term Memory rules stored per tenant.",
    )
    request_timeout_seconds: float = Field(default=20.0, gt=0.0)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton instance of application settings."""
    return Settings()
