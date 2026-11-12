"""Custom domain exceptions for API and ingestion error handling."""


class InvalidDocumentError(ValueError):
    """Raised when an uploaded file is empty, corrupted, encrypted, or unreadable."""


class VectorStoreOperationError(RuntimeError):
    """Raised when a vector database upsert, search, or delete operation fails."""
