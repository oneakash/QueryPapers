"""Structured JSON logging with request-id context and per-node timing decorator."""

import asyncio
import hashlib
import json
import logging
import sys
import time
import uuid
from contextvars import ContextVar
from functools import wraps
from typing import Any, Callable, TypeVar

# Context variable for request ID
_request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)

def get_request_id() -> str | None:
    """Get the current request ID from context."""
    return _request_id_var.get()

def set_request_id(request_id: str | None = None) -> str:
    """Set a request ID in context. Generates one if not provided."""
    rid = request_id or str(uuid.uuid4())[:8]
    _request_id_var.set(rid)
    return rid

class StructuredFormatter(logging.Formatter):
    """JSON formatter that includes request_id and standard fields."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON with request_id and standard fields."""
        # Get request_id from context
        request_id = get_request_id() or "-"

        # Build the log entry
        log_entry: dict[str, Any] = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "request_id": request_id,
            "message": record.getMessage(),
        }

        # Add exception info if present
        if record.exc_info:
            exc_type: str | None = (
                record.exc_info[0].__name__ if record.exc_info[0] else None
            )
            exc_msg: str | None = (
                str(record.exc_info[1]) if record.exc_info[1] else None
            )
            log_entry["exception"] = {"type": exc_type, "message": exc_msg}

        # Add extra fields from record
        extra_fields: dict[str, Any] = {}
        for key, value in record.__dict__.items():
            if key not in (
                "name",
                "msg",
                "args",
                "created",
                "filename",
                "funcName",
                "levelname",
                "levelno",
                "lineno",
                "module",
                "msecs",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "stack_info",
                "exc_info",
                "exc_text",
                "thread",
                "threadName",
                "message",
            ):
                # Never log sensitive fields
                if key not in ("user_id", "api_key", "query", "answer", "content"):
                    extra_fields[key] = value

        if extra_fields:
            log_entry["extra"] = extra_fields

        return json.dumps(log_entry, default=str)

def setup_logging(log_level: str = "INFO") -> None:
    """Configure structured JSON logging with the specified log level."""
    level = getattr(logging, log_level.upper(), logging.INFO)

    # Create a logger that outputs JSON
    logger = logging.getLogger()
    logger.setLevel(level)

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Add console handler with JSON formatter
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(StructuredFormatter())
    logger.addHandler(handler)

    # Set levels for noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

# Type variable for generic function signatures
F = TypeVar("F", bound=Callable[..., Any])

def hash_user_id(user_id: str | None) -> str:
    """Return a consistent hash of user_id for logging (never the raw ID)."""
    if not user_id:
        return "anonymous"
    return hashlib.sha256(user_id.encode()).hexdigest()[:16]

def log_node_timing(
    node_name: str,
    extract_fields: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> Callable[[F], F]:
    """Decorator that times a graph node and logs duration with safe fields.

    Args:
        node_name: Name of the node for logging.
        extract_fields: Optional function to extract node-specific fields from state.
                       Must NOT include document text, queries, answers, or API keys.
    """

    def decorator(func: F) -> F:
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.perf_counter()
            result = await func(*args, **kwargs)
            duration_ms = (time.perf_counter() - start_time) * 1000

            # Extract state from first positional arg (standard LangGraph pattern)
            state = args[0] if args else kwargs.get("state", {})
            if isinstance(state, dict):
                thread_id = (
                    state.get("config", {})
                    .get("configurable", {})
                    .get("thread_id", "unknown")
                )
                user_id = state.get("user_id")
            else:
                thread_id = "unknown"
                user_id = None

            # Build log fields
            log_fields = {
                "node": node_name,
                "duration_ms": round(duration_ms, 2),
                "thread_id": thread_id,
                "user_id_hash": hash_user_id(user_id),
            }

            # Add node-specific fields if extractor provided
            if extract_fields and isinstance(state, dict):
                log_fields.update(extract_fields(state))

            logger = logging.getLogger(__name__)
            logger.debug("Node timing", extra=log_fields)

            return result

        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            duration_ms = (time.perf_counter() - start_time) * 1000

            state = args[0] if args else kwargs.get("state", {})
            if isinstance(state, dict):
                thread_id = (
                    state.get("config", {})
                    .get("configurable", {})
                    .get("thread_id", "unknown")
                )
                user_id = state.get("user_id")
            else:
                thread_id = "unknown"
                user_id = None

            log_fields = {
                "node": node_name,
                "duration_ms": round(duration_ms, 2),
                "thread_id": thread_id,
                "user_id_hash": hash_user_id(user_id),
            }

            if extract_fields and isinstance(state, dict):
                log_fields.update(extract_fields(state))

            logger = logging.getLogger(__name__)
            logger.debug("Node timing", extra=log_fields)

            return result

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore[return-value]
        return sync_wrapper  # type: ignore[return-value]

    return decorator

# Convenience: pre-built extractors for common node fields
def extract_retrieve_fields(state: dict[str, Any]) -> dict[str, Any]:
    """Extract safe fields from retrieve node state."""
    fields: dict[str, Any] = {}
    if "top_score" in state:
        fields["top_score"] = round(state["top_score"], 4)
    if "retrieved_chunks" in state:
        fields["num_chunks"] = len(state["retrieved_chunks"])
    return fields

def extract_generate_fields(state: dict[str, Any]) -> dict[str, Any]:
    """Extract safe fields from generate node state."""
    fields: dict[str, Any] = {}
    if "top_score" in state:
        fields["top_score"] = round(state["top_score"], 4)
    if "retrieved_chunks" in state:
        fields["num_chunks"] = len(state["retrieved_chunks"])
    if "route" in state:
        fields["route"] = state["route"]
    if "attempts" in state:
        fields["attempts"] = state["attempts"]
    return fields

def extract_hitl_fields(state: dict[str, Any]) -> dict[str, Any]:
    """Extract safe fields from hitl_review node state."""
    fields: dict[str, Any] = {}
    if "top_score" in state:
        fields["top_score"] = round(state["top_score"], 4)
    if "hitl_approved" in state:
        fields["hitl_approved"] = state["hitl_approved"]
    return fields