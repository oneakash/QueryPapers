# LangGraph HITL RAG API

A production-grade, stateful Human-in-the-Loop (HITL) Retrieval-Augmented Generation (RAG) system built with **FastAPI**, **LangGraph**, **Qdrant**, and **PostgreSQL**.

This system is designed around SOLID principles, featuring strict tenant isolation, asynchronous document ingestion, deterministic fallback mechanisms, distributed locking, and seamless multi-provider LLM support (Google Gemini, OpenAI, Groq, Ollama).

## 🚀 Key Features

- **Human-in-the-Loop (HITL):** Automatically pauses graph execution (`awaiting_approval`) when retrieval confidence is below a configurable threshold, allowing human reviewers to approve, reject, or correct the response.
- **Short-Term & Long-Term Memory:** Utilizes PostgreSQL to checkpoint conversation history (resolving pronouns like "here" to "Paris") and a Long-Term Memory Store to persist reusable human reviewer rules across completely different threads.
- **Strict Tenant Isolation:** Enforces mandatory `X-User-ID` header filtering at the vector database level. Cross-tenant data leakage is structurally impossible.
- **Atomic Ingestion & Non-Blocking Parsing:** Offloads CPU-bound PDF parsing (`PyMuPDF`) to a threadpool and automatically rolls back Qdrant vector insertions if an upload fails mid-flight.
- **Multi-Provider LLM Support:** Switch between Google Gemini (Free Tier), OpenAI, Groq, or local Ollama instantly via `.env` variables without changing a single line of business logic.
- **Distributed Concurrency Control:** Uses PostgreSQL Transaction Advisory Locks to prevent race conditions during concurrent HITL `/resume` requests across multiple worker nodes.

## 🛠️ Technology Stack

- **Framework:** FastAPI, Uvicorn, Python 3.12+
- **Orchestration:** LangGraph (StateGraph, Checkpointer, Store)
- **Vector Database:** Qdrant (Async Client)
- **Database:** PostgreSQL (Checkpoints, LTM Store, Advisory Locks)
- **Document Parsing:** PyMuPDF, LangChain Text Splitters
- **Package Management:** `uv`
- **Linting & QA:** `black`, `isort`, `pycodestyle`, `mypy`, `pytest`

## ⚙️ Getting Started

### 1. Prerequisites
- [Docker & Docker Compose](https://docs.docker.com/get-docker/)
- [uv](https://github.com/astral-sh/uv) (Extremely fast Python package manager)
- Python 3.12+

### 2. Setup the Environment
Clone the repository and install dependencies using `uv`:

```bash
uv sync
```

Create your local environment variables file:

```bash
cp .env.example .env
```
*Edit `.env` to include your valid API keys (e.g., Google AI Studio Gemini key).*

### 3. Start the Infrastructure
Spin up the local PostgreSQL and Qdrant containers:

```bash
docker compose up -d
```

### 4. Run the API Server
Start the FastAPI application. The `lifespan` manager will automatically connect to Qdrant/Postgres and initialize the required collections and tables.

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 📖 API Usage Examples

Once the server is running, you can access the interactive Swagger documentation at `http://localhost:8000/docs`.

### 1. Upload a Document
Ingest a PDF or TXT file scoped to a specific user.

```bash
curl -X POST http://localhost:8000/upload \
  -H "X-User-ID: tenant_a" \
  -F "file=@/path/to/your/document.pdf;type=application/pdf"
```

### 2. Chat (Standard RAG)
Ask a question. The system will retrieve context from `tenant_a`'s documents.

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-User-ID: tenant_a" \
  -d '{
    "thread_id": "thread_01",
    "query": "Summarize the key points of the uploaded document."
  }'
```

### 3. Human-in-the-Loop Resume
If a query returns `"status": "awaiting_approval"`, a human reviewer can inject feedback to guide the LLM and save rules to Long-Term Memory.

```bash
curl -X POST http://localhost:8000/resume \
  -H "Content-Type: application/json" \
  -H "X-User-ID: tenant_a" \
  -d '{
    "thread_id": "thread_01",
    "approved": true,
    "feedback": "Always format the output as a Markdown table."
  }'
```

---

## 🧪 Testing & Code Quality

This project enforces strict code quality and typing standards. Run the following commands to verify the codebase before opening a PR:

```bash
# 1. Sort imports
uv run isort app/ tests/

# 2. Format code
uv run black app/ tests/

# 3. Check PEP 8 compliance
uv run pycodestyle app/ tests/

# 4. Strict Static Type Checking
uv run mypy app/ tests/

# 5. Run the test suite with coverage
uv run pytest --cov=app --cov-report=term-missing -v
```