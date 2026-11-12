.PHONY: infra-up backend-dev frontend-dev backend-check sync-types

# Start Postgres & Qdrant containers for local hybrid development
infra-up:
	docker compose up -d postgres qdrant

# Run FastAPI with hot-reload inside backend/
backend-dev:
	cd backend && uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Run strict PEP 8, Mypy, and Pytest checks inside backend/
backend-check:
	cd backend && uv run isort app/ tests/ && uv run black app/ tests/ && uv run pycodestyle app/ tests/ && uv run mypy app/ tests/ && uv run pytest -v

# Start Next.js dev server inside frontend/
frontend-dev:
	cd frontend && npm run dev

# Auto-generate TypeScript interfaces from FastAPI OpenAPI schema
sync-types:
	cd frontend && npx openapi-typescript http://localhost:8000/openapi.json -o src/types/api.generated.ts