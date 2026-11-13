.PHONY: backend-dev back-build frontend-dev backend-check sync-types

# Start Postgres & Qdrant containers for local hybrid development and Run FastAPI
backend-dev:
	docker compose up -d backend postgres qdrant

back-build:
	docker compose up --build -d backend

# Run strict PEP 8, Mypy, and Pytest checks inside backend/
backend-check:
	cd backend && uv run isort app/ tests/ && uv run black app/ tests/ && uv run pycodestyle app/ tests/ && uv run mypy app/ tests/ && uv run pytest -v

# Start Next.js dev server inside frontend/
frontend-dev:
	cd frontend && npm run dev

# Auto-generate TypeScript interfaces from FastAPI OpenAPI schema
sync-types:
	cd frontend && npx openapi-typescript http://localhost:8000/openapi.json -o src/types/api.generated.ts