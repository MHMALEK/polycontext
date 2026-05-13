# ticket-recon: common workflow shortcuts.
#
# One-command quick start:
#   make up        — bring up the full stack (postgres, redis, sourcebot, app)
#                    via Docker. Open http://localhost:8000/ui afterwards.
#
# Local dev with hot reload (recommended while editing code):
#   make dev       — runs backends in Docker, FastAPI + Vite on the host.
#                    Edit Python → FastAPI reloads. Edit web/ → browser reloads.
#                    Open http://localhost:5173 (Vite dev) or :8000/ui.
#
# Other:
#   make build     — rebuild the app image (includes UI build)
#   make logs      — tail app logs
#   make down      — stop everything
#   make clean     — stop and remove named volumes (CAUTION: wipes index)

.PHONY: up dev build logs down clean smoke install ui-install help
.DEFAULT_GOAL := help

# ---------------------------------------------------------------------------
# One-shot
# ---------------------------------------------------------------------------

up:  ## Full stack via Docker. UI served at http://localhost:8000/ui
	docker compose up -d --build
	@echo ""
	@echo "  API   : http://localhost:8000"
	@echo "  UI    : http://localhost:8000/ui"
	@echo "  Sbot  : http://localhost:3000"
	@echo ""
	@echo "Tail logs:    make logs"
	@echo "Stop stack:   make down"

build:  ## Rebuild the app image (incl. UI bundle)
	docker compose build app

logs:  ## Tail app logs
	docker compose logs -f app

down:  ## Stop everything
	docker compose down

clean:  ## Stop and delete named volumes (wipes Sourcebot index)
	docker compose down -v

# ---------------------------------------------------------------------------
# Dev with hot reload (host-level Python + Vite, dockerized backends)
# ---------------------------------------------------------------------------

dev: install ui-install  ## Dev mode: backends in docker, FastAPI + Vite on host
	@echo "Starting Postgres, Redis, Sourcebot in Docker..."
	docker compose up -d postgres redis sourcebot
	@echo ""
	@echo "FastAPI on :8000 + Vite on :5173. Ctrl-C to stop both."
	@echo "Open http://localhost:5173 (the Vite proxy talks to FastAPI for you)."
	@trap 'kill 0' EXIT INT TERM; \
	uv run uvicorn ticket_recon.api:app --reload --port 8000 & \
	(cd web && npm run dev) & \
	wait

install:  ## Install Python deps via uv
	uv sync

ui-install:  ## Install UI deps via npm
	@cd web && npm install --no-audit --no-fund

# ---------------------------------------------------------------------------
# Smoke
# ---------------------------------------------------------------------------

smoke:  ## Ask one canned question via the running API
	curl -sS http://localhost:8000/ask -H 'content-type: application/json' \
	  -d '{"question":"Where is the Suppliers page rendered?"}' | \
	  python -m json.tool

help:  ## Show this help.
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) | awk -F':.*##' '{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'
