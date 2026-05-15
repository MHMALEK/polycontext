# tech-decomposition: common workflow shortcuts.
#
# Three modes:
#
#   make dev      — Local dev with hot reload (RECOMMENDED for editing code).
#                   Backends (postgres, redis, sourcebot) run in Docker.
#                   FastAPI + Vite run on the host with --reload + HMR.
#                   Uses .env. Open http://localhost:5173.
#
#   make up       — Full local stack in Docker. Image is built (incl. UI
#                   bundle baked in). Same image artifact you'd ship to prod.
#                   Uses .env. Open http://localhost:8000/ui.
#
#   make prod     — Production-flavored run. Uses .env.prod (copy from
#                   .env.prod.example). Applies compose.prod.yaml overlay:
#                   Sourcebot NOT exposed to host, quieter logs, longer
#                   timeouts. Same image as `make up`. Use to verify the
#                   prod config locally before deploying.
#
# Other:
#   make build    — rebuild the app image (UI included)
#   make logs     — tail app logs
#   make down     — stop the local-mode stack
#   make down-prod  stop the prod-mode stack
#   make clean    — stop and remove named volumes (CAUTION: wipes index)

.PHONY: up dev prod build logs down down-prod clean smoke install ui-install help \
        eval-adapters eval-cases eval-run eval-report \
        adapters adapters-down adapters-logs
.DEFAULT_GOAL := help

# Compose file combinations
#   LOCAL    = base + local overlay (publishes Sourcebot's :3000 for the UI)
#   PROD     = base + prod overlay  (no host port for Sourcebot; quieter logs)
#   ADAPTERS = local + adapters overlay (Tabby + OpenHands sidecars)
COMPOSE_LOCAL    := -f docker-compose.yml -f compose.local.yaml
COMPOSE_PROD     := -f docker-compose.yml -f compose.prod.yaml --env-file .env.prod
COMPOSE_ADAPTERS := -f docker-compose.yml -f compose.local.yaml -f compose.adapters.yaml

# ---------------------------------------------------------------------------
# Local — full Docker stack with .env
# ---------------------------------------------------------------------------

up:  ## Local: full stack in Docker. Open http://localhost:8000/ui
	docker compose $(COMPOSE_LOCAL) up -d --build
	@echo ""
	@echo "  API    : http://localhost:8000"
	@echo "  UI     : http://localhost:8000/ui"
	@echo "  Sbot   : http://localhost:3000"
	@echo ""
	@echo "Tail logs:    make logs"
	@echo "Stop:         make down"

# ---------------------------------------------------------------------------
# Prod-flavored — full Docker stack with .env.prod + compose.prod.yaml
# ---------------------------------------------------------------------------

prod:  ## Production-flavored: uses .env.prod and compose.prod.yaml overlay
	@test -f .env.prod || (echo "missing .env.prod — copy .env.prod.example and fill it in"; exit 1)
	docker compose $(COMPOSE_PROD) up -d --build
	@echo ""
	@echo "  API    : http://localhost:8000"
	@echo "  UI     : http://localhost:8000/ui"
	@echo "  Sbot   : (not exposed in prod mode — reachable only inside docker network)"
	@echo ""
	@echo "Tail logs:    docker compose $(COMPOSE_PROD) logs -f app"
	@echo "Stop:         make down-prod"

# ---------------------------------------------------------------------------
# Dev with hot reload (host-level Python + Vite, dockerized backends)
# ---------------------------------------------------------------------------

dev: install ui-install  ## Local dev: backends in Docker, FastAPI + Vite on host with hot reload
	@echo "Starting Postgres, Redis, Sourcebot in Docker..."
	docker compose $(COMPOSE_LOCAL) up -d postgres redis sourcebot
	@echo ""
	@echo "FastAPI on :8000 + Vite on :5173. Ctrl-C to stop both."
	@echo "Open http://localhost:5173"
	@trap 'kill 0' EXIT INT TERM; \
	uv run uvicorn tech_decomposition.api:app --reload --port 8000 & \
	(cd web && npm run dev) & \
	wait

# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

build:  ## Build the app image (UI included)
	docker compose $(COMPOSE_LOCAL) build app

logs:  ## Tail app logs (local mode)
	docker compose $(COMPOSE_LOCAL) logs -f app

down:  ## Stop local-mode stack
	docker compose $(COMPOSE_LOCAL) down

down-prod:  ## Stop prod-mode stack
	docker compose $(COMPOSE_PROD) down

clean:  ## Stop and delete named volumes (wipes Sourcebot index)
	docker compose $(COMPOSE_LOCAL) down -v

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

install:  ## Install Python deps via uv
	uv sync

ui-install:  ## Install UI deps via npm
	@cd web && npm install --no-audit --no-fund

# ---------------------------------------------------------------------------
# Smoke
# ---------------------------------------------------------------------------

smoke:  ## Ask one canned question via the running API (override ADAPTER=...)
	curl -sS http://localhost:8000/v1/adapters/$${ADAPTER:-opencode}/ask \
	  -H 'content-type: application/json' \
	  -d '{"query":"Where is the Suppliers page rendered?"}' | \
	  python -m json.tool

# ---------------------------------------------------------------------------
# Adapter sidecars (currently just the cline-sdk-bridge)
# ---------------------------------------------------------------------------

adapters:  ## Start the cline-sdk-bridge sidecar
	docker compose $(COMPOSE_ADAPTERS) up -d --build cline_sdk_bridge
	@echo ""
	@echo "  cline-sdk-bridge  : http://localhost:$${CLINE_SDK_BRIDGE_PORT:-3040}"
	@echo ""
	@echo "Set in .env:"
	@echo "  CLINE_SDK_BRIDGE_URL=http://localhost:3040"
	@echo "Then 'make eval-adapters' to confirm health."

adapters-down:  ## Stop the cline-sdk-bridge sidecar
	docker compose $(COMPOSE_ADAPTERS) stop cline_sdk_bridge
	docker compose $(COMPOSE_ADAPTERS) rm -f cline_sdk_bridge

adapters-logs:  ## Tail cline-sdk-bridge logs
	docker compose $(COMPOSE_ADAPTERS) logs -f cline_sdk_bridge

# ---------------------------------------------------------------------------
# Adapter bake-off (eval/)
# ---------------------------------------------------------------------------

eval-adapters:  ## eval: list registered adapters and their health
	uv run python -m eval.bakeoff.cli list-adapters

eval-cases:  ## eval: list discovered cases (filter with JOB=ask|decompose|implement)
	uv run python -m eval.bakeoff.cli list-cases $(if $(JOB),--job $(JOB))

# Example: make eval-run ADAPTERS=cline_sdk,opencode JOB=ask
eval-run:  ## eval: run the bake-off (ADAPTERS=a,b JOB=ask|decompose|implement)
	@test -n "$(ADAPTERS)" || (echo "set ADAPTERS=a,b,c"; exit 1)
	uv run python -m eval.bakeoff.cli run --adapters $(ADAPTERS) $(if $(JOB),--job $(JOB)) $(if $(IDS),--ids $(IDS)) $(if $(TAGS),--tags $(TAGS))

eval-report:  ## eval: regenerate report.md/summary.json for RUN=eval-<ts>
	@test -n "$(RUN)" || (echo "set RUN=eval-<timestamp>"; exit 1)
	uv run python -m eval.bakeoff.cli report $(RUN)

help:  ## Show this help.
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) | awk -F':.*##' '{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'
