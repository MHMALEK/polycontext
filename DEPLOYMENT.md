# Deployment

Single-page guide for getting tech-decomposition running locally and shipping
it to a remote host. Built around `docker compose` — the whole stack (Postgres,
Redis, Sourcebot, the app + UI) lives in one file with one optional overlay
for prod tuning.

## Three modes

| Mode | Command | Env file | Compose files | When |
|---|---|---|---|---|
| **Dev** (hot reload) | `make dev` | `.env` | base only | While editing Python or web/. FastAPI + Vite on the host; only backends in Docker. |
| **Local** (prod-like) | `make up` | `.env` | base only | Verify the bundled image works end-to-end. Everything in Docker. UI baked in. |
| **Prod** (prod config) | `make prod` | `.env.prod` | base + `compose.prod.yaml` | Production-flavored run. Sourcebot **not** exposed to host. Quieter logs. Same image as `make up`. |

`make` with no args prints all targets. `make build` rebuilds the app image; `make down` / `make down-prod` stop the respective stack; `make clean` wipes volumes (CAUTION: re-index needed).

## First-time setup

For **dev** or **local** modes:
```bash
cp .env.example .env
# fill in at minimum:
#   GEMINI_API_KEY
#   SOURCEBOT_AUTH_SECRET, SOURCEBOT_ENCRYPTION_KEY (one-time generated)
#   (after first `make up`) SOURCEBOT_API_KEY — create in Sourcebot UI → Settings → API Keys
```

For **prod** mode, also:
```bash
cp .env.prod.example .env.prod
# fill in production values:
#   strong POSTGRES_PASSWORD (NOT 'changeme')
#   fresh SOURCEBOT_AUTH_SECRET, SOURCEBOT_ENCRYPTION_KEY (don't reuse dev)
#   GITLAB_TOKEN if you want Sourcebot to clone repos itself
```

## What changes between local and prod

The Docker image is **the same**. Only configuration changes via env file + the [`compose.prod.yaml`](compose.prod.yaml) overlay:

| | Local (`make up`) | Prod (`make prod`) |
|---|---|---|
| Env file | `.env` | `.env.prod` |
| Sourcebot host port | `${SOURCEBOT_HOST_PORT:-13000}` exposed | not exposed (docker network only) |
| Uvicorn log level | info | warning |
| `--proxy-headers` | off | on (for reverse proxies) |
| Sourcebot HTTP timeout | 30 s | 60 s |
| `SOURCEBOT_DISABLE_MCP_FALLBACK` | unset (fallback ok) | `true` (fail fast) |
| Container restart policy | `unless-stopped` | `always` |
| Postgres password | typically `changeme` | should be strong |

The app image is built once and reused across modes. Switching modes is just a different `--env-file` + overlay — no rebuild needed.

## Quick start (dev)

```bash
make dev    # backends in Docker, FastAPI + Vite on host
# open http://localhost:${UI_PORT}    (Vite proxies API calls to FastAPI on :${API_PORT})
# defaults: UI_PORT=15173, API_PORT=18000 — override in .env if they collide.
```

## Quick start (local)

```bash
make up     # everything in Docker
# open http://localhost:${API_PORT}/ui    (default :18000/ui)
```

Wait ~30 s on first run for Sourcebot to index, then:

```bash
# from the host
curl -sS http://localhost:18000/health
# {"status":"ok"}

# Every Q&A request routes through an adapter — pick one from /v1/adapters.
curl -sS http://localhost:18000/v1/adapters/cursor/ask \
  -H 'content-type: application/json' \
  -d '{"query":"Where is the supplier creation endpoint defined?"}' \
  | jq '{adapter: .result.adapter, run_id, model: .result.metrics.model}'
```

Optional — metrics digest inside the container:

```bash
docker exec -it tech-decomposition-app tech-decomposition-analyze --since 24h
```

## Services

| Service | Port (host) | Role |
|---|---|---|
| `app` (tech-decomposition) | 8000 | FastAPI |
| `sourcebot` | 3000 | Code index + AI Q&A backend |
| `postgres` | — | Sourcebot persistence (docker network only) |
| `redis` | — | Sourcebot queue (docker network only) |

## Environment variables

Required:
- `GEMINI_API_KEY` — default provider for both Flash (enrich) and Pro (decompose) stages.
- `SOURCEBOT_AUTH_SECRET`, `SOURCEBOT_ENCRYPTION_KEY` — Sourcebot infra secrets (one-time generate via `openssl rand -base64 33` / `... -base64 24`).

Required after first start:
- `SOURCEBOT_API_KEY` — generate in the Sourcebot UI (Settings → API Keys), paste into `.env`, then `docker compose restart app`.

Optional (each opt-in by setting the credential):
- `ANTHROPIC_API_KEY` — to use `anthropic:claude-...` as a per-stage model.
- `OPENAI_API_KEY` — to use `openai:gpt-...`.
- `OPENROUTER_API_KEY` — to use `openrouter:anthropic/claude-...` or any other OR-routed model.
- `CUSTOM_LLM_BASE_URL` + `CUSTOM_LLM_API_KEY` — any OpenAI-compatible endpoint (LiteLLM proxy, vLLM, Ollama).
- `GITLAB_TOKEN` — needed only when the `gitlab-company` connector in [config/sourcebot/config.json](config/sourcebot/config.json) is active.

Per-stage model selection — set in `.env`:

```bash
ENRICH_MODEL=gemini:gemini-2.5-flash
DECOMPOSE_MODEL=openrouter:anthropic/claude-opus-4-5
```

Switching providers requires no code changes; the model registry in
[core/llm_registry.py](src/tech_decomposition/core/llm_registry.py) resolves `provider:name`
strings at runtime.

## Repo management — two paths

### Path A — local-mount (default, simpler)

Sourcebot indexes whatever's under `REPOS_ROOT` on the host. You manage clones
manually:

```bash
mkdir -p ~/repos
cd ~/repos
git clone git@github.com:your-company/backend-api.git
git clone git@github.com:your-company/frontend-webapp.git
# ...
```

Set `REPOS_ROOT=/Users/me/repos` in `.env`. Sourcebot re-indexes on its own
schedule (~10 min). Run `git pull` on the host whenever you want fresh code; a
cron job is enough:

```bash
# crontab -e
*/15 * * * * cd /Users/me/repos && for d in */; do (cd "$d" && git pull --ff-only); done
```

### Path B — Sourcebot-managed clones (preferred for prod)

Add a `GITLAB_TOKEN` (scopes: `read_api`, `read_repository`) to `.env`. The
`gitlab-company` connector in [config/sourcebot/config.json](config/sourcebot/config.json)
will then clone every repo under the `your-company/backend-api` group into Sourcebot's
own data volume and keep them updated on a schedule. No host-side `git pull`
needed. To narrow scope, edit the `groups` / add `projects` arrays in that
config and restart Sourcebot:

```bash
docker compose restart sourcebot
```

Path A and Path B can coexist — both connectors are listed in the config. If
you only need one, comment out the other.

## Observability

- **Per-run metrics**: `outputs/metrics/runs.jsonl` (one stage row + one run row per invocation, JSONL).
- **Quick view**: `uv run tech-decomposition-analyze` (or the installed `tech-decomposition-analyze` on `$PATH`), with `--since 24h`, `--by-engine sourcebot`, `--mode-filter ask` filters.
- **Healthcheck**: `GET /health` returns `{"status":"ok"}`. Docker's healthcheck hits this every 10 s.

## Web UI

A React + Tailwind + DaisyUI single-page app lives under [web/](web/). It
talks to the FastAPI endpoints (`/ask`, `/runs`) — no separate backend.

**Dev (live reload):**

```bash
cd web
npm install        # first time only
npm run dev        # → http://localhost:${UI_PORT}, proxies /v1 + /runs to :${API_PORT}
```

Run the FastAPI server in another shell (`uv run uvicorn tech_decomposition.api:app --port …` or
`docker compose up app`); the Vite dev proxy forwards API calls to it.
Both ports come from `.env` (`UI_PORT` / `API_PORT`).

**Prod (bundled into FastAPI):**

```bash
cd web && npm run build       # writes web/dist/
# FastAPI auto-detects web/dist and mounts it at /ui
# Visit http://localhost:${API_PORT}/ui    (default :18000/ui)
```

The static mount in [api.py](src/tech_decomposition/api.py) is gated on the directory
existing, so if `web/dist` isn't present the API runs without the UI.

## Hosting beyond local docker

The `app` service is a stateless FastAPI container — fits any container
runtime. Two common shapes:

**Single managed container** (Cloud Run / Fly.io / Render):
- Build & push the image: `docker build -t registry.example.com/tech-decomposition:latest . && docker push ...`
- Point `SOURCEBOT_URL` at your hosted Sourcebot deployment.
- Set env vars in the platform's secret store.

**Internal Kubernetes / Nomad**:
- Use the same image; mount a `PersistentVolume` at `/app/outputs`.
- Front with your usual ingress.
- For multi-tenant scenarios, add an auth layer at the ingress (Cloudflare Access, oauth2-proxy, etc.) — the app itself doesn't authenticate today, by design.

The full stack (with Sourcebot) needs more thought — Sourcebot has its own
database and index volume requirements. For prod, **either** point at a
managed Sourcebot **or** run the whole compose stack on a single VM behind a
reverse proxy.

## Upgrades

```bash
git pull
docker compose build app           # rebuild only tech-decomposition
docker compose up -d               # apply
docker compose logs -f app         # tail
```

To wipe Sourcebot's index (re-clone everything from scratch):

```bash
docker compose down -v             # CAUTION: deletes named volumes
docker compose up -d
```
