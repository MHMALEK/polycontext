# Deployment

Single-page guide for getting ticket-recon running locally and shipping it to a
remote host. Built around `docker compose` — the whole stack (Postgres, Redis,
Sourcebot, Serena, ticket-recon) lives in one file.

## Quick start

```bash
cp .env.example .env       # fill in at minimum:
                           # GEMINI_API_KEY
                           # SOURCEBOT_AUTH_SECRET, SOURCEBOT_ENCRYPTION_KEY
make up                    # builds the image (with UI) + brings up the stack
```

Then open:
- **UI** — http://localhost:8000/ui
- **API** — http://localhost:8000
- **Sourcebot** — http://localhost:3000

`make up` is shorthand for `docker compose up -d --build` plus a status banner.
Run `make` (no args) to see all targets.

For development with hot reload (FastAPI + Vite on the host, backends in
Docker):

```bash
make dev   # runs both servers; Ctrl-C stops both
```

Wait ~30 s on first `make up` for Sourcebot to index, then:

```bash
# from the host
curl -sS http://localhost:8000/health
# {"status":"ok"}

curl -sS http://localhost:8000/ask \
  -H 'content-type: application/json' \
  -d '{"question":"Where is the supplier creation endpoint defined?"}' \
  | jq '{engine, wall_seconds, cost_usd, citations: .citations | length}'
```

CLI alternative (one-shot, in the running container):

```bash
docker exec -it ticket-recon-app ticket-recon ask "..."
docker exec -it ticket-recon-app ticket-recon decompose --ticket-key DEV-7543
docker exec -it ticket-recon-app ticket-recon analyze --since 24h
```

## Services

| Service | Port (host) | Role |
|---|---|---|
| `app` (ticket-recon) | 8000 | FastAPI + CLI |
| `sourcebot` | 3000 | Code index + AI Q&A backend |
| `serena` | 9121 | LSP-backed MCP server (experimental local agent) |
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
- `GITLAB_TOKEN` — needed only when the `gitlab-tract` connector in [config/sourcebot/config.json](config/sourcebot/config.json) is active.
- `JIRA_BASE_URL`, `JIRA_EMAIL`, `JIRA_API_TOKEN` — needed for `decompose --ticket-key` and `--post-to-jira`.

Per-stage model selection — set in `.env`:

```bash
ENRICH_MODEL=gemini:gemini-2.5-flash
DECOMPOSE_MODEL=openrouter:anthropic/claude-opus-4-5
```

Switching providers requires no code changes; the model registry in
[core/models.py](src/ticket_recon/core/models.py) resolves `provider:name`
strings at runtime.

## Repo management — two paths

### Path A — local-mount (default, simpler)

Sourcebot indexes whatever's under `REPOS_ROOT` on the host. You manage clones
manually:

```bash
mkdir -p ~/repos
cd ~/repos
git clone git@gitlab.com:tract1/application/api/traceability.git
git clone git@gitlab.com:tract1/application/frontend.git
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
`gitlab-tract` connector in [config/sourcebot/config.json](config/sourcebot/config.json)
will then clone every repo under the `tract1/application` group into Sourcebot's
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
- **Quick view**: `ticket-recon analyze`, with `--since 24h`, `--by-engine sourcebot`, `--mode-filter ask` filters.
- **Healthcheck**: `GET /health` returns `{"status":"ok"}`. Docker's healthcheck hits this every 10 s.

## Web UI

A React + Tailwind + DaisyUI single-page app lives under [web/](web/). It
talks to the FastAPI endpoints (`/ask`, `/runs`) — no separate backend.

**Dev (live reload):**

```bash
cd web
npm install        # first time only
npm run dev        # → http://localhost:5173, proxies /ask + /runs to :8000
```

Run the FastAPI server in another shell (`ticket-recon serve` or
`docker compose up app`); the Vite dev proxy forwards API calls to it.

**Prod (bundled into FastAPI):**

```bash
cd web && npm run build       # writes web/dist/
# FastAPI auto-detects web/dist and mounts it at /ui
# Visit http://localhost:8000/ui
```

The static mount in [api.py](src/ticket_recon/api.py) is gated on the directory
existing, so if `web/dist` isn't present the API runs without the UI.

## Hosting beyond local docker

The `app` service is a stateless FastAPI container — fits any container
runtime. Two common shapes:

**Single managed container** (Cloud Run / Fly.io / Render):
- Build & push the image: `docker build -t registry.example.com/ticket-recon:latest . && docker push ...`
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
docker compose build app           # rebuild only ticket-recon
docker compose up -d               # apply
docker compose logs -f app         # tail
```

To wipe Sourcebot's index (re-clone everything from scratch):

```bash
docker compose down -v             # CAUTION: deletes named volumes
docker compose up -d
```
