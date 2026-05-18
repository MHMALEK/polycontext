<div align="center">

  <p>
    <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
    <img src="https://img.shields.io/badge/Node-22+-339933?style=for-the-badge&logo=nodedotjs&logoColor=white" alt="Node" />
    <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI" />
    <img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker" />
    <img src="https://img.shields.io/badge/MIT-License-yellow?style=flat" alt="License" />
  </p>
  <p>
    <img src="https://img.shields.io/badge/Sourcebot-141414?style=flat" alt="Sourcebot" />
    <img src="https://img.shields.io/badge/Claude-D4A574?style=flat&logo=anthropic&logoColor=white" alt="Claude" />
    <img src="https://img.shields.io/badge/OpenAI-412991?style=flat&logo=openai&logoColor=white" alt="OpenAI" />
    <img src="https://img.shields.io/badge/Gemini-4285F4?style=flat&logo=google&logoColor=white" alt="Gemini" />
    <img src="https://img.shields.io/badge/Cursor-000000?style=flat&logo=cursor&logoColor=white" alt="Cursor" />
    <img src="https://img.shields.io/badge/Cline-0EA5E9?style=flat" alt="Cline" />
    <img src="https://img.shields.io/badge/OpenCode-22C55E?style=flat" alt="OpenCode" />
  </p>

  <h1>tech-decomposition</h1>
  <p align="center">
    One JSON API. Seven code-AI backends. Two modes (ask, decompose).<br/>
    Optional grounded retrieval, augmented with LSP-backed semantic search via Serena MCP.<br/>
    A web UI to drive both, an eval harness to compare them.
  </p>

  <p>
    <a href="#results">Results</a> ·
    <a href="#what-it-is">What it is</a> ·
    <a href="#adapters">Adapters</a> ·
    <a href="#http-api">HTTP API</a> ·
    <a href="#grounded-retrieval-opt-in">Grounding</a> ·
    <a href="#serena-mcp-optional">Serena</a> ·
    <a href="#web-ui">UI</a> ·
    <a href="#quick-start">Quick start</a>
  </p>

</div>

---

## Results

We don't ship features without measuring them. Every number below is from
the eval harness in [`eval/`](eval/) running real questions from production
Slack channels against the actual codebase (5 git repos, ~1.2M LOC), with
the answers scored by both a deterministic rubric (`must_mention` substrings)
and a Gemini Flash LLM-judge against a hand-written gold answer.

**Best result, full bake-off (5 hand-curated questions × 5 adapters × 2 modes, scored by rubric + LLM-judge against hand-written gold answers):**

| Recommended config | Score |
|---|---|
| **`cursor` ungrounded** (default) | **0.773** |
| `gemini` ungrounded (agentic, strong prompt) | 0.677 |
| `cursor` grounded — for enumeration questions only | 0.731 (and 0.984 on `q-user-roles` specifically) |

Everything else trails. Cline_sdk and OpenCode both scored in the 0.26–0.48 range and showed high run-to-run variance (Cline swung 0.79 on the same question across two runs). Sourcebot's own chat scored 0.45.

Per-question pattern that emerged:

| Question pattern | What works |
|---|---|
| Bounded symbol lookup (`q-jwt-stateless`, `q-geojson-country-check`) | Cursor / Gemini ungrounded, both hit 1.0 |
| Enumeration ("list all X" — `q-user-roles`) | Cursor **grounded** (0.984) |
| Behavior trace | Gemini ungrounded (1.0 on `q-data-sharing-geolocation` from 3-case eval) |
| Validator/regex lookup (`q-farm-name-unicode`) | **Nothing works yet.** Every adapter 0.10–0.46. Real product gap. |

Three things drove those numbers:

1. **Forced tool use for Gemini.** The default `_ASK_SYSTEM` prompt told Gemini that tools "were available." Gemini ignored them — ~1 call per question, answers from training knowledge. The new prompt makes tool use *mandatory* with a 3-step workflow. Tool calls jumped from 1 → 7.67 per case. Score 0.64 → 0.79.

2. **Serena MCP** wired into every adapter that supports MCP — Cursor, Claude Code, Cline, OpenCode, OpenAI Agents. Cline + OpenCode actually *use* Serena's LSP-backed tools (`find_symbol`, `search_for_pattern`, `find_referencing_symbols`); Cursor's Composer-2 and gpt-4o-mini saw the tools but never called them. The model matters more than the tools.

3. **Serena as a grounding source.** For adapters that can't take MCP themselves (Gemini direct, Sourcebot's `/api/chat`), the grounding pipeline now runs Sourcebot search AND Serena search in parallel and merges the snippets.

**What didn't pan out (recorded honestly):**

- **Grounding is not a useful default.** Pre-fetched snippets make the model trust the first hit and stop investigating, which catastrophically fails on enumerative and trace questions (Gemini dropped 1.0 → 0.148 on `q-geojson-country-check` when grounded). Default is `grounded=false`. Flip on per-request for enumeration questions or for OpenCode specifically.
- **Cursor (Composer-2) ignored Serena.** It listed Serena's 28 tools every request but never called one. Wiring is in place for free if a future Cursor model changes that.
- **The regex/validator pattern is unsolved.** None of the adapters, grounding, MCP, or prompt tricks moved `q-farm-name-unicode` above 0.46.

---

## What it is

A thin **uniform contract** over a handful of code-AI tools — Claude Code, Cursor, Cline, Gemini, OpenAI Agents, OpenCode, and Sourcebot. Same JSON request shape, same response shape, regardless of which one runs it. Two modes:

- **`ask`** — code Q&A. The adapter explores the workspace (each SDK uses its own tools) and answers.
- **`decompose`** — turn a ticket / task into a typed `Decomposition`: overview, affected repos, subtasks (with files + acceptance criteria), risks, open questions.

You drive it via FastAPI (`/v1/adapters/{name}/{ask,decompose}`) or the bundled React UI. Swap adapters per request. Run the same input through several at once (`/v1/bakeoff/{job}`) and compare. Optionally pre-fetch Sourcebot-indexed snippets as a grounding step.

That's the whole project. Nothing else hides under the hood.

---

## Adapters

Registered in `src/tech_decomposition/adapters/registry.py`. Most SDK-backed adapters run inside `services/agent-node` (Node, Fastify) — the Python side is a thin HTTP client. `sourcebot` is the exception: it's a Python adapter that talks directly to Sourcebot's chat API.

| Adapter | Backend | Runtime | Capabilities | Serena MCP | Default mode |
|---|---|---|---|---|---|
| **`cursor`** | [`@cursor/sdk`](https://www.npmjs.com/package/@cursor/sdk) | Node | `ask`, `decompose` | per-request* | **ungrounded (0.773)** — recommended |
| **`gemini`** | [`@google/genai`](https://www.npmjs.com/package/@google/genai) | Node | `ask`, `decompose` | via grounding | ungrounded agentic (0.677) |
| **`sourcebot`** | Sourcebot `/api/chat/blocking` | Python | `ask` | via grounding | grounded (0.45) — no agent loop |
| **`opencode`** | [`@opencode-ai/sdk`](https://opencode.ai/) | Node | `ask`, `decompose` | **per-request ✓** | grounded (0.406) |
| **`cline_sdk`** | [`@cline/sdk`](https://www.npmjs.com/package/@cline/sdk) | Node | `ask`, `decompose` | **per-request ✓** | noisy — judge case-by-case |
| **`claude_code`** | [`@anthropic-ai/claude-agent-sdk`](https://www.npmjs.com/package/@anthropic-ai/claude-agent-sdk) | Node | `ask`, `decompose` | per-request | (Docker arm64-musl native binary issue — works on host) |
| **`openai_agents`** | [`@openai/agents`](https://github.com/openai/openai-agents-js) | Node | `ask`, `decompose` | per-request* | tier-1 TPM caps gpt-4.1; gpt-4o-mini works |

\* Wired but the underlying model (Composer-2 / gpt-4o-mini) doesn't call MCP tools in measurements so far. Cline (Gemini Pro driving) and OpenCode (Gemini Pro driving) do call them and benefit.

Each Node-side adapter exposes the SDK's own workspace tools (`read_file`, `list_directory`, `grep_search`, `search_files`) so its agent can explore your local clones at `REPOS_ROOT`. When `SERENA_URL` is set, Cursor / Claude Code / Cline / OpenCode / OpenAI Agents get Serena's tools registered automatically over MCP; Gemini and Sourcebot (which can't take MCP) get Serena via the grounding step instead.

Discover what's installed + healthy at runtime:

```http
GET /v1/adapters
```

---

## HTTP API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness |
| `GET` | `/v1/adapters` | List adapters, capabilities, health |
| `POST` | `/v1/adapters/{name}/ask` | Code Q&A through one adapter |
| `POST` | `/v1/adapters/{name}/decompose` | Ticket → `Decomposition` |
| `POST` | `/v1/grounding/retrieve` | Sourcebot search alone — see snippets + metrics |
| `POST` | `/v1/bakeoff/{job}` | `ask`\|`decompose` — fan one input across N adapters |
| `GET` | `/runs`, `/runs/{id}` | History list / detail |
| `POST` | `/runs/{id}/replay` | Re-run a saved request |

**Request example (ask, grounded):**

```bash
curl -sS http://localhost:18000/v1/adapters/cursor/ask \
  -H 'content-type: application/json' \
  -d '{"query":"how does upload master data work?","grounded":true}' \
  | jq '{adapter: .result.adapter, snippets: (.result.grounding.snippets|length), tokens: .result.metrics.tokens_in}'
```

**Response shape (ask):**

```json
{
  "run_id": "ab12...",
  "thread_id": "ab12...",
  "result": {
    "adapter": "cursor",
    "answer": "...",
    "citations": [],
    "metrics": { "tokens_in": 8200, "tokens_out": 410, "duration_ms": 7400, "model": "composer-2", ... },
    "grounding": {
      "snippets": [{ "repo": "...", "path": "...", "start_line": 23, "end_line": 47, "content": "...", "url": "..." }],
      "grounding_block": "## Relevant code (grounded retrieval)\n...",
      "metrics": { "duration_ms": 850, "snippet_count": 6, "total_chars": 14200, "sources": ["sourcebot"], "sourcebot_files_seen": 6 }
    }
  }
}
```

`result.grounding` is only present when the request set `grounded: true`.

---

## Grounded retrieval (opt-in)

Grounding is a **single composable step**, not a default. You ask for it per request:

- **As a standalone call**: `POST /v1/grounding/retrieve` returns the snippets, the rendered markdown block, and timing/char metrics — no LLM call. Useful to see *what* grounding produces and *what it costs* before deciding to inject it.
- **As a flag on ask**: `POST /v1/adapters/{name}/ask` with `{"grounded": true}` runs the same retrieval, prepends the block to the user prompt, then calls the adapter. The adapter's `metrics` (tokens/cost) and `grounding.metrics` (latency/snippets/chars) are reported separately so you can compare grounded vs not on the same question.

**Two parallel sources** (when both are configured):

1. **Sourcebot `/api/search`** — zoekt-style regex search across all indexed repos. Returns chunked file matches with surrounding context lines. Picks the LLM Sourcebot's chat uses from `config/sourcebot/config.json`.
2. **Serena `search_for_pattern`** — LSP-backed semantic search via MCP. Adds structured matches that complement Sourcebot's regex hits. Only runs when `SERENA_URL` is set; degrades silently otherwise.

The two sources fan out in parallel, are deduped by file path, and merged up to `top_k` snippets. `GroundingMetrics` reports `sourcebot_files_seen` and `serena_hits` separately so you can see which source contributed what.

### Default: leave grounding off

The 5-case bake-off (see [eval/outputs/bakeoff-20260518T134739Z/](eval/outputs/bakeoff-20260518T134739Z/)) settles this empirically: the **best single result** is **`cursor` ungrounded at 0.773**. Grounding helps on one specific question pattern (enumeration — "list all X") and hurts on everything else, sometimes catastrophically (Gemini dropped 1.0 → 0.148 on `q-geojson-country-check` when grounded).

| Adapter | Avg ungrounded | Avg grounded | Verdict |
|---|---|---|---|
| `cursor` | **0.773** | 0.731 | Default off. Flip on per-request for enumeration questions (`q-user-roles`-style). |
| `gemini` | **0.677** | 0.428 | Default off. Grounding consistently hurts. |
| `sourcebot` | 0.450 | 0.384 | No agent loop — grounding can't make it worse than its own chat. Leave on. |
| `cline_sdk` | 0.263–0.453 (noisy) | 0.301–0.483 (noisy) | Either way; high run-to-run variance, judge case-by-case. |
| `opencode` | 0.275 | **0.406** | Default on. Grounding helps this adapter on average. |

**Bottom line for callers**: `grounded` is already `false` by default on `AdapterAskInput`. Don't change that. The two exceptions worth a `grounded=true` request are: any adapter on an enumeration question, and OpenCode in general.

What grounding does **not** include (vs. the v0 pipeline that was removed):

- No ChromaDB / Sentence Transformers local index
- No tree-sitter repo map / structural prelude
- No reranker

Reason: each SDK adapter has its own retrieval inside its agent loop. Adding our own embedding store on top adds latency without proportional gain — Serena's LSP-backed lookups subsume what those layers were going to provide, and are scoped to the adapters that actually benefit.

---

## Serena MCP (optional)

[Serena](https://github.com/oraios/serena) is an LSP-backed MCP server that
exposes semantic code tools (`find_symbol`, `find_references`,
`get_symbols_overview`, `search_for_pattern`, ...) that go beyond plain
grep. When `SERENA_URL` is set in `.env`, two things happen automatically:

1. **Per-request MCP wiring** for Cursor, Claude Code, Cline, OpenCode, and
   OpenAI Agents — Serena's tools are registered alongside the SDK's
   native workspace tools so the agent can call them as it explores.
2. **Grounding augmentation** for Gemini direct and Sourcebot (the two
   adapters whose runtimes don't accept MCP) — Serena
   `search_for_pattern` is fanned out per term in parallel with Sourcebot
   and the results are merged into the snippet block.

Run Serena out-of-band:

```bash
uvx --from git+https://github.com/oraios/serena serena start-mcp-server \
  --transport streamable-http --port 9121 --context agent \
  --project "$REPOS_ROOT"

# in .env:
SERENA_URL=http://localhost:9121/mcp           # host agent-node
# or:
SERENA_URL=http://host.docker.internal:9121/mcp  # docker agent-node
```

Leave `SERENA_URL` unset to skip Serena entirely — every adapter falls back
to its prior behavior.

---

## Web UI

A React + Tailwind + DaisyUI single-page app under [`web/`](web/). Two tabs:

- **Ask** — chat-style threaded conversation. Pick adapter, optionally toggle **Grounded**, send. The answer card shows the meta strip (model, wall, tokens, cost), the markdown answer, citations, and (when grounded) a collapsible grounding panel that lists each injected snippet inline.
- **Decompose** — single-shot ticket → structured `Decomposition`. Renders subtasks as cards (title, repo, complexity badge, files with optional Sourcebot URLs, acceptance criteria) plus collapsible risks / open questions, with raw markdown tucked at the bottom.

History sidebar covers both modes. Click any prior run to see its details (including its grounding payload if it had one).

```bash
cd web && npm install && npm run dev   # → http://localhost:15173, vite proxies API to :18000
```

For prod the UI is bundled with `npm run build`; FastAPI serves `web/dist/` at `/ui` when present.

---

## Architecture

```mermaid
flowchart TB
  subgraph clients[Clients]
    UI[Web UI]
    CLI[curl / scripts / eval]
  end

  API[FastAPI<br/>/v1/adapters/.../ask<br/>/v1/adapters/.../decompose]

  subgraph grounding[Grounding pre-fetch — when grounded=true]
    direction LR
    SBS[Sourcebot /api/search<br/>zoekt regex]
    SER1[Serena search_for_pattern<br/>LSP-backed]
    MERGE[merge + dedupe<br/>up to top_k snippets]
    SBS --> MERGE
    SER1 -. parallel .-> MERGE
  end

  subgraph adapters[Adapters]
    direction LR
    AN[agent-node<br/>Fastify + SDKs]
    SBA[Sourcebot adapter<br/>/api/chat/blocking]
  end

  subgraph sdks[Per-request SDK runtimes]
    direction LR
    Cline[Cline SDK]
    OC[OpenCode SDK]
    Cur[Cursor SDK]
    CC[Claude Code SDK]
    Gem[Gemini @google/genai]
    OAI[OpenAI Agents SDK]
  end

  SER2[Serena MCP<br/>find_symbol · find_references<br/>get_symbols_overview · ...]
  RS[(SQLite runstore<br/>history + replay)]

  UI --> API
  CLI --> API
  API -- if grounded=true --> grounding
  grounding -. snippets prepended .-> API
  API --> AN
  API --> SBA
  AN --> sdks
  Cline -. MCP per request .-> SER2
  OC -. MCP per request .-> SER2
  Cur -. MCP per request .-> SER2
  CC -. MCP per request .-> SER2
  OAI -. MCP per request .-> SER2
  API --> RS

  classDef extra fill:#fef9c3,stroke:#ca8a04;
  class grounding,SER2 extra;
```

The two yellow blocks are optional and gated on env config:
- **Grounding** runs only when the request has `grounded=true`; degrades to Sourcebot-only when Serena isn't configured.
- **Serena MCP** is wired into the tool-using adapters only when `SERENA_URL` is set.

Without either, the path is straight: API → adapter → SDK → LLM → response → runstore.

---

## Repo layout

```text
src/tech_decomposition/
├── adapters/            # 7 adapters + registry + Decomposition prompt helpers
├── clients/sourcebot.py # Sourcebot HTTP client (chat/blocking + answer-style suffix)
├── core/
│   ├── context.py       # RunContext (per-request state)
│   ├── runstore.py      # SQLite-backed run history + replay
│   ├── grounding.py     # GroundedContext + retrieve_grounded_context()
│   ├── llm_registry.py  # model spec → pydantic-ai model (for any in-process LLM use)
│   └── usage.py         # token-usage extraction
├── api.py               # FastAPI surface
├── analyze.py           # log analyzer CLI (entry: tech-decomposition-analyze)
├── config.py            # Settings (env-driven)
└── models.py            # Decomposition, Subtask, Snippet, Ticket, EnrichedQuery

services/agent-node/     # Fastify + JS SDKs for Claude / Cursor / Cline / Gemini / OpenAI Agents / OpenCode
web/                     # Vite + React + Tailwind + DaisyUI
eval/                    # Bake-off: TOML/YAML cases, runner, scorer, report (eval/bakeoff)
config/sourcebot/        # config.json — Sourcebot's models + connectors
```

---

## Stack

**Python:** FastAPI, Uvicorn, Pydantic v2, pydantic-ai, httpx, Rich, uv (packaging).

**Node:** Fastify, Zod, official JS SDKs (`@google/genai`, `@openai/agents`, `@anthropic-ai/claude-agent-sdk`, `@cursor/sdk`, `@cline/sdk`, `@opencode-ai/sdk`).

**Web:** React 18, Vite, Tailwind 4, DaisyUI, react-markdown + remark-gfm.

**Infra (Compose):** Sourcebot, PostgreSQL 16, Redis 8.

---

## Quick start

**Prereqs:** Python 3.11+, [uv](https://github.com/astral-sh/uv), Node 22+, Docker (for the Sourcebot stack).

```bash
cp .env.example .env
# Fill in:
#   At least one provider key (OPENAI_API_KEY / ANTHROPIC_API_KEY / GEMINI_API_KEY).
#   SOURCEBOT_AUTH_SECRET + SOURCEBOT_ENCRYPTION_KEY (openssl rand -base64 ...).
#   After first start: SOURCEBOT_API_KEY (from Sourcebot UI → Settings → API Keys).
```

Point Sourcebot at local clones:

```env
REPOS_ROOT=./repos
REPOS=backend-api,frontend-webapp
```

```bash
make install && make ui-install
make dev-all            # Docker: Postgres + Redis + Sourcebot. Host: FastAPI + Vite + agent-node.
# or: make up           # everything in Docker (uses bundled UI)
```

Smoke test:

```bash
curl -sS http://localhost:18000/health
curl -sS http://localhost:18000/v1/adapters | jq '.adapters[] | {name, health}'
```

Open the UI at `http://localhost:${UI_PORT:-15173}` (dev) or `http://localhost:${API_PORT:-18000}/ui` (built).

---

## Evaluation

A bake-off runner under [`eval/`](eval/) — fan a TOML/YAML case set across
one or more adapters, score with two layers, write a markdown report.

**Scoring layers** (`eval/bakeoff/scorer.py`):

1. **Rule-based** — deterministic substring + structure checks (`must_mention`,
   `min_chars`, `min_citations`). Cheap, runs on every case.
2. **LLM-as-judge** — when a case carries a hand-written `gold_answer`, a
   single Gemini Flash call grades the adapter's answer on `coverage`
   (fraction of facts from gold present) and `accuracy` (no contradictions),
   merged as weighted checks. Enable with `--use-judge`.

**Cases** live in [`eval/questions.toml`](eval/questions.toml) — 13 real
questions harvested from production Slack channels with gold answers
written by domain experts.

```bash
make eval-adapters                              # show registered adapters + health
make eval-cases                                 # list discovered cases (JOB=ask|decompose)
uv run python -m eval.bakeoff.cli run --job ask --adapters cline_sdk,opencode --use-judge
uv run python -m eval.bakeoff.cli run --job ask --adapters gemini --grounded --use-judge
```

Output: `eval/outputs/eval-<timestamp>/report.md` + `summary.json` + per-run
JSON. The report includes a leaderboard, per-case score breakdown, full
answer bodies for diff-style comparison across adapters, and (when enabled)
LLM-judge verdicts inline.

---

## Observability

- **Run history:** every API call is persisted in `outputs/runstore.db` (SQLite). The UI sidebar reads from `/runs`; full detail at `/runs/{id}`.
- **Metrics digest:** `uv run tech-decomposition-analyze --since 24h` summarizes recent runs by adapter / cost / wall.

---

<p align="center"><sub>One uniform contract. Many code-AI backends. Bring your own keys.</sub></p>
