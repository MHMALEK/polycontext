# ticket-recon

Turn a Jira ticket into an **AI-friendly tech decomposition** with multi-repo
context and clickable GitLab permalinks — designed so an autonomous coding
agent can pick up the subtasks and act.

```
Jira ticket  ──fetch──▶  Ticket
                            │
                            │  enrich  ─── Gemini 2.5 Flash + Pydantic AI ──▶ EnrichedQuery
                            │                                                (entities, code keywords,
                            │                                                 suspected repos, queries)
                            ▼
                     RetrievedContext ◀── retriever fan-out
                            │              · ripgrep   (local clones)
                            │              · Sourcebot (indexed multi-repo search)
                            │              · Serena    (LSP-backed semantic search)
                            │
                            │  decompose ─── Gemini 2.5 Pro + Pydantic AI ──▶ Decomposition
                            │                                                (typed subtasks per repo,
                            ▼                                                 acceptance criteria, risks)
                  Markdown file with GitLab permalinks
                  + (optional) posted as a Jira comment
                  + metrics row appended to outputs/metrics/runs.jsonl
```

**Why a two-model pipeline?** Flash extracts structure from messy ticket text
for ~$0.001 per ticket; Pro only ever sees curated, retrieved snippets so the
expensive call stays small. Empirically: ~$0.03 / 28s per ticket on the four
Tract application repos.

---

## Status

Real, end-to-end working. Tested against [`code-geppetto.atlassian.net`](https://code-geppetto.atlassian.net)
and the four Tract application repos (`traceability`, `frontend`,
`data-cloud-functions`, `data`).

## Retrievers (cheap mode)

| retriever  | how it's used |
|------------|---|
| `anchor`   | extracts every file path / filename mentioned in the ticket, reads each one whole, and follows their imports one hop (Python `ast`, TS regex). Pulls the load-bearing files no matter what keywords the model invents. Score 10.0 (anchors) / 8.0 (followed imports). |
| `ripgrep`  | fast keyword search on local clones, N lines of context |
| `sourcebot`| indexed multi-repo search with `repo:`/`lang:` filters, ranked results |
| `serena`   | LSP-backed semantic search, sees all 4 repos as one project via `activate_project` |

Test files (`*_test.py`, `*.test.ts`, `/tests/...`) are auto-down-ranked
across all sources so implementation beats tests in a constrained context
window. Failures or unset endpoints degrade silently — the pipeline never
fails just because one upstream is down.

## Two modes: cheap and deep

|                | **cheap** (default)     | **deep** (agentic)               |
|----------------|-------------------------|----------------------------------|
| Pipeline       | enrich → static retrieval → decompose | enrich → Pro loops with tools |
| Per-ticket cost| ~$0.05–$0.10            | ~$0.25–$1.00                     |
| Latency        | 50–80s                  | 80s–5min                         |
| Quality (easy) | very good               | very good (not worth paying for) |
| Quality (multi-repo migration) | misses dependency chains | catches dependencies + contradictions |

**`mode` field** on `DecomposeRequest` / `--mode` CLI flag:

- `cheap` — always single-pass
- `deep` — always agentic
- `auto` (default) — cheap unless enrichment confidence is `low`, the ticket
  mentions ≥3 repos, or the body contains migration/refactor cues
  (`migrate`, `port to`, `move to`, `refactor across`). Then deep.

Deep-mode tools the model gets:

- `read_file(repo, path)` — full file with line numbers
- `grep(pattern, repo?)` — ripgrep across one or all repos
- `list_directory(repo, path)` — see what's there
- `find_symbol(name_path_pattern)` — Serena LSP symbol lookup
- `get_file_overview(repo, path)` — Serena structural outline

Bounded by `DEEP_MAX_REQUESTS = 20` model turns (in `deep_decompose.py`) so
a runaway exploration can't blow the bill.

---

## Prerequisites

- Python ≥ 3.11
- [Docker](https://docs.docker.com/desktop/) (for Sourcebot + Serena)
- [`uv`](https://github.com/astral-sh/uv) (recommended) or `pip`
- `ripgrep` on PATH: `brew install ripgrep`
- A Gemini API key (Google AI Studio)
- Optional: Jira API token (only if you submit ticket keys instead of raw text)

---

## Setup

```bash
git clone <this-repo>
cd ticket-recon
cp .env.example .env
```

### 1. Fill in `.env`

At minimum:

```bash
GEMINI_API_KEY=...                              # required
REPOS_ROOT=/absolute/path/to/parent/of/repos   # where the repo clones live
```

If you want Jira fetching + comment posting:
```bash
JIRA_BASE_URL=https://<your-org>.atlassian.net
JIRA_EMAIL=you@example.com
JIRA_API_TOKEN=...
```

Generate the Sourcebot infra secrets:
```bash
echo "SOURCEBOT_AUTH_SECRET=$(openssl rand -base64 33)" >> .env
echo "SOURCEBOT_ENCRYPTION_KEY=$(openssl rand -base64 24)" >> .env
```

### 2. Install the Python package

```bash
uv venv
uv pip install -e .
```

### 3. Start Sourcebot + Serena

```bash
docker compose up -d
docker compose logs -f sourcebot          # wait ~30s for indexing to finish
```

- Sourcebot UI at http://localhost:3000 → sign up to create the admin user → **Settings → API keys** → generate one → paste into `.env` as `SOURCEBOT_API_KEY`.
- Serena MCP-over-SSE at http://localhost:9121/sse — no auth, app talks directly.

Both containers bind-mount `REPOS_ROOT`. Sourcebot reads only; Serena needs
write access to put its `.serena/` config dir at the project root (it does
**not** modify your repos).

### 4. Run

```bash
# Raw-text ticket (no Jira needed)
ticket-recon --ticket-text-file examples/sample_ticket.txt

# Real Jira ticket (uses JIRA_* creds)
ticket-recon --ticket-key SCRUM-17

# Real ticket + post the decomposition back as a Jira comment
ticket-recon --ticket-key SCRUM-17 --post-to-jira
```

Output:
```
✓ decomposition written to: outputs/20260510T221120Z-SCRUM-17.md
  enriched intent: feature (confidence: high)
  affected repos: traceability, frontend
  subtasks: 2
  timing: total 27.6s (enrich 3.0s · retrieve 4.5s · decompose 19.2s)
  retrieval: 30 snippets, 12,489 chars (ripgrep=25, sourcebot=5)
  cost: $0.0331
  posted Jira comment id: 10111
```

---

## API

```bash
uvicorn ticket_recon.api:app --reload --port 8765
```

```bash
curl -sX POST http://127.0.0.1:8765/decompose \
  -H 'content-type: application/json' \
  -d '{"ticket_key":"SCRUM-17","post_to_jira":true}' | jq '.markdown_path, .jira_comment_id, .metrics.total_cost_usd'
```

The response includes the structured `decomposition`, the rendered `markdown`,
`markdown_path`, `jira_comment_id` (if posted), and the run's `metrics`.

OpenAPI docs at http://127.0.0.1:8765/docs.

---

## Metrics & cost

Every run appends one JSON line to `outputs/metrics/runs.jsonl`:

```json
{
  "run_id": "20260510T221053Z-c65d4",
  "ticket_key": "SCRUM-17",
  "total_seconds": 27.567,
  "enrich":     {"seconds": 2.96, "model": "gemini-2.5-flash", "input_tokens": 845,  "output_tokens": 312, "cost_usd": 0.001033},
  "retrieval":  {"seconds": 4.48, "total_snippets": 30, "total_chars": 12489,
                  "by_source": {"ripgrep": 25, "sourcebot": 5},
                  "by_repo": {"traceability": 12, "frontend": 18}},
  "decompose":  {"seconds": 19.19, "model": "gemini-2.5-pro", "input_tokens": 4998, "output_tokens": 2803, "cost_usd": 0.032277},
  "total_cost_usd": 0.033310,
  "posted_to_jira": true,
  "subtask_count": 2,
  "affected_repos": ["traceability", "frontend"]
}
```

Quick summary across all runs:

```bash
ticket-recon-analyze
```

```
run_id                  ticket    subs  tot_s  enr_s  ret_s  dec_s in_tok out_tok    cost jira
20260510T221053Z-c65d4  SCRUM-17     2  27.57   2.96   4.48  19.19   5843    3115 $0.0331 ✓
runs: 1   total cost: $0.0331   retrieval mix: ripgrep=25, sourcebot=5
latency p50/p95: 27.6s / 27.6s
```

---

## Layout

```
src/ticket_recon/
├── api.py                 FastAPI: POST /decompose
├── cli.py                 ticket-recon CLI
├── analyze.py             ticket-recon-analyze CLI (metrics summary)
├── config.py              pydantic-settings — all env vars
├── models.py              Ticket, EnrichedQuery, Snippet, Decomposition, Subtask, …
├── jira.py                fetch ticket via REST; post comment in ADF
├── markdown_to_adf.py     minimal markdown → ADF converter for Jira comments
├── enrich.py              Pydantic AI agent, Flash, structured EnrichedQuery output
├── decompose.py           Pydantic AI agent, Pro, structured Decomposition output, context trimming
├── output.py              markdown rendering + GitLab permalink injection
├── metrics.py             RunMetrics model, JSONL writer, cost estimation
├── pipeline.py            ties stages together, records timings, posts to Jira, writes metrics
└── retrievers/
    ├── base.py            Retriever ABC
    ├── ripgrep.py
    ├── sourcebot.py       HTTP POST /api/search, repo-anchored, post-filtered
    └── serena.py          MCP-over-SSE, calls activate_project + search_for_pattern
config/sourcebot/
└── config.json            tells Sourcebot to index file:///data/repos/*
docker-compose.yml         Sourcebot + Postgres + Redis + Serena
examples/                  sample ticket for testing without Jira
outputs/                   generated markdown + metrics (gitignored)
```

---

## Configuration reference

All in `.env`. Defaults work for the Tract setup out of the box.

| key | default | notes |
|---|---|---|
| `GEMINI_API_KEY` | — | required |
| `REPOS_ROOT` | `~/tract-projects/aider-experiments` | parent folder containing repo clones |
| `REPOS` | `traceability,frontend,data-cloud-functions,data` | comma-separated repo dir names |
| `GITLAB_BASE_URL` | `https://gitlab.com` | |
| `GITLAB_PROJECTS` | configured | per-repo project paths for permalinks |
| `JIRA_BASE_URL` / `JIRA_EMAIL` / `JIRA_API_TOKEN` | empty | only needed for `ticket_key` / `ticket_url` input or `post_to_jira` |
| `SOURCEBOT_URL` | `http://localhost:3000` | |
| `SOURCEBOT_API_KEY` | empty | generate in Sourcebot UI |
| `SERENA_URL` | `http://localhost:9121/sse` | |
| `ENRICH_MODEL` | `gemini-2.5-flash` | any Pydantic AI–supported model |
| `DECOMPOSE_MODEL` | `gemini-2.5-pro` | same |
| `RETRIEVAL_MAX_HITS` | 40 | per repo, after dedupe |
| `RETRIEVAL_SNIPPET_LINES` | 8 | lines of context per hit |
| `DECOMPOSE_MAX_CONTEXT_CHARS` | 80,000 | char budget for snippets sent to Pro |

---

## Switching models

Models are Pydantic AI–native — no LangChain, no LiteLLM, no extra abstraction.
To swap a provider, edit two lines (`enrich.py`, `decompose.py`):

```python
# Anthropic
from pydantic_ai.models.anthropic import AnthropicModel
model = AnthropicModel("claude-sonnet-4-5")

# OpenAI
from pydantic_ai.models.openai import OpenAIModel
model = OpenAIModel("gpt-4o")
```

You still need the corresponding `*_API_KEY` env var. Pricing in `metrics.py`
will need an entry for cost tracking on new models.

---

## Operational notes

- **Sourcebot indexes committed git refs only.** Uncommitted working-tree
  changes are invisible to it. If your local clone is dirty, ripgrep will see
  the changes but Sourcebot won't.
- **Serena spins down between SSE connections.** That's by design — each
  request gets a fresh session. The `activate_project` tool is called per
  retrieval, so this is invisible to callers.
- **`post_to_jira` failures don't fail the run.** If the comment can't be
  posted (network/auth), the markdown is still written and metrics still
  recorded; the failure is logged and `jira_comment_id` is `null` in the
  response.
- **Sensitive data**: snippets sent to Pro can include real source code.
  Don't point this at private repos unless you've vetted your model provider's
  data retention policy.

---

## Known limitations (v0.1.0)

- No request-level auth on the API endpoint — anyone with the URL can spend
  your Gemini quota
- Sync request/response (no job queue) — Pro stage can take 20-30s, fragile
  behind aggressive proxy timeouts
- No idempotency — same ticket twice = two full Pro calls
- No evaluation harness — no ground-truth ticket-to-decomposition pairs to
  detect prompt regressions
- Prompt versions aren't tracked in metrics, so A/B comparison of prompt
  edits across runs requires git archaeology

The full priority-ranked list of "next things to do" is in commit history.

---

## License

TBD.
