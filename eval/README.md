# Adapter bake-off

A standalone harness for comparing the agent adapters side-by-side. Reads
the same `/v1/adapters/*` endpoints the UI does, writes results to
`eval/outputs/`, never touches the runtime app code.

## Agent node

The harness calls the same **`/v1/adapters/*`** routes as the web UI. The
**`cursor`**, **`cline_sdk`**, **`claude_code`**, **`gemini`**, and **`sourcebot`** adapters are
thin HTTP clients to **`services/agent-node`** (Cursor SDK, Cline SDK, Claude Agent SDK,
[`@google/genai`](https://www.npmjs.com/package/@google/genai), Sourcebot
blocking chat). So for those adapters you need agent-node **up and reachable
at `AGENT_NODE_URL`** before you run a bake-off — whether you use the default
in-process FastAPI client or `--base-url` against a live server.

**Keep this harness in Python:** it already speaks the canonical FastAPI adapter
routes. **`services/agent-node`** is where the Cursor / Claude / Gemini / Sourcebot SDKs live;
running the evaluator in Node would duplicate orchestration without adding much.
(`agent-node` also exposes **`openai_agents`** — not yet wired behind a Python registry
adapter, so omit it from **`ADAPTERS=...`** until that bridge exists.)

```bash
make adapters
# or: cd services/agent-node && npm install && npm run dev
```

Set **`AGENT_NODE_URL`**, **`CURSOR_API_KEY`** (for `cursor`), **`ANTHROPIC_API_KEY`**
(for `claude_code`), and at least one of **`GEMINI_API_KEY`** / **`ANTHROPIC_API_KEY`** /
**`OPENAI_API_KEY`** (for `cline_sdk` and **`gemini`**). For **`sourcebot`**, set
**`SOURCEBOT_URL`** and **`SOURCEBOT_API_KEY`**. Cline’s optional `find_code` tool inside Node also
uses **`SOURCEBOT_*`** from the agent-node environment.

**Note:** “In-process” mode only means the **FastAPI app** runs inside the
eval process; adapter calls still perform **real HTTP** from Python to
agent-node.

## Layout

```
eval/
├── README.md                   ← this file
├── questions.toml              ← ask cases (existed before the harness)
├── cases/
│   ├── decompose/*.yaml        ← one decompose case per file
│   └── implement/*.yaml        ← one implement case per file
├── outputs/eval-<timestamp>/   ← results from each run (gitignored)
│   ├── manifest.json
│   ├── runs/<case-id>/<adapter>.json
│   ├── report.md
│   └── summary.json
└── bakeoff/                    ← Python package: case loaders, runner, scorer, reporter, CLI
```

The harness uses FastAPI's `TestClient` in-process by default, so you don't
have to start `make dev` first. Pass `--base-url` to hit a running server.

## Report output (comparison)

After each **`run`** the harness writes:

- **`runs/<case>/<adapter>.json`** — full `RunRecord` (answer Markdown, citations,
  `metrics` from adapters, scorer breakdown, **`run_id`**, wall-clock **ms**).
- **`summary.json`** — flat rows keyed by case and adapter (`tokens_*`, **`tool_calls`**,
  **`cost_usd`**, derived **`answer_chars` / citation_count** for **`ask`**), plus leaderboard.
- **`report.md`** — leaderboard with success rate, weighted rubric **`avg_score`**, wall-clock
  latency totals, **`ask`**-only averages (answer length, citation count, tokens, tool calls when
  the adapter emitted them).

For **`ask`**, each case includes a **metrics comparison table** (all adapters, same question)
followed by **expandable `<details>` blocks per adapter** listing structured citations plus the **full
Markdown answer**. Use Beyond Compare / `diff` across two `eval-eval-*/report.md` files for A/B drift.

Example — only the master-data question already in **`questions.toml`**:

```bash
make eval-run ADAPTERS=cursor,cline_sdk,claude_code,gemini,sourcebot JOB=ask --IDS q7-master-data-upload-e2e
```

For **`--verbose`**, invoke the CLI directly (Make does not forward extra flags):

```bash
uv run python -m eval.bakeoff.cli -v run --adapters cursor,cline_sdk --job ask --IDS q7-master-data-upload-e2e
```

## Quick start

```bash
# 1. See which adapters are installed and healthy.
make eval-adapters

# 2. See the case set.
make eval-cases JOB=ask
make eval-cases JOB=decompose
make eval-cases JOB=implement

# 3. Run the bake-off against a couple of adapters.
make eval-run ADAPTERS=cursor,cline_sdk JOB=ask

# 4. Open the report.
open eval/outputs/eval-<timestamp>/report.md
```

## Adding cases

### Ask (Q&A)

Append a `[[questions]]` block to `questions.toml`:

```toml
[[questions]]
id = "q-my-question"
tags = ["frontend", "validation"]
text = "What does the foo validator do exactly?"

[questions.expected]      # optional — leaving this off uses default checks
min_chars      = 120
min_citations  = 1
must_mention   = ["regex"]
should_mention = ["validator"]
```

### Decompose

Drop a YAML file in `cases/decompose/`:

```yaml
id: my-decompose
tags: [cross-repo, validation]
description: One-line note for humans reading the case set.

input:
  ticket_text: |
    Title: ...
    Body...
  mode: auto             # cheap | deep | auto
  repos: [frontend]      # optional — limit to a subset

expected:
  min_subtasks: 2
  max_subtasks: 10
  must_touch_repos: [frontend]
  must_mention_files: [validate]
```

### Implement

Drop a YAML file in `cases/implement/`:

```yaml
id: my-implement
tags: [trivial, single-file]

input:
  repo: traceability                  # key in settings.gitlab_projects
  free_text: |                        # OR provide `subtask:` with a full Subtask object
    Add a docstring to the first function missing one.
  base_branch: main
  draft: true

expected:
  min_files_changed: 1
  must_touch_paths: [".py"]
  must_not_touch_paths: ["package-lock.json", "node_modules"]
  require_mr: false                   # set to true once GITLAB_TOKEN is configured
```

## Scoring

Every check has a weight; the overall score is the weighted pass-rate
(0..1). The leaderboard sorts by **`avg_score`**, then **longer average `ask` answers** (more
complete responses), then **lower average wall latency**.

You can read the exact rubrics in [`bakeoff/scorer.py`](bakeoff/scorer.py).
Run `make eval-run ... -- -v` for the full per-check breakdown in the
markdown report's `<details>` blocks.

## Tips

- **No network**: the in-process TestClient backend lets the harness run
  offline when adapters only touch local disk and a local model (otherwise the
  agent-node and provider APIs need connectivity).
- **Real network**: set `--base-url http://localhost:${API_PORT:-18000}` to run against
  a `make dev` instance — necessary for adapters that need Docker sidecars.
- **Filters**: use `--ids id1 id2` to re-run just a handful of cases, or
  `--tags validation cross-repo` to scope by topic.
- **Timeouts**: per-case timeout is 900s by default; lower with `--timeout`
  if you want fail-fast behavior while iterating on rubrics.
