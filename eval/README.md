# Adapter bake-off

A standalone harness for comparing the agent adapters side-by-side. Reads
the same `/v1/adapters/*` endpoints the UI does, writes results to
`eval/outputs/`, never touches the runtime app code.

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

## Quick start

```bash
# 1. See which adapters are installed and healthy.
make eval-adapters

# 2. See the case set.
make eval-cases JOB=ask
make eval-cases JOB=decompose
make eval-cases JOB=implement

# 3. Run the bake-off against a couple of adapters.
make eval-run ADAPTERS=cline_sdk,opencode JOB=ask

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
(0..1). Sorting in the report is by `avg_score`, ties broken by latency.

You can read the exact rubrics in [`bakeoff/scorer.py`](bakeoff/scorer.py).
Run `make eval-run ... -- -v` for the full per-check breakdown in the
markdown report's `<details>` blocks.

## Installing adapter tools

```bash
# CLI binaries
brew install opencode                  # or: curl -fsSL https://opencode.ai/install | bash
# cursor-agent ships with the Cursor IDE; run `cursor-agent login` once.

# Cline SDK sidecar (used by cline_sdk + cline_sdk_grounded)
make adapters                          # starts the cline_sdk_bridge container
# Then add to .env:
#   CLINE_SDK_BRIDGE_URL=http://localhost:3040
```

Then `make eval-adapters` to confirm health turned green.

## Tips

- **No network**: the in-process TestClient backend lets the harness run
  fully offline if your adapters do (e.g. OpenCode pointed at a local Ollama).
- **Real network**: set `--base-url http://localhost:8000` to run against
  a `make dev` instance — necessary for adapters that need Docker sidecars.
- **Filters**: use `--ids id1 id2` to re-run just a handful of cases, or
  `--tags validation cross-repo` to scope by topic.
- **Timeouts**: per-case timeout is 900s by default; lower with `--timeout`
  if you want fail-fast behavior while iterating on rubrics.
