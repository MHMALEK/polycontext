# Handoff prompt — paste into Claude / Cursor / GPT-5 to continue this project

Copy everything between the `---` lines below into your other AI tool.

---

You're picking up work on **ticket-recon**, a Python service that turns
Jira tickets into AI-friendly tech decompositions with multi-repo
context and clickable GitLab permalinks. The project lives at
`/Users/mohammadhosseinmalek/tract-projects/ticket-recon` and runs against
four real repos cloned at `/Users/mohammadhosseinmalek/tract-projects/aider-experiments/`:
`traceability`, `frontend`, `data-cloud-functions`, `data`.

## Architecture (what already works)

The pipeline:
```
Jira ticket
    │
    ▼  enrich (Gemini 2.5 Flash + Pydantic AI, temp=0, structured output)
    │  Outputs: entities, code keywords, suspected repos, confidence
    │
    ▼  gather_context (cheap mode) OR deep_decompose (agentic mode)
    │
    │   cheap mode retrieval (single-pass, parallel):
    │     - anchor extraction (regex file tokens from ticket) → read whole files
    │     - 3-hop transitive imports via prebuilt import_index
    │       (Python ast + tree-sitter for TS/TSX/JS)
    │     - ripgrep (local clones, fast keyword)
    │     - Sourcebot HTTP /api/search (indexed multi-repo, zoekt-backed)
    │     - Serena MCP /sse (LSP-backed semantic, activate_project per session)
    │   Test files down-ranked. Anchors score 10.0, followed imports 8.0.
    │
    │   deep mode (agentic loop, capped at 20 turns):
    │     Pydantic AI agent with tools: read_file, grep, list_directory,
    │     find_symbol, get_file_overview, find_referencing_symbols,
    │     what_does_this_import, who_imports.
    │     Mode is auto-selected on "migrate / refactor / port to" cues or
    │     ≥3 suspected repos or low enrichment confidence.
    │
    ▼  decompose (Gemini 2.5 Pro + Pydantic AI, structured Decomposition output)
    │   Strict grounding prompt: no inventing files/symbols, mark
    │   unknowns as "requires investigation".
    │
    ▼  contradiction-check (Flash, ~$0.001) — flags subtasks that
    │   contradict ticket criteria. Findings appended to risks section.
    │
    ▼  attach_gitlab_links + render_markdown + write_markdown
    │  + optionally post_comment to Jira (ADF format)
    │
    ▼  metrics row appended to outputs/metrics/runs.jsonl
       (latency per stage, tokens, cost, retrieval breakdown by source)
```

Two CLI entry points: `ticket-recon` (decomposition) and `ticket-recon-analyze`
(metrics summary). One more entry point: `ticket-recon --ask "..."` for
free-form code Q&A — tries Sourcebot's `/api/ask` first, falls back to a local
agentic loop with the same tool inventory but `output_type=Answer`.

Models swappable via Pydantic AI — `GeminiModel`/`AnthropicModel`/`OpenAIModel`
are one-line drops in `enrich.py` and `decompose.py`.

## What's been measured

Real runs against `SCRUM-17` (single-repo refactor) and `SCRUM-18` (hard
multi-repo Airflow→CF migration) on the four cloned repos. Manual baseline
written by Claude Opus 4.7 in Claude Code. Comparison files in `outputs/`.

| | cheap mode | deep mode | manual (Opus, Claude Code) |
|---|---|---|---|
| SCRUM-17 wall | 43s | — | n/a |
| SCRUM-17 cost | $0.05 | — | n/a |
| SCRUM-18 wall | 76s | 122s | ~3 min |
| SCRUM-18 cost | $0.09 | $0.30 | ~$0.50–$1.00 |
| SCRUM-18 contradictions caught | 1 ("no new deps") | 2 ("no new deps" + wrong DAG name) | hand-spotted ~5 |

Deep mode currently produces output that's ~70-80% of a careful manual
exploration. The bottleneck is no longer retrieval (anchors + 3-hop imports
+ Sourcebot + Serena gets the right files); it's the model under-using its
own tools (e.g. on the Farm Name Q&A, `find_symbol` was called 1/8 times
despite a strict prompt rule). One factual line-number error in the most
recent run (`node_name_validator` claimed at `validator.py:655-657`,
actually at 763).

## What's already on disk

- `src/ticket_recon/` — 18 modules, ~1700 LOC
- `docker-compose.yml` — Sourcebot + Postgres + Redis + Serena. Sourcebot
  config at `config/sourcebot/config.json` (also registers Gemini models
  for the `/api/chat` feature — but see "stuck on" below).
- `outputs/.import_index/<repo>-<sha8>.json` — cached per-repo import graph
- `outputs/metrics/runs.jsonl` — one JSON row per pipeline run
- `outputs/COMPARISON-*.md` — manual-vs-app comparisons for two tickets
- `docs/improvement-roadmap.md` — surveyed tools and what would help most
- `.env` — has GEMINI_API_KEY, JIRA_*, SOURCEBOT_*, Postgres password

## What we got stuck on

**Sourcebot's `ask_codebase` (their internal AI Q&A) doesn't work on our
deployment.** The full debugging trail so you don't repeat it:

| endpoint | status on OSS 4.17.1 | notes |
|---|---|---|
| `POST /api/ask` | 404 | what sentinel uses; not in our version |
| `POST /api/chat` | 400 schema → 404 inside | needs `{id, messages, languageModel, selectedSearchScopes}`. Even fully-formed: 404 |
| `POST /api/ask_codebase` | 404 | doesn't exist at REST layer |
| `POST /api/mcp` (Streamable HTTP MCP) | **works** | exposes 11 tools incl. `ask_codebase` |
| MCP tool `list_language_models` | **works** | returns our two configured Gemini models |
| MCP tool `ask_codebase` | **fails** | returns `"Language model X is not configured"` |
| MCP tools `grep`, `read_file`, `find_symbol_definitions`, etc. | **work** | useful but duplicate of what we have |

We set:
- Models in `config/sourcebot/config.json` under `"models"` array
- `GOOGLE_GENERATIVE_AI_API_KEY` on the Sourcebot container env
- Tried both `{"token": {"env": "..."}}` and omitting `token` (schema says it defaults to env var)
- Hard container recreate
- Sentinel works because it runs against a different Sourcebot (likely hosted or EE)

**Workaround that's currently live**: `ticket-recon --ask` tries Sourcebot MCP
first (`ask_codebase`), captures the "not configured" error, falls back to
`local_ask.py` (same Pydantic AI agentic loop as `deep_decompose` but with
`output_type=Answer`). End-to-end works. ~$0.04 for a Q&A on our setup.

**If you crack the Sourcebot config**, no client code needs to change —
`ask.py` is already wired to call `ask_codebase` via `/api/mcp`. Likely
next debugging step: read Sourcebot's source for the `ask_codebase`
handler to see what config it actually checks (vs what `list_language_models`
checks). The repo is at `gitlab.com/sourcebot-dev/sourcebot` — grep for
`'is not configured'` to find the check.

## What I'd ask you to do next, in priority order

These are the highest-leverage items from `docs/improvement-roadmap.md`,
re-stated here so you can pick them up directly:

### Tier 1 (do these first)

1. **Build a 10-ticket eval set with a rough scoring rubric.** Without
   measurable baselines, every prompt or retrieval change is guesswork.
   Use the [Inspect framework](https://inspect.ai-safety-institute.org.uk/)
   or roll your own. Score:
   - `affected_repos_correct` (set match)
   - `key_files_cited` (recall over expected files)
   - `no_invented_files` (0 hallucinations)
   - `no_contradictions_with_ticket`
   - `subtask_count_reasonable` (3-7 for medium, 1-3 for small)

   Run eval before & after every change. Output a single composite score
   per run and a per-axis breakdown. Add it to CI.

2. **Add a verifier loop.** After `decompose` produces a Decomposition:
   - For every cited file path, verify it exists in the configured repos.
     If not, mark the subtask `requires investigation` and remove the path.
   - For every symbol referenced in subtask titles/descriptions, call
     `find_symbol` via Serena. If not found, demote to `requires investigation`.
   - For every acceptance criterion, run the contradiction-check stage we
     already have but generalize it (not just ticket-criteria contradictions
     but also internal contradictions across subtasks).

   This is half a day and would eliminate the line-number-wrong class of
   error we still see.

3. **Crack Sourcebot's `/api/chat` (or `/api/ask`).** Steps to try:
   - Read https://docs.sourcebot.dev/docs/configuration/language-model-providers
     fully — there might be a feature flag we missed.
   - Try posting from inside the Sourcebot container (auth might be cookie-based
     in the OSS UI but bearer in API; maybe we need a session cookie).
   - Check if Sourcebot has an Agents config section we haven't enabled.
   - Look at sentinel's `.env.example` for any Sourcebot-side config we missed
     (`SOURCEBOT_LANGUAGE_MODEL_PROVIDER` etc. — those look client-side, but
     maybe they have server-side counterparts).
   - If still stuck, decide: pay for Sourcebot hosted, or commit to the local
     agentic path and remove the half-built Sourcebot adapter.

### Tier 2 (after Tier 1's eval is in place)

4. **Code embeddings + vector retrieval** as a fourth retriever. Voyage's
   `voyage-code-3` model + LanceDB. Build the index once per repo HEAD,
   query alongside existing retrievers, dedupe with score normalization.

5. **Cross-encoder reranker** between retrieval and decompose. Cohere
   Rerank or `BAAI/bge-reranker-v2-m3` (OSS, local). Pull 100 candidates,
   send top 20 to Pro.

6. **Routing by complexity.** A pre-stage that classifies the ticket as
   `simple|medium|hard|critical` and picks the model + mode:
   - simple → Flash for decompose too (~$0.005)
   - medium → Pro single-shot (current cheap, ~$0.07)
   - hard → Pro agentic loop (current deep, ~$0.30)
   - critical → Opus or Sonnet 4.6 agentic (~$1-2)

   This reduces average cost without sacrificing quality on hard tickets.

7. **LangFuse for tracing.** OSS, self-hostable. Pydantic AI has OTel
   support. Wire up so every stage's tokens / latency / cost is visible
   in a UI rather than just JSONL.

### Tier 3 (advanced)

8. **Tool-use enforcement** in deep mode. When the model proposes a subtask
   referencing a symbol it didn't actually call `find_symbol` on, reject
   the subtask programmatically and retry the decompose stage with the
   ungrounded symbols flagged. Deterministic guardrail, not a hopeful prompt.

9. **Aider-style repomap** as an alternative retriever — purely
   tree-sitter-driven symbol map per repo, no LLM. Read their implementation
   at https://github.com/Aider-AI/aider/blob/main/aider/repomap.py.

10. **Contextual retrieval (Anthropic-style)** on the symbol index — prepend
    a one-sentence summary of each chunk's context before embedding. Cuts
    retrieval failures by ~50% per Anthropic's data.

## How to verify your changes

```bash
cd /Users/mohammadhosseinmalek/tract-projects/ticket-recon
# Already has .venv. If missing:  uv venv && uv pip install -e .

# Make sure docker stack is up:
docker compose ps        # all 4 healthy: postgres, redis, sourcebot, serena
docker compose up -d     # if anything is down

# Smoke test the cheap path:
.venv/bin/ticket-recon --ticket-key SCRUM-17 --mode cheap

# Smoke test the deep path:
.venv/bin/ticket-recon --ticket-key SCRUM-18 --mode deep

# Smoke test --ask (falls back to local agent since Sourcebot AI is broken):
.venv/bin/ticket-recon --ask "What special characters are not allowed in Farm Name?"

# Compare runs:
.venv/bin/ticket-recon-analyze

# Inspect a generated artifact:
ls -t outputs/*.md | head -1 | xargs cat
```

## Important constraints / things not to break

1. **Don't rewrite the pipeline.** It works. Add steps, replace single stages,
   but keep the `enrich → gather_context → decompose → contradiction → render`
   shape.
2. **Keep Pydantic AI as the agent framework.** Don't switch to LangChain or
   LangGraph — we evaluated those and chose Pydantic AI for typed outputs
   and clean tool registration. Adding LangGraph for "fancier orchestration"
   buys nothing on a 4-stage pipeline.
3. **Keep the typed outputs.** `Decomposition`, `EnrichedQuery`, `Answer`,
   `Citation` are the contracts. Downstream agents will eventually consume
   them via the JSON API. Don't loosen the schemas.
4. **Never write a real API key to the repo.** `.env` is gitignored. The
   GEMINI_API_KEY, JIRA_API_TOKEN, and SOURCEBOT_API_KEY currently in `.env`
   were pasted in chat by the user — they should be rotated, but that's the
   user's call.
5. **Don't auto-post to Jira without `post_to_jira=True`.** It's gated for
   a reason — accidentally commenting on real tickets is bad.
6. **Don't fine-tune anything.** No eval set, no training data, way too early.

## Quick reference: where things live

| concern | file |
|---|---|
| FastAPI app + routes | `src/ticket_recon/api.py` |
| CLI | `src/ticket_recon/cli.py` |
| Config (env, defaults) | `src/ticket_recon/config.py` |
| Typed models | `src/ticket_recon/models.py` |
| Pipeline orchestration | `src/ticket_recon/pipeline.py` |
| Jira fetch + comment | `src/ticket_recon/jira.py` |
| Enrich (Flash) | `src/ticket_recon/enrich.py` |
| Decompose (Pro, cheap) | `src/ticket_recon/decompose.py` |
| Decompose (Pro, agentic) | `src/ticket_recon/deep_decompose.py` |
| Contradiction check | `src/ticket_recon/contradiction.py` |
| Local Q&A agent | `src/ticket_recon/local_ask.py` |
| Sourcebot /api/ask adapter | `src/ticket_recon/ask.py` |
| Markdown render + GitLab links | `src/ticket_recon/output.py` |
| Markdown → ADF for Jira | `src/ticket_recon/markdown_to_adf.py` |
| Metrics + cost | `src/ticket_recon/metrics.py` |
| Run summary CLI | `src/ticket_recon/analyze.py` |
| Per-repo import index | `src/ticket_recon/import_index.py` |
| Per-language import parsing | `src/ticket_recon/retrievers/import_follow.py` |
| Anchor retrieval | `src/ticket_recon/retrievers/anchors.py` |
| Ripgrep retriever | `src/ticket_recon/retrievers/ripgrep.py` |
| Sourcebot retriever | `src/ticket_recon/retrievers/sourcebot.py` |
| Serena retriever | `src/ticket_recon/retrievers/serena.py` |
| Improvement roadmap (read this) | `docs/improvement-roadmap.md` |

---

End of prompt. Reply with which Tier 1 item you want to start with.
