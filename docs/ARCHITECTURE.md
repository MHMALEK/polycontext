# Architecture

Single-source-of-truth for what's running, how things connect, and the vocabulary you need to read the rest of the codebase. Read this before [`README.md`](../README.md) for the wiring; read README after for the per-feature changelog.

## TL;DR

```
                  ┌────────────┐                ┌──────────────────────────────┐
   User browser   │   web/     │  HTTP+JSON     │       FastAPI (Python)       │
   localhost      │  Vite/React│ ──────────────▶│   src/tech_decomposition/    │
   :15173         │            │                │   api.py + adapters + core   │
                  └────────────┘                └─────────────┬────────────────┘
                                                              │
                  ┌──────────────────────────────┐             │
                  │      agent-node (Node TS)    │             │
                  │   services/agent-node/       │ ◀───────────┤ HTTP
                  │   Fastify, runs SDK clients  │             │
                  └─┬──────┬──────┬───────┬──────┘             │
                    │      │      │       │                    │
              cursor│gemini│cline │opencode│ claude_code       │
              SDK   │SDK   │SDK   │SDK     │ SDK               │
                    │      │      │        │                   │
                    │      │      │        │                   │
                    │      │      │        ▼                   │
                    │      │      │  ┌────────────┐            │
                    │      │      │  │ opencode   │            │
                    │      │      │  │ serve :4096│            │  (host, separate process)
                    │      │      │  │ (or spawn) │            │
                    │      │      │  └────────────┘            │
                    │                                          │
                  Provider APIs                                │
                  (Anthropic, OpenAI, OpenRouter, Google)  ◀───┤
                                                              │
                  ┌──────────────────────────────┐             │
                  │  Sourcebot (Docker)          │ ◀───────────┤ /api/search (zoekt-style)
                  │  index of git repos          │             │
                  └──────────────────────────────┘             │
                                                              │
                  ┌──────────────────────────────┐             │
                  │  Serena (uvx)                │ ◀───────────┤ MCP JSON-RPC
                  │  pyright + tree-sitter LSP   │             │
                  └──────────────────────────────┘             │
                                                              │
                  ┌──────────────────────────────┐             │
                  │  Repos on disk               │ ◀───────────┤ file reads (grep/glob/read)
                  │  $REPOS_ROOT/<repo>/...      │             │
                  └──────────────────────────────┘             │
                                                              │
                  ┌──────────────────────────────┐             │
                  │  SQLite runstore             │ ◀───────────┤ ./outputs/runstore.db
                  │  every API call persisted    │             │
                  └──────────────────────────────┘             ▼
```

Everything is HTTP between services. Nothing in this project uses message queues, websockets (except Vite HMR), or shared memory.

---

## Vocabulary

These four words mean very specific things. Use them precisely; the rest of the codebase does.

### Adapter

A Python class that knows how to drive one external code-AI service end-to-end. Implements `ask` and/or `decompose`. Lives under [`src/tech_decomposition/adapters/`](../src/tech_decomposition/adapters/). Registered by name in [`registry.py`](../src/tech_decomposition/adapters/registry.py).

There are 8 adapters today:

| Adapter | Drives | Capabilities |
|---|---|---|
| `cursor` | Cursor SDK via agent-node | ask, decompose |
| `gemini` | Google `@google/genai` SDK via agent-node | ask, decompose |
| `claude_code` | Claude Agent SDK via agent-node | ask, decompose |
| `cline_sdk` | Cline SDK via agent-node | ask, decompose |
| `opencode` | OpenCode SDK + standalone `opencode serve` | ask, decompose |
| `openai_agents` | OpenAI Agents SDK via agent-node | ask, decompose |
| `sourcebot` | Sourcebot `/api/chat/blocking` (in-process Python) | ask, decompose |
| `pipeline` | Our custom tiered RAG flow (in-process Python) | ask, decompose |

### SDK

The vendor library that an adapter calls. OpenCode's `@opencode-ai/sdk`, Anthropic's `@anthropic-ai/claude-agent-sdk`, etc. Most run inside `agent-node` (the Node service) because they're Node-only packages; `pipeline` and `sourcebot` are pure-Python.

In UI copy we sometimes use "SDK" interchangeably with "adapter" because users pick an adapter and what they're really doing is picking which SDK runs their query.

### Grounding

Pre-fetching relevant code snippets before the adapter sees the query. Implemented in [`src/tech_decomposition/core/grounding.py`](../src/tech_decomposition/core/grounding.py). Combines two retrieval sources running in parallel:

- **Sourcebot** — zoekt-based keyword search over indexed git repos. Returns files + chunks with line numbers.
- **Serena** — LSP-backed semantic search (find_symbol, find_referencing_symbols, search_for_pattern). Useful when the user's vocabulary doesn't lexically match the code's symbols.

The retrieved snippets are merged, deduped, optionally reranked with a cross-encoder (BGE-reranker-v2-m3), per-repo capped for diversity, window-expanded by reading the files from disk and widening each snippet by ±N lines, then formatted as a markdown block and prepended to the user's query.

The adapter sees the grounded query in place of the raw one. It can also call its own tools on top — the model decides whether the prefetched snippets are enough or it needs to explore further.

### Tools (agent tools)

The SDK-provided functions that the model can call mid-conversation: `read` a file, `grep` for a pattern, `glob` matching paths, `ls` a directory. OpenCode's standard read-side toolset is `read / grep / glob / ls / task` (`task` spawns a subagent). The agent decides per-query whether to use them.

Tools are independent of grounding: an adapter can have tools without us pre-fetching, can be pre-fetched without tools, can do both, or do neither.

---

## The 4 modes (UI-facing)

The Mode picker in the Ask form maps to two boolean flags on the wire: `grounded` and `tools_enabled`. Combining them gives 4 distinct behaviors:

| Mode | grounded | tools_enabled | What happens |
|---|---|---|---|
| **Hybrid** (default) | `true` | `true` | API prefetches Sourcebot+Serena snippets, prepends to query. Adapter receives the grounded query AND has tools available. Model decides whether to call tools on top. |
| **Grounded only** | `true` | `false` | API prefetches snippets. Adapter receives the grounded query but tools are masked off (`tools: {read:false, grep:false, ...}` to OpenCode). Model answers single-shot from the snippets. Fastest. |
| **Agent only** | `false` | `true` | No prefetch. Adapter receives the raw query and navigates the repo entirely via tools. Slowest, most autonomous. |
| **Single-shot** | `false` | `false` | Raw query goes straight to the model. No retrieval, no exploration. Useful for general questions the model can answer from its training. |

Why this matters: in our eval (May 2026), pipeline+Flash with `Hybrid` scored gold_acc 0.57; OpenCode+DeepSeek with `Hybrid` scored 0.82. The model uses tools when grounding is insufficient (the agentic loop fires) and answers from snippets when grounding is sufficient. Two cases out of 12 in our eval fired the loop; the other 10 answered from prefetched context.

**Grounded only** is best when you want the cheapest possible inference and you trust the retrieval. **Agent only** is best when you have a sparse / uncurated codebase and don't trust the retrieval. **Hybrid** is the safe default.

---

## The `pipeline` adapter (special-case)

Most adapters delegate to an external SDK. The `pipeline` adapter is different — it's a tiered RAG flow that runs entirely in Python and tries to keep the hot path on the cheapest possible single-shot synthesis:

```
   query + tags
        │
        ▼
   1. Router (heuristic regex)
      → assigns TaskTier (SIMPLE / ENUMERATION / TRACE / COMPLEX)
      → produces prefetch_top_k, retrieval_mode (default | broad), tag-aware boost terms
        │
        ▼
   2. Grounded retrieval (core/grounding.py)
      → Sourcebot keyword search (with zoekt noise filters)
      → Serena search_for_pattern in parallel
      → Serena find_symbol on symbol-shaped LLM-suggested terms
      → merge + per-repo cap + optional cross-encoder rerank + window expansion
        │
        ▼
   3. Coverage scoring (core/coverage.py)
      → weighted score (snippet count, char total, repo diversity, serena bonus)
      → tier-specific sufficiency threshold
        │
        ▼
   4. Synthesis (core/synthesize.py)
      → pydantic-ai single-shot call to settings.pipeline_synthesis_model
        (routes through llm_registry — Gemini, OpenAI, Anthropic, OpenRouter,
         custom OpenAI-compatible all supported)
      → cheap Flash-class model by default
      → escalate to Pro when tags ∈ {validation, master-data} or coverage low + cross-repo
        │
        ▼
   5. Verifier (core/verifier.py)   [decompose only]
      → walk decomposition.subtasks
      → for each cited file: does the path exist under REPOS_ROOT?
      → missing files become risks
        │
        ▼
   6. Conditional agent fallback
      → only fires when route.allow_agent_fallback AND
         coverage insufficient AND synthesis text matches "not enough info"
      → invokes settings.pipeline_fallback_adapter (default: gemini agentic loop)
        │
        ▼
   final answer + structured Decomposition + telemetry
```

The pipeline philosophy is **"give the model enough context, then a single cheap call"**. The agent fallback is the safety net for the cases where prefetch genuinely missed.

Code lives in [`src/tech_decomposition/adapters/_pipeline.py`](../src/tech_decomposition/adapters/_pipeline.py). The router, coverage, reranker, synthesizer, and verifier are each in their own module under [`src/tech_decomposition/core/`](../src/tech_decomposition/core/).

---

## The `opencode` adapter (focus)

OpenCode is the agentic adapter we've invested in most. It runs the same SDK both Cursor and Claude Code use internally — a multi-step prompt loop where the model can read files, grep, glob, ls, and spawn subagents (`task`).

### How it's wired today

```
                                          ┌───────────────────┐
   Python opencode adapter                │  agent-node       │
   src/.../_opencode_sdk.py               │  services/...     │
        │                                 │                   │
        │  POST /adapters/opencode/run    │                   │
        │  {prompt, model, toolsEnabled,  │                   │
        │   providerID, apiKey, ...}      │                   │
        ▼                                 ▼                   │
   ┌──────────────────────────────────────────────────────┐  │
   │  agent-node opencode handler                          │  │
   │  uses @opencode-ai/sdk/v2 client                      │  │
   │    1. client.auth.set({providerID, apiKey})           │  │
   │    2. client.session.create({agent: "build", model})  │  │
   │    3. client.session.prompt({parts, agent, tools?})   │  │
   │    4. client.session.messages() — read full trace     │  │
   │    5. count tool_use parts + extract read filepaths   │  │
   └─────────────────┬────────────────────────────────────┘  │
                     │                                        │
                     │ HTTP /v2/* per session                 │
                     ▼                                        │
   ┌──────────────────────────────────────────────────────┐  │
   │  opencode serve --port 4096 (HOST process)            │  │
   │  loads the embedded LLM agent loop                    │  │
   │  exposes read/grep/glob/ls/bash/edit/.../task tools   │  │
   └─────────────────┬────────────────────────────────────┘  │
                     │                                        │
                     │ provider call                          │
                     ▼                                        │
   ┌──────────────────────────────────────────────────────┐  │
   │  Provider API                                         │  │
   │   ▸ openrouter/deepseek/deepseek-v3.2                 │  │
   │   ▸ openrouter/qwen/qwen3-235b-a22b-2507              │  │
   │   ▸ anthropic/claude-sonnet-4-5                       │  │
   │   ▸ google/gemini-2.5-pro                             │  │
   └──────────────────────────────────────────────────────┘  │
                                                             │
   Tool execution happens inside opencode serve:             │
     read tool   → opens files under cwd (REPOS_ROOT)        │
     grep tool   → ripgrep over cwd                          │
     glob tool   → fast-glob over cwd                        │
     task tool   → spawns a subagent with its own prompt     │
```

### Key learnings (paid for in eval cycles)

1. **session.prompt's response carries ONLY the final assistant message.** The agentic loop's intermediate messages (each carrying tool calls) live in the session's message log. You must call `client.session.messages({sessionID, directory})` after prompt resolves to see the actual tool trace.
2. **Tools are server-loaded, not request-set.** `Config.tools` in the SDK type is a DISABLE filter applied on top of agent permissions. All tools default to true. Passing `tools: {read: true}` is a no-op; passing `tools: {read: false}` disables it.
3. **`opencode serve` and `opencode run` use the same backend.** The CLI is just a thin wrapper around the SDK. There's no special CLI path for tools.
4. **Pass `agent` and `model` to `session.create`, not just `session.prompt`.** Some server-side state binds to the session-level agent.
5. **Docker network proxy times out long-running requests at 302s.** Run agent-node on host (not in Docker) when calling opencode serve on the host, or install opencode binary inside the agent-node container.

### What we use, what we leave on the table

| OpenCode feature | Status |
|---|---|
| `read` tool | ✅ used |
| `grep` tool | ✅ used |
| `glob` tool | ✅ used |
| `ls` tool | ✅ used |
| `task` (subagent) | ✅ used (model decides when) |
| `bash`, `edit`, `write`, `multiedit`, `patch` | 🔒 masked in our adapter — Q&A is read-only |
| `webfetch` | 🔒 masked — privacy + cost |
| `todoread` / `todowrite` | ⚪ not yet — could help multi-step planning |
| Streaming response | ⚪ not yet — we await final, UX regression |
| Memory (`.serena/memories/` style) | ⚪ not yet — would persist project glossary across sessions |
| Custom agents (beyond `build`) | ⚪ not yet — could define `qa-build` agent in `opencode.json` |
| Permission ruleset (deny/ask/allow) | ⚪ basic; could harden per-tool |

The ones marked ⚪ are concrete future wins.

---

## Models, providers, and credentials

There are three layers of model resolution in this project:

### 1. Configured defaults (`.env`)

```
PIPELINE_SYNTHESIS_MODEL=gemini:gemini-2.5-flash
OPENCODE_SDK_MODEL=openrouter/deepseek/deepseek-v3.2
GEMINI_SDK_MODEL=gemini-2.5-pro
CURSOR_SDK_MODEL=composer-2
# ...etc
```

Used when no per-request override is sent.

### 2. Per-request override (UI + API)

The Ask form's **Model** picker writes a per-request value into `AdapterAskInput.model`. Adapter-specific format:

- `opencode`: `provider/model` slash form (e.g. `openrouter/deepseek/deepseek-v3.2`, `anthropic/claude-sonnet-4-20250514`).
- `pipeline`: `provider:model` colon form (e.g. `openrouter:deepseek/deepseek-v3.2`, `gemini:gemini-2.5-pro`).
- `gemini`: bare Gemini id (`gemini-2.5-pro`, `gemini-2.5-flash`).

When the user picks a per-request model, the adapter also re-resolves the credential to use:

- Anything starting `openrouter/` or `openrouter:` → `OPENROUTER_API_KEY`
- `anthropic/...` → `ANTHROPIC_API_KEY`
- `google/...` or `gemini:...` → `GEMINI_API_KEY`
- `openai/...` → `OPENAI_API_KEY`

So you can have all four provider keys in `.env` and switch between models freely from the UI.

### 3. Model catalog (UI picker)

The Ask form's Model dropdown is populated from two sources, both via `GET /v1/adapters/{name}/models`:

- **Recommended (curated)**: hand-maintained list in [`adapters/_models_catalog.py`](../src/tech_decomposition/adapters/_models_catalog.py) with annotations (price, context, caveats, our eval notes). 5-7 entries per adapter.
- **OpenRouter live (when applicable)**: all ~350 OpenRouter models fetched live from `https://openrouter.ai/api/v1/models` via [`_openrouter_models.py`](../src/tech_decomposition/adapters/_openrouter_models.py). Cached 1 hour server-side. Refreshable via `POST /v1/providers/openrouter/models/refresh`.

For adapters that accept OpenRouter ids (`opencode`, `pipeline`), the live list appears as a second `<optgroup>` after the curated section. Pick anything from either.

---

## End-to-end request walks

### Walk 1: UI ask with Hybrid mode on OpenCode + DeepSeek

```
User types "what are the user roles?" in the Ask composer.
SDK = opencode, Model = DeepSeek V3.2 (OpenRouter), Mode = Hybrid.
[Submit ⌘+Enter]
   ▼
Browser → POST localhost:18000/v1/adapters/opencode/ask
   body: { query, grounded: true, tools_enabled: true,
           model: "openrouter/deepseek/deepseek-v3.2" }
   ▼
FastAPI api.adapter_ask:
   1. Resolve adapter from registry → OpencodeSDKAdapter
   2. Run retrieve_grounded_context(query, settings, repos, top_k=8)
      → Sourcebot + Serena retrieve snippets
      → normalize, dedup, per-repo cap, rerank, expand
      → grounding_block = "## Relevant code\n### `traceability/...`\n```python\n...\n```"
   3. Wrap: final_query = grounding_block + "---\nUse the snippets..." + query
   4. inp = AdapterAskInput(query=final_query, model=..., tools_enabled=true)
   5. Call adapter.ask(inp)
   ▼
OpencodeSDKAdapter.ask:
   1. Resolve credential: model starts with "openrouter/" → OPENROUTER_API_KEY
   2. HTTP POST agent-node:13100/adapters/opencode/run
      body: { prompt, model, toolsEnabled: true, providerID, apiKey, ... }
   ▼
agent-node opencode handler:
   1. Get @opencode-ai/sdk v2 client pointed at opencode serve (host:4096)
   2. client.auth.set({providerID: "openrouter", auth: {key}})
   3. client.session.create({directory, agent: "build", model})
   4. client.session.prompt({sessionID, parts, agent: "build"})
      (toolsEnabled=true → no tools mask)
   ▼
opencode serve (host process):
   For each step in the agent loop (until model says "done"):
     a. Send {system, user, prior parts} to provider (DeepSeek via OpenRouter)
     b. Receive assistant message: text + maybe tool_use
     c. For each tool_use: execute the tool (read/grep/glob/...)
     d. Add tool result to message history
   ▼
Provider (OpenRouter → DeepSeek V3.2):
   - In our run: 0-13 tool calls depending on whether prefetch was enough
   - Returns text on the final step
   ▼
agent-node:
   5. Wait for prompt to resolve (returns final assistant message only)
   6. Fetch full session.messages(sessionID) → array of all assistant messages
   7. Walk parts: count tool_use parts, dedupe tool names, extract read filepaths
   8. Return { ok, answer, toolCalls, toolNames, groundingPaths, tokens, cost }
   ▼
OpencodeSDKAdapter.ask:
   - Map to AdapterAskResult { adapter, answer, metrics }
   - metrics.tool_calls = 5; metrics.extra.opencode_tool_names = ["read","grep","glob"]
   - metrics.extra.grounding_paths = ["traceability/src/...", ...]
   ▼
api.adapter_ask:
   - Save to runstore.db (SQLite)
   - Return JSON to browser
   ▼
Browser renders the answer in the transcript.
```

Total wall time: 30s-2min depending on tool call count.

### Walk 2: Agent-only mode (no prefetch)

Same as Walk 1 except:
- API skips step 2 (grounded retrieval) because `grounded: false`
- final_query = original user query (no grounding block prepended)
- OpenCode session runs the agentic loop from scratch, must navigate the repo via tools
- Wall time is typically longer: 1-3 min

### Walk 3: Grounded-only mode (no tools)

- API runs grounded retrieval
- Adapter sends `toolsEnabled: false`
- agent-node passes `tools: {read:false, grep:false, ...}` to session.prompt
- Model can only answer from the prepended snippets
- Single-shot, 5-15s

---

## Eval methodology

The eval lives in [`eval/`](../eval/). Three independent signals scored per case:

| Signal | How computed | Source of truth |
|---|---|---|
| **context_recall** | substring-match labeled `expected_files` against `metrics.extra.grounding_paths` | Hand-labeled `expected_files` in each case |
| **gold_coverage** | LLM judge (Gemini Flash): fraction of reference facts in the answer | Hand-written `gold_answer` |
| **gold_accuracy** | LLM judge: 1 − contradictions/answer | Hand-written `gold_answer` |

Each case in [`eval/questions.toml`](../eval/questions.toml) (ask) or [`eval/cases/decompose/`](../eval/cases/decompose/) (decompose) carries:

- `text` — the question
- `tags` — used by the pipeline router for tier classification
- `expected.expected_files` — paths that retrieval should surface (recall)
- `expected.gold_answer` — reference prose for the LLM judge
- `expected.must_mention` / `should_mention` — substring assertions

Run with:

```bash
uv run python -m eval.bakeoff.cli run \
    --adapters opencode \
    --job ask \
    --use-judge \
    --grounded
```

Output lands in `eval/outputs/eval-<ts>/` with a `report.md` containing the leaderboard, quality matrix, per-case answers, and judge verdicts.

---

## Why responses can be slow

Three layers stack:

1. **Provider latency** — DeepSeek V3.2 via OpenRouter typically 3-8s per call. Some routes (especially during peak hours) hit 15-30s.
2. **Agentic loop is serial** — every tool call adds a round-trip: model emits tool_use → tool executes locally → result fed back → next model call. 5 tool calls × 6s/call ≈ 30s of LLM time alone, plus tool execution time.
3. **Tool execution** — `read` is ~10ms, `grep` over a 1M-LOC monorepo is a few hundred ms, `glob` similar. Negligible relative to LLM time.

Headline numbers from our eval:

| Mode | Provider | Adapter | Avg wall |
|---|---|---|---|
| Grounded-only | Gemini Flash | `pipeline` | 5-15s |
| Grounded-only | DeepSeek V3.2 | `pipeline` (with openrouter:) | 10-30s |
| Hybrid (3-5 tools fire) | DeepSeek V3.2 | `opencode` | 60-90s |
| Hybrid (10+ tools fire) | DeepSeek V3.2 | `opencode` | 90-180s |
| Agent-only (full exploration) | DeepSeek V3.2 | `opencode` | 90-300s |

Mitigations:

- Use **Grounded only** mode for questions where you trust retrieval — drops to single-shot speed (5-15s).
- Use cheaper/faster providers for synthesis — Qwen3-235B is faster than DeepSeek V3.2 in our tests.
- Cap tool budget per query (not implemented yet — `pipeline_max_tool_rounds` exists for the fallback path but isn't enforced on opencode).
- Streaming output would improve perceived latency (UI shows tokens as they arrive) but not actual completion time.

---

## Local development setup

### Services that need to be running

| Service | Where | Port | When needed |
|---|---|---|---|
| FastAPI | host (`uv run uvicorn ...`) | `:18000` | always |
| Vite | host (`npm run dev` in `web/`) | `:15173` | for the UI |
| agent-node | Docker (`docker compose` default) or host | `:13100` | for cursor/cline/gemini/opencode/openai_agents/claude_code adapters |
| Sourcebot | Docker (`docker compose`) | `:13000` | for grounded modes (`pipeline`, hybrid/grounded-only modes) |
| Serena | host (`uvx ...`) | `:9121` | for symbol-graph expansion in grounding |
| opencode serve | host (`opencode serve --port 4096`) | `:4096` | for the `opencode` adapter |

The agent-node container needs to be able to reach `opencode serve` on the host. Two options:
- `OPENCODE_SDK_BASE_URL=http://host.docker.internal:4096` (Docker → host)
- Run agent-node on host directly (`node services/agent-node/dist/server.js`) and set `OPENCODE_SDK_BASE_URL=http://127.0.0.1:4096`. This bypasses the Docker network proxy 302s timeout we hit during eval.

### Smallest "everything off" config

You can run with **just** FastAPI + Vite (no Docker, no agent-node, no Sourcebot, no Serena, no opencode):
- All adapters except `pipeline` will fail health checks
- `pipeline` with `Single-shot` or `Agent only` mode will fail (needs Sourcebot)
- `pipeline` with `Grounded only` works because grounding fails open and we fall through to synthesis

For the OpenCode + OpenRouter "cheap agentic" path you described:
- FastAPI + Vite (host)
- agent-node (host or Docker)
- opencode serve (host, `:4096`)
- Sourcebot (only if using Hybrid or Grounded modes)
- Serena (only if using Hybrid/Grounded with symbol-graph expansion)

OpenRouter only needs the API key — no local service.

---

## File map (where to look for what)

| What | Where |
|---|---|
| HTTP API | [`src/tech_decomposition/api.py`](../src/tech_decomposition/api.py) |
| Adapter registry | [`src/tech_decomposition/adapters/registry.py`](../src/tech_decomposition/adapters/registry.py) |
| Pipeline adapter | [`src/tech_decomposition/adapters/_pipeline.py`](../src/tech_decomposition/adapters/_pipeline.py) |
| OpenCode adapter | [`src/tech_decomposition/adapters/_opencode_sdk.py`](../src/tech_decomposition/adapters/_opencode_sdk.py) + [`services/agent-node/src/adapters/opencode.ts`](../services/agent-node/src/adapters/opencode.ts) |
| Grounding | [`src/tech_decomposition/core/grounding.py`](../src/tech_decomposition/core/grounding.py) |
| Router / coverage / verifier | [`src/tech_decomposition/core/{router,coverage,verifier}.py`](../src/tech_decomposition/core/) |
| Synthesis (pipeline's LLM call) | [`src/tech_decomposition/core/synthesize.py`](../src/tech_decomposition/core/synthesize.py) |
| Model registry (provider routing) | [`src/tech_decomposition/core/llm_registry.py`](../src/tech_decomposition/core/llm_registry.py) |
| Curated model catalog (UI) | [`src/tech_decomposition/adapters/_models_catalog.py`](../src/tech_decomposition/adapters/_models_catalog.py) |
| Live OpenRouter fetch | [`src/tech_decomposition/adapters/_openrouter_models.py`](../src/tech_decomposition/adapters/_openrouter_models.py) |
| Eval cases | [`eval/questions.toml`](../eval/questions.toml) + [`eval/cases/decompose/`](../eval/cases/decompose/) |
| Eval scoring | [`eval/bakeoff/scorer.py`](../eval/bakeoff/scorer.py) |
| Eval report | [`eval/bakeoff/report.py`](../eval/bakeoff/report.py) |
| Web UI | [`web/src/App.tsx`](../web/src/App.tsx) |
