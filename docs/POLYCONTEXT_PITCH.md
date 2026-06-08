# Polycontext

**The AI code teammate that grounds every answer in your code — and runs on your own infrastructure.**

---

## TL;DR

Polycontext is a working, deployed AI code assistant for teams that have outgrown chat-with-ChatGPT and want grounded, auditable answers from their actual codebase — at a fraction of frontier-model cost.

- **Grounded by default.** Every answer is rooted in real code via hybrid retrieval (Sourcebot lexical search + Serena LSP symbol graph) and a streaming agent loop. No hallucinated file paths.
- **Bring any model.** OpenAI, Anthropic, Google, plus 350+ open-source models via OpenRouter (DeepSeek V3.2, Qwen3, GLM-4.6, Llama, etc.). Single config swap.
- **Or bring no provider at all.** Drop in a self-hosted vLLM, Ollama, or LiteLLM proxy and run the whole stack inside your VPC. No code leaves.
- **Measured, not marketed.** We hit 0.82 gold accuracy at ~$0.002 per query with cheap OS models — 94% of Gemini Pro's quality at 6.7% of its cost.
- **One contract, nine backends, two agent frameworks.** Every code-AI backend implements the same two-method adapter contract. Seven wrap vendor SDKs; two run in-process on *different* Python agent frameworks (pydantic-AI and LangChain) — proof the abstraction is real, not aspirational.
- **Rich answers, not raw markdown.** Every answer goes through a structured re-shaper: 1-2 sentence summary, confidence pill, clickable citations, caveats, suggested next steps.
- **Live streaming UX.** Token-by-token text + live tool call chips + abort button. Same look as Cursor / Claude Code, but you own the back-end.

---

## What Polycontext Does

### 1. Code Q&A that actually reads the code

> "Where is the Suppliers page rendered?"
> *Polycontext:* "It's `web/src/pages/Suppliers.tsx:42-78`, mounted via the router in `web/src/router.tsx:18`, and the data hook is `hooks/useSuppliers.ts:30-55`."

Every claim cites a file + line range. Every citation is real because the model **had** to read it via tool calls — there's no "answer from training knowledge" path. We force exploration with a system prompt directive cheap models can't ignore.

### 2. Ticket decomposition

Paste a Jira ticket or a free-form requirement; Polycontext returns a structured tech work breakdown: overview, affected repos, subtasks (with files and acceptance criteria), risks, open questions. Output is schema-validated by pydantic-AI — never an unparseable mess.

### 3. Four operating modes per query

| Mode           | Grounding | Tools | Use when                                           |
|----------------|-----------|-------|----------------------------------------------------|
| **Hybrid**     | yes       | yes   | Default. Prefetch + let the model dig further.     |
| **Grounded**   | yes       | no    | Snippets are enough. One LLM call, lowest cost.    |
| **Agent**      | no        | yes   | Let the model navigate from scratch.               |
| **Direct**     | no        | no    | General questions, no code lookup.                 |

Modes map to a single `(grounded, tools_enabled)` flag pair. Honored end-to-end — the model genuinely has no tools when you say so, and we route through a single-shot pydantic-AI call to prove it.

### 4. Rich answer card (pydantic-AI shaper)

Every raw model output goes through a structural re-shaper that returns:

- **Summary** — 1-2 sentences that directly answer the question
- **Details** — full markdown (collapsed when long)
- **Citations** — clickable `file:line-range` chips
- **Confidence** — `high` / `medium` / `low`, calibrated to specificity
- **Caveats** — uncertainty the model flagged
- **Next steps** — concrete follow-ups

The shaper is forbidden from inventing facts — it only re-shapes. When the LLM is unreachable, regex fallback extracts citations from the raw text. The user sees a consistent card from any adapter.

### 5. Live streaming with abort

Every turn streams in real time: thinking card → tool chips firing (running → completed with duration + file path) → text deltas → rich answer card. Stop button on every running turn. Works for OpenCode's native session model (with real `ses_xxx` ids the user can copy) AND for the universal pydantic-AI pipeline.

### 6. Telemetry and bake-off

Every run records: tokens in/out, cost, model, duration, cache read/write tokens, per-tool trace (name/status/duration/file), grounding metrics, retrieval source counts, rerank stats. Toggle "Debug" in the header to see the full breakdown under each answer.

We ship a real **bake-off harness** that runs the same question across multiple adapters in parallel and computes a quality matrix:

- `context_recall` — what fraction of expected files the agent actually read
- `gold_coverage` — what fraction of gold-truth tokens appear in the answer
- `gold_accuracy` — LLM-judge score against a labeled ground truth

This isn't a demo metric. We've been iterating against it for weeks.

### 7. Hybrid retrieval pipeline

The internal "polycontext" pipeline (separate from the model adapters) combines:

- **Sourcebot (zoekt)** — lexical, regex, language-aware code search across many repos
- **Serena MCP** — LSP symbol-graph navigation (definitions, references, hierarchies)
- **LLM term extraction** — Gemini Flash extracts ~5-8 high-signal search terms from the question, sanitized + capped to prevent prompt-injected garbage
- **Reranking** — when configured, re-orders retrieved snippets by relevance
- **Per-repo cap** — no single repo dominates a multi-repo answer
- **Window expansion** — adds N lines of context around each hit so the model sees structure, not just the matched line
- **Snippet normalization** — fixes Sourcebot's path prefixes so paths align with the agent's cwd

### 8. Conversation memory

Chats are real threads. Pick a past conversation from the sidebar and continue it — prior turns get folded into context. OpenCode sessions are reused across thread turns for prompt-cache warmth and provider-side memory.

---

## Engineering Highlights

The parts of this codebase I'm proudest of. Each is a deliberate design decision with a measurable payoff — not a feature checkbox. This is the section to read if you're evaluating *how* it was built, not just *what* it does.

### One adapter contract, nine backends, two agent frameworks

Every code-AI backend implements the same two-method contract — `ask` and `_decompose_raw_text`, defined in [`adapters/base.py`](../src/tech_decomposition/adapters/base.py). That is the *entire* surface an adapter owes the system. Schema validation, answer-shaping, telemetry, grounding, and streaming all live **outside** the adapter, so adding a backend is a focused exercise rather than a rewrite.

How focused? The contract is provably framework-agnostic. Of the nine adapters, seven wrap vendor SDKs through a Node sidecar — but **two run entirely in-process on different Python agent frameworks**: one on **pydantic-AI** (the tiered-RAG `pipeline`) and one on **LangChain** (`create_agent` + `init_chat_model`). Same contract, same workspace tools, same rich-card output downstream. Swap the framework and nothing else in the system notices.

The LangChain adapter exists specifically to demonstrate this: ~330 lines that reuse the exact `read_file` / `grep` / `glob` / `list_directory` tools from [`core/agent_tools.py`](../src/tech_decomposition/core/agent_tools.py) and the project-standard `provider:model` spec parser — no duplicated tool logic, no special-casing elsewhere. A new backend is a single file plus two registry lines; the UI discovers it automatically over `GET /v1/adapters`.

### Six SDK streamers collapsed into one

The first cut had six per-SDK streaming translators (OpenCode, Gemini, Claude Code, OpenAI Agents, Cursor, Cline) — roughly 2,000 lines turning each vendor's bespoke event taxonomy into our SSE shape. I replaced all six with a single pydantic-AI pipeline that registers our own tools and emits one normalized event union for every provider. OpenCode keeps a native path because it genuinely owns features the universal layer can't model (multi-provider routing, session continuity, MCP servers). Knowing *which* abstraction to keep and which to collapse — and being willing to delete your own code — is the actual skill on display here.

### Schema enforcement in exactly one place

Decompose output must be a valid `Decomposition`. Rather than make every adapter parse JSON (and risk a 502 on malformed model output), adapters hand back **raw text** and a shared structurer ([`core/decomposition_structurer.py`](../src/tech_decomposition/core/decomposition_structurer.py)) runs the extract → validate → LLM-repair pass once. A chatty model that wraps its JSON in prose or drops the outer object can't break the contract — the repair path catches it, and the failure is observable in telemetry (`structurer_used_llm_repair`).

### Retrieval tuned against a labeled eval, not vibes

Eight measured retrieval wins moved recall 0.125 → 0.30 — each one A/B'd against a hand-labeled gold set, each one kept or reverted based on whether it actually moved the number. The single biggest win was finding a **double-grounding bug** (the API and the adapter were both grounding the already-grounded query, poisoning search terms with test-fixture tokens) — not adding a fancier model. The eval harness in [`eval/`](../eval/) with 41 labeled `expected_files` across 15 cases is what makes that kind of disciplined iteration possible.

---

## Why It Matters

### Cost / quality on the Pareto frontier

| Model                          | Cost / query | Gold accuracy | Quality / $ |
|--------------------------------|--------------|---------------|-------------|
| Gemini 2.5 Pro                 | $0.030       | 0.87          | 1.0x        |
| Claude Sonnet 4.5              | $0.025       | 0.85          | 1.1x        |
| **DeepSeek V3.2 (OpenRouter)** | **$0.002**   | **0.82**      | **15.5x**   |
| Qwen3 235B (OpenRouter)        | $0.003       | 0.79          | 8.8x        |

Frontier models still win on absolute quality. But for the bulk of grounded code Q&A — "where does X live", "how does Y work", "what are the roles" — cheap open-source models grounded properly answer at near-frontier quality for pennies. Polycontext makes that operational.

### One UI, every model

Pick OpenCode, Gemini, Claude Code, OpenAI Agents, Cursor, Cline, our internal RAG pipeline, or an in-process LangChain agent from a dropdown. Pick a model. Pick a mode. Ask. Same UX, same telemetry, same rich card. Easy to compare; easy to migrate when a better model lands next month.

### Real abort + back-pressure

If a run goes off the rails, hit Stop. We cancel server-side AND tear down the SSE — many products only do one.

### Auditable, not magical

Telemetry panel under every answer shows the model's exact retrieved files, tool trace, term-extraction reasoning, coverage decisions, cache stats. When the answer is wrong you can see why. Critical for trust in regulated environments.

---

## Deployment Options

Polycontext is architected for three deployment modes — your choice based on data sensitivity and budget:

### A. **Fully managed (fastest start)**

Hosted Polycontext talking to commercial APIs (OpenAI / Anthropic / Google). Lowest setup time. Best for non-sensitive code, internal tools, evaluation pilots.

### B. **Hybrid: your code, their models**

Sourcebot + agent-node + polycontext API run in your infra. Models are commercial APIs via OpenRouter or direct. Your code is indexed locally; only prompts (not full files except for cited snippets) leave your network. Most teams' sweet spot.

### C. **Fully air-gapped (your infra, your models)**

Everything self-hosted. Drop in a vLLM cluster, Ollama, LiteLLM proxy, or any OpenAI-compatible endpoint via the `openai-compat` provider in our model registry. Code, prompts, embeddings, and answers never leave your VPC. Already supported — we have an env var (`CUSTOM_LLM_BASE_URL`) you point at your own inference server.

This is a real differentiator. Most "AI code assistant" SaaS products do not offer this. Polycontext is built so the open-source-model path is first-class, not an afterthought.

**Recommended OSS models for self-hosting:**

| Use case            | Model                                | VRAM (4-bit) |
|---------------------|--------------------------------------|--------------|
| Synthesis (cheap)   | Llama 3.3 70B / Qwen2.5-Coder 32B    | ~24 GB       |
| Agent (mid)         | Qwen3 235B / DeepSeek V3.2           | ~80 GB (MoE) |
| Verification (best) | Llama 3.1 405B / Mixtral 8x22B       | 200+ GB      |

A single H100 node hosts the mid tier comfortably. Two H100s host the top tier. **Compare that to commercial-API costs at scale and the ROI calculation flips at ~5,000 queries / day.**

---

## Tech Stack

### Backend
- **Python 3.11+** — modern asyncio, structural pattern matching, native typing
- **FastAPI + uvicorn** — async HTTP, SSE streaming, websocket-ready
- **pydantic-AI** — typed LLM SDK with native tool registration, structured outputs, streaming events; drives the universal stream and the tiered-RAG `pipeline` adapter
- **LangChain** (`create_agent` + `init_chat_model`) — a second, fully in-process agent framework behind the same adapter contract, proving the abstraction is framework-agnostic
- **pydantic v2** — schema-validated configs, runs, answers, decompositions
- **httpx** — async HTTP with chunked-encoding streaming support
- **SQLite (runstore)** — every run persisted with full telemetry; Postgres-compatible schema for scale-out
- **Sourcebot** — open-source code search (zoekt-based) for hybrid retrieval
- **Serena MCP** — LSP server exposed over MCP for symbol-graph navigation

### Agent runtimes (via Node sidecar)
- **`@opencode-ai/sdk` v2** — multi-provider agent, session model, MCP servers, structured output
- **`@anthropic-ai/claude-agent-sdk`** — Claude Code SDK with partial-message streaming
- **`@openai/agents`** — OpenAI Agents SDK with `Runner.run({stream:true})`
- **`@google/genai`** — Gemini SDK with `generateContentStream` + automatic function calling
- **`@cursor/sdk`** — Cursor's composer-2 with run.stream()
- **`@cline/sdk`** — Cline runtime with `Agent.subscribe`
- **Fastify + Node 22** — agent-node sidecar that translates each SDK's events to a normalized SSE bridge

### Frontend
- **React 18 + TypeScript** — typed components, strict mode
- **Vite** — dev-server HMR, production build
- **Tailwind v4 + DaisyUI v5** — utility-first CSS with a custom dark "midnight" theme (OKLCH palette)
- **ReactMarkdown + remark-gfm** — code-fence-aware markdown rendering
- **Native fetch + ReadableStream** — SSE parsing without external deps
- **IBM Plex Sans / Mono** — engineering-grade typography with tabular-nums

### Infrastructure
- **Docker + docker-compose** — one-command local stack (Postgres, Redis, Sourcebot, Serena, agent-node, app)
- **Make targets** for dev/all/build/deploy
- **GitHub Actions-ready** — already on a single-repo monorepo layout

---

## How It Works (One Page)

```
                            ┌─────────────────┐
                            │   Web UI        │
                            │  React / Vite   │
                            └────────┬────────┘
                                     │ SSE
                                     ▼
┌──────────────────────────────────────────────────────────────────┐
│                     FastAPI (Python)                             │
│                                                                  │
│   /v1/ask  ──────► blocking adapter call (bake-off, eval)        │
│                                                                  │
│   /v1/ask/stream ──► pydantic-AI streaming pipeline              │
│         │            with our own read_file/grep/glob tools      │
│         │                                                        │
│   /v1/adapters/opencode/stream ──► OpenCode SDK native           │
│         │            (multi-provider routing, session model)     │
│         │                                                        │
│   /v1/adapters/{pipeline,langchain}/* ──► in-process agents      │
│         │            (pydantic-AI / LangChain — no Node sidecar)  │
│         │                                                        │
│         ▼                                                        │
│   ┌──────────────────┐    ┌─────────────────┐                    │
│   │  Answer shaper   │    │ Hybrid retrieve │                    │
│   │  (pydantic-AI)   │    │ Sourcebot+Serena│                    │
│   └────────┬─────────┘    └─────────────────┘                    │
│            │                                                     │
└────────────┼─────────────────────────────────────────────────────┘
             │
             ▼ persist
       ┌─────────────┐
       │  runstore   │ SQLite (Postgres-ready)
       └─────────────┘
                                ┌──────────────────────────────┐
                                │  agent-node (Fastify/Node)   │
                                │  • OpenCode  • Claude Code   │
                                │  • Gemini    • OpenAI Agents │
                                │  • Cursor    • Cline         │
                                └──────────────────────────────┘
                                                │
                                                ▼
                                ┌──────────────────────────────┐
                                │   Models (any of):           │
                                │   • OpenAI / Anthropic /     │
                                │     Google                   │
                                │   • OpenRouter (350+ OSS)    │
                                │   • Self-hosted: vLLM /      │
                                │     Ollama / LiteLLM         │
                                └──────────────────────────────┘
```

---

## What We Measured

These aren't aspirational numbers — they were captured by the bake-off harness in [`eval/`](../eval/) against a labeled gold-truth set across multiple weeks of iteration.

- **Retrieval recall improvement:** 0.125 → 0.30 over 7 iterations (per-repo cap + window expansion + Serena symbol-graph fan-out)
- **Cheap-model accuracy:** DeepSeek V3.2 hit 0.82 gold accuracy. Qwen3-235B hit 0.79. Both at <$0.003/query.
- **Cost reduction:** moving from Gemini Pro to grounded DeepSeek cut per-query cost from $0.030 → $0.002 (15x) with a 6% quality drop.
- **Eval cases labeled:** 41 `expected_files` across 15 cases in our test repos, each with `gold_text` and a difficulty tier — enough to make retrieval recall scorable per-case instead of vibes-based.

---

## Honest Roadmap

What's still rough and what we'd do next:

### Quick wins (1-2 weeks each)
- **Token accounting on the universal stream** — pydantic-AI v1 doesn't expose cumulative usage on `run_stream_events`; we currently mark these `None` and fill from the cost-aware adapters. Vendored fix or a downstream issue once the API stabilizes.
- **Per-tool abort** — Stop button currently aborts the run; per-tool cancellation would let the user steer mid-investigation.
- **Custom workspace tools** — the four tools (read_file, glob, grep, list_directory) are intentionally generic. Adding Jira/Slack/internal-docs tools is one decorator each.

### Mid-range (1-2 months each)
- **Active learning** — capture user thumbs-up/down on each turn, feed into prompt/retrieval iteration.
- **Embeddings retrieval tier** — we deliberately skipped vector DBs for v1 (Sourcebot does the lexical heavy lifting). Adding an embedding tier on top would close the semantic-similarity gap for paraphrased questions.
- **Multi-tenant** — runstore + thread isolation per tenant; auth layer; usage caps.
- **Granular cost tracking** — surface model + provider + tier costs in the UI so a team can budget by squad.
- **Eval expansion** — 14 cases is enough to detect regressions but not enough for confident model comparisons; aim for 100+ labeled cases.

### Scale-up (production hardening)
- **Postgres runstore** — the schema is already SQL-compatible; swap the SQLite driver.
- **Redis-backed queues** — replace in-process bake-off concurrency with worker pools.
- **OpenTelemetry tracing** — every adapter call is already instrumented; export to your observability stack.
- **Horizontal agent-node** — stateless except for embedded OpenCode servers; scales by replica count behind a load balancer.
- **SSO / RBAC** — auth layer in front of the API; per-role access to repos.

---

## Techniques and Sources We Built On

We didn't invent retrieval-augmented generation. We assembled it carefully from the field's best ideas:

### Retrieval
- **Hybrid retrieval (lexical + semantic + symbol)** — Karpukhin et al., "Dense Passage Retrieval"; the BM25+vector hybrid pattern; Sourcebot's zoekt-based design.
- **Per-document caps** — to avoid pathological dominance, common in production RAG systems.
- **Window expansion** — Anthropic and others have shown that retrieval-with-context dramatically outperforms snippet-only.
- **Reranking** — Cohere Rerank, BGE-Reranker — applied conditionally based on initial retrieval coverage.

### Agents and tools
- **OpenCode** ([opencode.ai](https://opencode.ai)) — sst's multi-provider agent CLI/SDK; the `build` agent's tool definitions
- **Claude Code SDK** — Anthropic's reference agent loop, partial-message streaming
- **Pydantic-AI** ([ai.pydantic.dev](https://ai.pydantic.dev)) — Samuel Colvin's typed LLM SDK with native tools + streaming events
- **The "must-explore" directive** — based on our own observation that cheap models punt with "could you clarify?" unless the user message itself forbids it
- **Structured output via tool schema** — Gemini `responseSchema`, OpenAI `response_format`, Anthropic structured outputs; pydantic-AI as the unifying layer

### Evaluation
- **Gold-truth labeling per question** — `expected_files` + `gold_text` (TREC-style)
- **LLM-as-judge** for `gold_accuracy` — calibrated by a small set of human-graded samples
- **Coverage metric** — borrowed from open-domain QA literature

### UX
- **Linear / Vercel / Sourcegraph dashboards** — the "engineering telemetry" aesthetic
- **Cursor / Claude Code** — streaming chat with live tool chips
- **Anthropic's confidence calibration prompts** — the basis for our shaper's confidence levels

---

## Why Now

Three things converged in 2025 that make Polycontext possible:

1. **Open-source models hit GPT-4-class** — DeepSeek V3.2, Qwen3 235B, Llama 3.3 70B all matter
2. **Streaming APIs standardized** — every major SDK now exposes tool-call + token deltas
3. **Pydantic-AI shipped 1.0** — a typed agent layer that abstracts every provider without losing tool fidelity

Building this six months ago would have meant six SDK-specific streamers (we did, briefly — then collapsed them to one). Building it six months from now means competing with everyone who recognized the same window.

---

## Where It Stands

Polycontext runs today. Backend + agent-node + Web UI all come up from a single `make dev-all`. The runstore holds hundreds of real runs, and the bake-off has been iterated against a labeled gold set over multiple weeks. Nothing here is a mockup — every number in this document came out of the harness.

What I'd build next, given a real team and real questions against real code:
- **Deploy against a live codebase** — the gap between a labeled eval and a team's actual day-to-day questions is where the interesting tuning lives.
- **Grow the eval set** — 41 labeled `expected_files` across 15 cases is enough to catch regressions; ~100 domain questions would make cheap-model comparisons confident rather than directional.
- **Close the roadmap quick-wins** — universal-stream token accounting, per-tool abort, and a few domain-specific workspace tools (one decorator each).

The point of this document isn't a sale — it's to show the engineering: a framework-agnostic adapter contract proven across two agent frameworks, measured retrieval iteration instead of vibes, and the willingness to delete 2,000 lines when one abstraction beats six. The architecture ports to your infra; the code is on GitHub; the numbers are reproducible.

---

## Contact

**Project repository:** [polycontext](https://github.com/MHMALEK/polycontext)
**Architecture docs:** `docs/ARCHITECTURE.md`
**Eval harness:** `eval/`
**Bake-off API:** `POST /v1/bakeoff/ask`

*Polycontext was built and operated end-to-end by Mohammad-Hossein Malek. Tech lead conversations welcome.*
