# Sourcebot + Serena + free/cheap generation (no laptop GPU)

Same pipeline for every option:

```text
Question → Sourcebot search (+ Serena) → CONTEXT → cheap model → answer
```

| Adapter | Generation | When |
|---------|------------|------|
| **`sourcebot_rag_api`** | Internet API (OpenRouter, Groq, …) | **No laptop / no Ollama** |
| **`sourcebot_ollama_max`** | Ollama local or [Ollama Cloud](https://ollama.com/cloud) | Weak laptop OK with 7B, or cloud API key |
| `sourcebot_ollama` | Same, lighter retrieval | Baseline |

Set **`SERENA_URL`** for LSP snippets (recommended on all paths).

---

## Best options on the internet (2026)

### 1. OpenRouter free models (easiest, $0, no Ollama account)

Sign up: [openrouter.ai](https://openrouter.ai) → API key.

Strong **free** models for **code Q&A** (check [openrouter.ai/models?q=free](https://openrouter.ai/models?q=free)):

| Model id | Why |
|----------|-----|
| `nvidia/nemotron-3-super-120b-a12b:free` | Programming-focused, 1M context |
| `poolside/laguna-m.1:free` | Coding agent model |
| `deepseek/deepseek-v4-flash:free` | Fast, 1M context, good for long CONTEXT |
| `openai/gpt-oss-20b:free` | Open-weight; **same family as Ollama** `gpt-oss:20b` locally later |

```env
ENABLED_ADAPTERS=sourcebot_rag_api
SOURCEBOT_URL=http://localhost:13000
SOURCEBOT_API_KEY=...
SERENA_URL=http://localhost:9121/mcp

RAG_OPENAI_BASE_URL=https://openrouter.ai/api/v1
RAG_OPENAI_MODEL=deepseek/deepseek-v4-flash:free
OPENROUTER_API_KEY=sk-or-...
```

(You can use `CUSTOM_LLM_BASE_URL` / `CUSTOM_LLM_MODEL` instead of `RAG_OPENAI_*`.)

### 2. Ollama Cloud (best path to **same model locally later**)

Same API as local Ollama — when you get a laptop: `ollama pull <same-model-name>`.

1. [ollama.com](https://ollama.com) → sign in → [API key](https://ollama.com/settings/keys)
2. Pick a [cloud model](https://ollama.com/search?c=cloud) (e.g. `gpt-oss:20b-cloud`, `qwen3-coder:480b-cloud`)

```env
ENABLED_ADAPTERS=sourcebot_ollama_max
SOURCEBOT_URL=...
SOURCEBOT_API_KEY=...
SERENA_URL=http://localhost:9121/mcp

OLLAMA_BASE_URL=https://ollama.com
OLLAMA_MODEL=gpt-oss:20b-cloud
OLLAMA_API_KEY=...
```

**Later locally:** `OLLAMA_BASE_URL=http://localhost:11434/v1` and `ollama pull gpt-oss:20b` (or `qwen2.5-coder:7b`).

### 3. Groq free tier (very fast, smaller free models)

[console.groq.com](https://console.groq.com) → API key. Free tier rotates; check current models.

```env
ENABLED_ADAPTERS=sourcebot_rag_api
RAG_OPENAI_BASE_URL=https://api.groq.com/openai/v1
RAG_OPENAI_MODEL=llama-3.3-70b-versatile
CUSTOM_LLM_API_KEY=gsk_...   # or RAG_OPENAI_API_KEY
```

Coder variant (`qwen-2.5-coder-32b`) may be paid-only on Groq — verify in console.

---

## What you still run locally (light)

- **Sourcebot** (`make up`) — search index
- **This API** (`make dev` / Docker app)
- **Serena** (optional MCP on port 9121)

You do **not** need Ollama on the laptop for `sourcebot_rag_api`.

---

## Serena

```bash
uvx --from git+https://github.com/oraios/serena serena start-mcp-server \
  --transport streamable-http --port 9121 --context agent --project ./repos
```

```env
SERENA_URL=http://localhost:9121/mcp
```

---

## Try

```bash
make eval-adapters
curl -sS http://localhost:18000/v1/adapters/sourcebot_rag_api/ask \
  -H 'content-type: application/json' \
  -d '{"query":"What are the user roles in the platform?"}'
```

```bash
# Golden eval (needs GEMINI_API_KEY for judge only)
OPENROUTER_API_KEY=... make eval-golden-minimal ADAPTER=sourcebot_rag_api
```

---

## Map: internet now → Ollama later

| Hosted (now) | Local Ollama (later) |
|--------------|----------------------|
| `openai/gpt-oss-20b:free` (OpenRouter) | `gpt-oss:20b` |
| `gpt-oss:20b-cloud` (Ollama Cloud) | `gpt-oss:20b` |
| `qwen3-coder` cloud | `qwen2.5-coder:7b` / `cline-qwen:7b` Modelfile |

OpenRouter model names ≠ Ollama tags 1:1 — prefer **Ollama Cloud** or **gpt-oss** if you care about matching local weights later.

---

## Revert experiments

Remove from `ENABLED_ADAPTERS`: `sourcebot_rag_api`, `sourcebot_ollama_max`.

Delete registry lines in `adapters/registry.py` for adapters you do not want.
