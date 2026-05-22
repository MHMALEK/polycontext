# Experimental: `sourcebot_ollama_max`

Test adapter: **max retrieval (Sourcebot + Serena) → local Ollama**. Does not
change `sourcebot_ollama` or other adapters.

## Enable

```bash
# .env — opt in explicitly
ENABLED_ADAPTERS=sourcebot_ollama_max
OLLAMA_MODEL=cline-qwen:7b
SERENA_URL=http://localhost:9121/mcp   # recommended for this experiment
```

UI / API: choose adapter **`sourcebot_ollama_max`**.

## Compare vs baseline

```bash
# Baseline RAG
make eval-golden-minimal ADAPTER=sourcebot_ollama

# Max-context experiment
make eval-golden-minimal ADAPTER=sourcebot_ollama_max
```

Check `[retrieval]` output: more snippets, `supplement(N variants)`, Serena hits.

## Tunables (`.env`)

| Variable | Default | Role |
|----------|---------|------|
| `EXPERIMENTAL_OLLAMA_MAX_TOP_K` | 48 | Sourcebot file/snippet cap |
| `EXPERIMENTAL_OLLAMA_MAX_SNIPPET_LINES` | 12 | Lines of context per hit |
| `EXPERIMENTAL_OLLAMA_MAX_VARIANT_SEARCHES` | 16 | Parallel variant queries |
| `EXPERIMENTAL_OLLAMA_MAX_NUM_CTX` | 49152 | Ollama context window |
| `DECOMPOSE_MAX_CONTEXT_CHARS` | 80000 | Trim packed CONTEXT before Ollama |

## Revert

1. Remove `sourcebot_ollama_max` from `ENABLED_ADAPTERS`.
2. Delete registry entry in `src/tech_decomposition/adapters/registry.py`.
3. Delete `src/tech_decomposition/adapters/_sourcebot_ollama_max.py`.
4. (Optional) Remove `experimental_ollama_max_*` fields from `config.py` and
   `aggressive_retrieval` branch in `core/grounding.py` if unused elsewhere.

No other adapters are affected.
