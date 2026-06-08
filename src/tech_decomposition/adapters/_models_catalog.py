"""Curated per-adapter model catalog — what the UI's model picker shows.

Each entry is just enough info for the picker to render an option + tooltip:
adapter-formatted id, friendly name, provider, optional price hint per 1M
tokens (input/output), context window, and an optional ``note`` for known
caveats (e.g. requires extra env var, reasoning model not great at grounded
synthesis, etc.).

Keep the list small and curated. We're not mirroring OpenRouter's 200+
options — we're pointing at the ones that actually shipped well in our
eval (DeepSeek V3.2 for cheap agentic, Pro for accuracy ceiling, etc.).

To add a model: append to the right adapter's list. To deprecate one:
delete it. The UI re-reads on every page load.
"""
from __future__ import annotations


_MODEL_CATALOG: dict[str, list[dict]] = {
    # OpenCode CLI/SDK accepts ``provider/model`` ids. OpenRouter is the
    # cheap-OS gateway; anthropic/google are direct providers (require
    # their own env keys).
    "opencode": [
        {
            "id": "openrouter/deepseek/deepseek-v3.2",
            "name": "DeepSeek V3.2 (OpenRouter)",
            "provider": "openrouter",
            "in_per_m_usd": 0.25,
            "out_per_m_usd": 0.38,
            "context_k": 131,
            "note": "Best cost-quality on our eval — 0.82 acc at ~$0.002/query.",
        },
        {
            "id": "openrouter/qwen/qwen3-235b-a22b-2507",
            "name": "Qwen3 235B Instruct (OpenRouter)",
            "provider": "openrouter",
            "in_per_m_usd": 0.07,
            "out_per_m_usd": 0.10,
            "context_k": 262,
            "note": "Cheapest credible option — sub-Flash pricing.",
        },
        {
            "id": "openrouter/z-ai/glm-4.6",
            "name": "GLM 4.6 (OpenRouter)",
            "provider": "openrouter",
            "in_per_m_usd": 0.43,
            "out_per_m_usd": 1.74,
            "context_k": 203,
            "note": "Strong instruction-following; use the ``:exacto`` variant if schema strict.",
        },
        {
            "id": "openrouter/anthropic/claude-sonnet-4-5",
            "name": "Claude Sonnet 4.5 (via OpenRouter)",
            "provider": "openrouter",
            "context_k": 200,
            "note": "Strong on code-Q&A. Higher price than the OS options.",
        },
        {
            "id": "anthropic/claude-sonnet-4-20250514",
            "name": "Claude Sonnet 4 (direct)",
            "provider": "anthropic",
            "context_k": 200,
            "note": "Requires ANTHROPIC_API_KEY.",
        },
        {
            "id": "google/gemini-2.5-pro",
            "name": "Gemini 2.5 Pro (direct)",
            "provider": "google",
            "in_per_m_usd": 1.25,
            "out_per_m_usd": 10.0,
            "context_k": 2048,
            "note": "Accuracy ceiling in our eval (0.87) but 15× pricier than DeepSeek.",
        },
        {
            "id": "google/gemini-2.5-flash",
            "name": "Gemini 2.5 Flash (direct)",
            "provider": "google",
            "in_per_m_usd": 0.30,
            "out_per_m_usd": 2.50,
            "context_k": 1024,
            "note": "Cheap, lower faithfulness than DeepSeek on grounded code Q&A.",
        },
    ],
    # Pipeline adapter routes its synthesis call through llm_registry, which
    # uses ``provider:model`` form (colon). Different from opencode's slash
    # form — same set of models, different syntax.
    "pipeline": [
        {
            "id": "gemini:gemini-2.5-flash",
            "name": "Gemini Flash (default)",
            "provider": "gemini",
        },
        {
            "id": "openrouter:deepseek/deepseek-v3.2",
            "name": "DeepSeek V3.2 (OpenRouter)",
            "provider": "openrouter",
            "note": "+10% gold_accuracy over Flash on identical retrieval.",
        },
        {
            "id": "openrouter:qwen/qwen3-235b-a22b-2507",
            "name": "Qwen3 235B (OpenRouter)",
            "provider": "openrouter",
            "note": "Sub-Flash pricing.",
        },
        {
            "id": "gemini:gemini-2.5-pro",
            "name": "Gemini Pro",
            "provider": "gemini",
            "note": "Accuracy ceiling, ~15× cost of DeepSeek.",
        },
        {
            "id": "anthropic:claude-sonnet-4-5",
            "name": "Claude Sonnet 4.5",
            "provider": "anthropic",
        },
    ],
    "gemini": [
        {"id": "gemini-2.5-flash", "name": "Gemini 2.5 Flash", "provider": "google"},
        {"id": "gemini-2.5-pro", "name": "Gemini 2.5 Pro", "provider": "google"},
    ],
    # In-process LangChain agent. Uses the same ``provider:model`` colon spec as
    # the pipeline adapter (routed through init_chat_model, not llm_registry).
    "langchain": [
        {"id": "gemini:gemini-2.5-flash", "name": "Gemini 2.5 Flash (default)", "provider": "gemini"},
        {"id": "gemini:gemini-2.5-pro", "name": "Gemini 2.5 Pro", "provider": "gemini"},
        {"id": "anthropic:claude-sonnet-4-5", "name": "Claude Sonnet 4.5", "provider": "anthropic"},
        {"id": "openai:gpt-4.1", "name": "GPT-4.1", "provider": "openai"},
        {
            "id": "openrouter:deepseek/deepseek-v3.2",
            "name": "DeepSeek V3.2 (OpenRouter)",
            "provider": "openrouter",
            "note": "Cheap agentic; routes via OpenRouter — not code-privacy compliant.",
        },
    ],
    "cursor": [
        {"id": "composer-2", "name": "Composer-2", "provider": "cursor"},
    ],
    "claude_code": [
        {"id": "claude-sonnet-4-5", "name": "Claude Sonnet 4.5", "provider": "anthropic"},
        {"id": "claude-opus-4-5", "name": "Claude Opus 4.5", "provider": "anthropic"},
    ],
    "openai_agents": [
        {"id": "gpt-4.1", "name": "GPT-4.1", "provider": "openai"},
        {"id": "gpt-4o-mini", "name": "GPT-4o Mini", "provider": "openai"},
    ],
    "cline_sdk": [
        {"id": "gemini-2.5-pro", "name": "Gemini 2.5 Pro", "provider": "google"},
        {"id": "claude-sonnet-4-5", "name": "Claude Sonnet 4.5", "provider": "anthropic"},
    ],
    # ``sourcebot`` runs through its own blocking chat endpoint — model is
    # chosen by Sourcebot config, not per-request. UI shows a static row.
    "sourcebot": [
        {
            "id": "(sourcebot-managed)",
            "name": "Sourcebot-managed (see /api/chat config)",
            "provider": "sourcebot",
            "note": "Model is selected by Sourcebot, not per-request from this API.",
        },
    ],
}


def models_for_adapter(name: str) -> list[dict]:
    """Return the curated model list for the named adapter. Empty list when
    the adapter isn't in the catalog — UI hides the model picker in that case."""
    return list(_MODEL_CATALOG.get(name, []))
