"""Adapter discovery — name → class lookup.

Registration is by **late import**: each adapter module is imported lazily on
first lookup so a missing optional dependency does not take down the whole API.
``health()`` reports the failure if anyone asks for that adapter specifically.

Current set: ``cursor`` (Cursor SDK), ``cline_sdk`` (Cline SDK),
``claude_code`` (Claude Agent SDK / Claude Code), ``gemini`` (Google Gemini via
``@google/genai``), ``openai_agents``, ``opencode`` (OpenCode @opencode-ai/sdk),
``sourcebot`` (blocking Sourcebot chat). All non-Python
runtimes are served by ``services/agent-node``. The HTTP API
routes through these adapters; the CLI may still call ``ask_sourcebot`` directly.
"""
from __future__ import annotations

import importlib

from .base import Adapter


_REGISTRY: dict[str, tuple[str, str]] = {
    "cursor": ("tech_decomposition.adapters._cursor_sdk", "CursorSDKAdapter"),
    "cline_sdk": ("tech_decomposition.adapters._cline_sdk", "ClineSDKAdapter"),
    "claude_code": ("tech_decomposition.adapters._claude_code_sdk", "ClaudeCodeSDKAdapter"),
    "gemini": ("tech_decomposition.adapters._gemini", "GeminiAdapter"),
    "openai_agents": ("tech_decomposition.adapters._openai_agents", "OpenAIAgentsAdapter"),
    "opencode": ("tech_decomposition.adapters._opencode_sdk", "OpencodeSDKAdapter"),
    "sourcebot": ("tech_decomposition.adapters._sourcebot", "SourcebotAdapter"),
    # Sourcebot retrieval + local Ollama generation. RAG pattern: free-grade
    # answers using Sourcebot's strong BM25/Flash-classifier retrieval and a
    # local model's free generation. Zero LLM cost per call in the common case.
    "sourcebot_ollama": ("tech_decomposition.adapters._sourcebot_ollama", "SourcebotOllamaAdapter"),
    # EXPERIMENTAL — max Sourcebot+Serena context → Ollama. Revert: delete this line + module.
    "sourcebot_ollama_max": (
        "tech_decomposition.adapters._sourcebot_ollama_max",
        "SourcebotOllamaMaxContextAdapter",
    ),
    # EXPERIMENTAL — Sourcebot+Serena → OpenRouter/Groq/etc. (no local Ollama).
    "sourcebot_rag_api": (
        "tech_decomposition.adapters._sourcebot_rag_api",
        "SourcebotRagApiAdapter",
    ),
}


def _load_class(name: str) -> type[Adapter]:
    if name not in _REGISTRY:
        raise KeyError(f"unknown adapter: {name!r}. Known: {sorted(_REGISTRY)}")
    module_name, class_name = _REGISTRY[name]
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    if not issubclass(cls, Adapter):
        raise TypeError(f"{module_name}.{class_name} is not an Adapter subclass")
    return cls


def get_adapter(name: str, settings) -> Adapter:
    """Instantiate the named adapter with the given Settings."""
    cls = _load_class(name)
    return cls(settings)


def list_adapters(settings) -> list[dict]:
    """Best-effort listing of every registered adapter and what it supports."""
    out: list[dict] = []
    for name in _REGISTRY:
        try:
            cls = _load_class(name)
        except ImportError as e:
            out.append({
                "name": name,
                "installed": False,
                "capabilities": [],
                "description": "",
                "health": {"ok": False, "reason": f"import failed: {e}"},
            })
            continue
        try:
            instance = cls(settings)
            health = instance.health()
        except Exception as e:  # noqa: BLE001 — health is best-effort
            health = {"ok": False, "reason": f"{type(e).__name__}: {e}"}
        out.append({
            "name": cls.name,
            "installed": True,
            "capabilities": sorted(cls.capabilities),
            "description": cls.description,
            "health": health,
        })
    return out


def register(name: str, module: str, classname: str) -> None:
    """Programmatic registration — useful for tests and plugins."""
    _REGISTRY[name] = (module, classname)
