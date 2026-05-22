"""EXPERIMENTAL: Sourcebot + Serena retrieval → hosted OpenAI-compatible API.

Use when you have **no local GPU** (OpenRouter free models, Groq, etc.).
Revert: remove ``sourcebot_rag_api`` from ``registry.py`` / ``ENABLED_ADAPTERS``.

Later on a laptop, switch to ``sourcebot_ollama`` + the same Qwen/OSS weights via Ollama.
"""
from __future__ import annotations

import time
from typing import Any

from ..core.grounding import retrieve_grounded_context
from ._openai_http import openai_chat, openai_health_check, openai_is_configured
from ._sourcebot_ollama import (
    SourcebotOllamaAdapter,
    _ASK_SYSTEM,
    _DECOMPOSE_SYSTEM,
    _snippet_to_citation,
)
from ._prompts import DECOMPOSE_PREAMBLE, query_blob
from .base import (
    AdapterAskInput,
    AdapterAskResult,
    AdapterDecomposeInput,
    Capability,
    RawDecomposeText,
)


class SourcebotRagApiAdapter(SourcebotOllamaAdapter):
    name = "sourcebot_rag_api"
    capabilities: set[Capability] = {"ask", "decompose"}
    description = (
        "[EXPERIMENTAL] Sourcebot (+ Serena) RAG → OpenAI-compatible hosted API "
        "(OpenRouter free, Groq, etc.). No local Ollama. See "
        "docs/serena-sourcebot-free-model.md."
    )

    def health(self) -> dict:
        if not (self.settings.sourcebot_url or "").strip():
            return {"ok": False, "reason": "SOURCEBOT_URL not set"}
        if not self.settings.sourcebot_api_key:
            return {"ok": False, "reason": "SOURCEBOT_API_KEY not set"}
        if not openai_is_configured(self.settings):
            return {
                "ok": False,
                "reason": (
                    "Set RAG_OPENAI_* or CUSTOM_LLM_BASE_URL + CUSTOM_LLM_MODEL + "
                    "OPENROUTER_API_KEY (see docs/serena-sourcebot-free-model.md)"
                ),
            }
        api_h = openai_health_check(self.settings)
        if not api_h.get("ok"):
            return api_h
        out = {"ok": True, **{k: v for k, v in api_h.items() if k != "ok"}}
        out["serena"] = (
            "configured" if (self.settings.serena_url or "").strip() else "unset (recommended)"
        )
        return out

    async def ask(self, inp: AdapterAskInput) -> AdapterAskResult:
        t0 = time.monotonic()
        ctx = await retrieve_grounded_context(
            query=inp.query,
            settings=self.settings,
            repos=inp.repos,
            top_k=self._retrieval_top_k(),
            context_lines=self._retrieval_context_lines(),
            aggressive_retrieval=True,
        )
        prompt = self._ask_prompt(inp.query, ctx.grounding_block)
        gen = await openai_chat(
            self.settings,
            system=_ASK_SYSTEM,
            user=prompt,
            max_tokens=4096,
        )
        metrics = self._metrics(t0=t0, ctx=ctx, gen=gen)
        metrics.extra["hosted_api"] = True
        metrics.model = gen.get("model")
        return AdapterAskResult(
            adapter=self.name,
            answer=(gen.get("content") or "").strip(),
            citations=_snippet_to_citation(ctx.snippets),
            metrics=metrics,
        )

    async def _decompose_raw_text(self, inp: AdapterDecomposeInput) -> RawDecomposeText:
        t0 = time.monotonic()
        query = inp.query or ""
        ctx = await retrieve_grounded_context(
            query=query,
            settings=self.settings,
            repos=inp.repos,
            top_k=self._retrieval_top_k(),
            context_lines=self._retrieval_context_lines(),
            aggressive_retrieval=True,
        )
        prompt = (
            DECOMPOSE_PREAMBLE.format(model_tag=self.name)
            + query_blob(inp)
            + "\n\n"
            + self._context_block(ctx.grounding_block)
        )
        gen = await openai_chat(
            self.settings,
            system=_DECOMPOSE_SYSTEM,
            user=prompt,
            max_tokens=8192,
        )
        metrics = self._metrics(t0=t0, ctx=ctx, gen=gen)
        metrics.extra["hosted_api"] = True
        metrics.model = gen.get("model")
        return RawDecomposeText(
            text=(gen.get("content") or ""),
            metrics=metrics,
        )
