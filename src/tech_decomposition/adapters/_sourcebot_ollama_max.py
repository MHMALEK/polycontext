"""EXPERIMENTAL: max Sourcebot (+ Serena) context → local Ollama.

Test-only adapter — revert by removing ``sourcebot_ollama_max`` from
``adapters/registry.py`` and ``ENABLED_ADAPTERS``. Does not change
``sourcebot_ollama`` behavior.

Pipeline
--------
Same as ``sourcebot_ollama``, but retrieval is tuned for breadth:

- Higher ``top_k`` / ``context_lines`` (config: ``EXPERIMENTAL_OLLAMA_MAX_*``).
- ``aggressive_retrieval=True`` in grounding (more variant searches, Serena cap ×2).
- Optional context trim before Ollama when block exceeds ``DECOMPOSE_MAX_CONTEXT_CHARS``.
"""
from __future__ import annotations

import time
from typing import Any

from ..core.grounding import retrieve_grounded_context
from ._prompts import DECOMPOSE_PREAMBLE, query_blob
from ._ollama_http import ollama_chat, ollama_health_check
from ._sourcebot_ollama import (
    SourcebotOllamaAdapter,
    _ASK_SYSTEM,
    _DECOMPOSE_SYSTEM,
    _snippet_to_citation,
)
from .base import (
    AdapterAskInput,
    AdapterAskResult,
    AdapterDecomposeInput,
    RawDecomposeText,
)


class SourcebotOllamaMaxContextAdapter(SourcebotOllamaAdapter):
    name = "sourcebot_ollama_max"
    description = (
        "[EXPERIMENTAL] Max Sourcebot + Serena retrieval → local Ollama. "
        "Higher snippet count and aggressive search fan-out. "
        "Revert: drop from ENABLED_ADAPTERS and registry."
    )

    def health(self) -> dict:
        if not (self.settings.sourcebot_url or "").strip():
            return {"ok": False, "reason": "SOURCEBOT_URL not set"}
        if not self.settings.sourcebot_api_key:
            return {"ok": False, "reason": "SOURCEBOT_API_KEY not set"}
        ollama_h = ollama_health_check(self.settings)
        if not ollama_h.get("ok"):
            return ollama_h
        if not (self.settings.serena_url or "").strip():
            return {
                "ok": True,
                "serena": "unset (recommended for this adapter)",
                **{k: v for k, v in ollama_h.items() if k != "ok"},
            }
        return {"ok": True, "serena": "configured", **{k: v for k, v in ollama_h.items() if k != "ok"}}

    def _retrieval_top_k(self) -> int:
        return max(24, int(self.settings.experimental_ollama_max_top_k or 48))

    def _retrieval_context_lines(self) -> int:
        return max(8, int(self.settings.experimental_ollama_max_snippet_lines or 12))

    def _ollama_num_ctx(self) -> int:
        return max(8192, int(self.settings.experimental_ollama_max_num_ctx or 49152))

    async def _retrieve(self, *, query: str, repos: list[str] | None):
        return await retrieve_grounded_context(
            query=query,
            settings=self.settings,
            repos=repos,
            top_k=self._retrieval_top_k(),
            context_lines=self._retrieval_context_lines(),
            aggressive_retrieval=True,
        )

    def _trim_block(self, block: str) -> str:
        cap = int(self.settings.decompose_max_context_chars or 80_000)
        b = (block or "").strip()
        if len(b) <= cap:
            return b
        return (
            b[:cap]
            + "\n\n… [context truncated for local model — increase "
            "DECOMPOSE_MAX_CONTEXT_CHARS or lower EXPERIMENTAL_OLLAMA_MAX_TOP_K]"
        )

    async def ask(self, inp: AdapterAskInput) -> AdapterAskResult:
        t0 = time.monotonic()
        ctx = await self._retrieve(query=inp.query, repos=inp.repos)
        block = self._trim_block(ctx.grounding_block)
        prompt = self._ask_prompt(inp.query, block)
        gen = await ollama_chat(
            self.settings,
            system=_ASK_SYSTEM,
            user=prompt,
            num_ctx=self._ollama_num_ctx(),
        )
        metrics = self._metrics(t0=t0, ctx=ctx, gen=gen)
        metrics.extra["experimental_max_context"] = True
        metrics.extra["context_chars"] = len(block)
        return AdapterAskResult(
            adapter=self.name,
            answer=(gen.get("content") or "").strip(),
            citations=_snippet_to_citation(ctx.snippets),
            metrics=metrics,
        )

    async def _decompose_raw_text(self, inp: AdapterDecomposeInput) -> RawDecomposeText:
        t0 = time.monotonic()
        query = inp.query or ""
        ctx = await self._retrieve(query=query, repos=inp.repos)
        block = self._trim_block(ctx.grounding_block)
        prompt = (
            DECOMPOSE_PREAMBLE.format(model_tag=self.name)
            + query_blob(inp)
            + "\n\n"
            + self._context_block(block)
        )
        gen = await ollama_chat(
            self.settings,
            system=_DECOMPOSE_SYSTEM,
            user=prompt,
            num_ctx=self._ollama_num_ctx(),
        )
        metrics = self._metrics(t0=t0, ctx=ctx, gen=gen)
        metrics.extra["experimental_max_context"] = True
        metrics.extra["context_chars"] = len(block)
        return RawDecomposeText(
            text=(gen.get("content") or ""),
            metrics=metrics,
        )
