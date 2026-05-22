"""Sourcebot retrieval + local Ollama generation (RAG, free-local answers).

Why this exists
---------------
The bake-off showed two clean facts:

  1. Sourcebot's retrieval is excellent — the ``sourcebot`` adapter
     (Sourcebot ``/api/chat/blocking`` driving Gemini Flash) scored 0.806
     on 13 ask cases.
  2. Free-local generation via Cline + qwen2.5:14b scored only 0.572 —
     not because the model is bad, but because it picks weak search
     queries (e.g. ``USER_ROLES`` when the enum is actually
     ``UserRoles``) and gets little help from its own tool calls.

This adapter pairs Sourcebot's strong retrieval with a local model's
free generation. Tool calls are off the table — the model only ever
sees a packed grounding block and is asked to write an answer.

Pipeline
--------
1. ``core.grounding.retrieve_grounded_context(query, settings)``
   → typed snippets + prompt-ready markdown ``grounding_block``.
   Search-term extraction uses regex first, falls back to a small
   Gemini Flash classifier ONLY when regex terms are weak (most code
   questions land in the regex path = $0 per call).
2. Pack the grounding block + question into a strict system prompt
   ("answer ONLY from the snippets; cite paths") and POST to
   ``OLLAMA_BASE_URL/api/chat`` with the model from ``OLLAMA_MODEL``.
3. Return the answer plus citations carried over from snippet metadata.

Cost
----
- Per call: ~$0.0001 (occasional Flash classifier) or $0 (most calls).
- Generation: $0 (local Ollama).
"""
from __future__ import annotations

import time
from pathlib import Path  # noqa: F401  (used by signature parity with other adapters)
from typing import Any

from ..core.grounding import retrieve_grounded_context
from ._ollama_http import ollama_chat, ollama_health_check
from ._prompts import DECOMPOSE_PREAMBLE, query_blob
from .base import (
    Adapter,
    AdapterAskInput,
    AdapterAskResult,
    AdapterDecomposeInput,
    AdapterMetrics,
    Capability,
    RawDecomposeText,
    Snippet,
)


_ASK_SYSTEM = (
    "You are a code Q&A assistant. Answer using ONLY the code snippets in the "
    "CONTEXT block below.\n\n"
    "Rules:\n"
    "- Every factual claim must appear in a snippet (identifiers, enums, "
    "constants, function behavior, file paths).\n"
    "- Prefer primary definitions (models, constants, enums, route handlers) "
    "over tests, mocks, or tangential utilities unless the question is about tests.\n"
    "- Use exact identifiers as written in the snippets (case-sensitive).\n"
    "- Structure the answer: short overview, then bullets or a numbered list "
    "for enumerations (roles, validators, steps, etc.).\n"
    "- If snippets are insufficient, say what is missing — do NOT invent from "
    "training knowledge.\n"
    "- Cite paths as shown above each snippet (``path/to/file.py:LINE-LINE``)."
)

_DECOMPOSE_SYSTEM = (
    "You decompose engineering tickets into structured tech work using ONLY "
    "the CONTEXT snippets below.\n\n"
    "Rules:\n"
    "- Subtasks and file paths must be grounded in snippets; do not invent paths.\n"
    "- Prefer touching repos/files that clearly relate to the ticket.\n"
    "- Output exactly one JSON object per the user schema — no prose, no fences."
)


class SourcebotOllamaAdapter(Adapter):
    name = "sourcebot_ollama"
    capabilities: set[Capability] = {"ask", "decompose"}
    description = (
        "Sourcebot (+ optional Serena) retrieval → Ollama generation. No agent "
        "tools; the model only reads a packed CONTEXT block. Set SERENA_URL for "
        "LSP snippets. Generation: free on local Ollama or Ollama Cloud free tier."
    )

    def health(self) -> dict:
        # Sourcebot must be reachable so we can retrieve.
        if not (self.settings.sourcebot_url or "").strip():
            return {"ok": False, "reason": "SOURCEBOT_URL not set"}
        if not self.settings.sourcebot_api_key:
            return {"ok": False, "reason": "SOURCEBOT_API_KEY not set"}
        ollama_h = ollama_health_check(self.settings)
        if not ollama_h.get("ok"):
            return ollama_h
        out = {"ok": True}
        if (self.settings.serena_url or "").strip():
            out["serena"] = "configured"
        else:
            out["serena"] = "unset (optional — set SERENA_URL for LSP snippets)"
        return out

    def _retrieval_top_k(self) -> int:
        return max(8, int(self.settings.retrieval_max_hits or 40))

    def _retrieval_context_lines(self) -> int:
        return max(3, int(self.settings.retrieval_snippet_lines or 8))

    async def ask(self, inp: AdapterAskInput) -> AdapterAskResult:
        t0 = time.monotonic()
        ctx = await retrieve_grounded_context(
            query=inp.query,
            settings=self.settings,
            repos=inp.repos,
            top_k=self._retrieval_top_k(),
            context_lines=self._retrieval_context_lines(),
        )
        prompt = self._ask_prompt(inp.query, ctx.grounding_block)
        gen = await ollama_chat(
            self.settings,
            system=_ASK_SYSTEM,
            user=prompt,
            num_ctx=32768,
        )
        return AdapterAskResult(
            adapter=self.name,
            answer=(gen.get("content") or "").strip(),
            citations=_snippet_to_citation(ctx.snippets),
            metrics=self._metrics(t0=t0, ctx=ctx, gen=gen),
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
        )
        prompt = (
            DECOMPOSE_PREAMBLE.format(model_tag=self.name)
            + query_blob(inp)
            + "\n\n"
            + self._context_block(ctx.grounding_block)
        )
        gen = await ollama_chat(
            self.settings,
            system=_DECOMPOSE_SYSTEM,
            user=prompt,
            num_ctx=32768,
        )
        return RawDecomposeText(
            text=(gen.get("content") or ""),
            metrics=self._metrics(t0=t0, ctx=ctx, gen=gen),
        )

    # ---- internals ---------------------------------------------------------

    def _ask_prompt(self, question: str, grounding_block: str) -> str:
        return (
            self._context_block(grounding_block)
            + "\n\nQuestion: " + question.strip()
        )

    @staticmethod
    def _context_block(grounding_block: str) -> str:
        gb = (grounding_block or "").strip()
        if not gb:
            return "CONTEXT: (no code snippets retrieved — say you cannot answer.)"
        return "CONTEXT (code snippets from the workspace):\n\n" + gb

    def _metrics(
        self,
        *,
        t0: float,
        ctx: Any,
        gen: dict[str, Any],
    ) -> AdapterMetrics:
        gm = getattr(ctx, "metrics", None)
        snippet_paths = [
            f"{getattr(s, 'repo', '') or ''}:{getattr(s, 'path', '')}"
            for s in (getattr(ctx, "snippets", None) or [])[:25]
            if getattr(s, "path", None)
        ]
        extra: dict[str, Any] = {
            "grounding_snippets": getattr(gm, "snippet_count", None)
            if gm is not None
            else len(getattr(ctx, "snippets", None) or []),
            "grounding_sources": list(getattr(gm, "sources", []) or []),
            "grounding_extractor": getattr(gm, "extractor", None),
            "grounding_error": getattr(gm, "error", None),
            "grounding_classifier_ms": getattr(gm, "classifier_ms", 0),
            "grounding_search_query": getattr(gm, "search_query", None),
            "grounding_extracted_terms": list(getattr(gm, "extracted_terms", []) or []),
            "grounding_snippet_paths": snippet_paths,
            "ollama_model": self.settings.ollama_model,
        }
        return AdapterMetrics(
            duration_ms=int((time.monotonic() - t0) * 1000),
            tokens_in=gen.get("prompt_eval_count"),
            tokens_out=gen.get("eval_count"),
            model=self.settings.ollama_model,
            tool_calls=0,
            cost_usd=0.0,
            extra=extra,
        )


def _snippet_to_citation(snippets: list[Any]) -> list[Snippet]:
    # GroundingSnippet uses start_line/end_line + optional repo; the Snippet
    # model needed for citations uses line_start/line_end + required repo.
    out: list[Snippet] = []
    for s in snippets[:20]:
        try:
            out.append(
                Snippet(
                    repo=getattr(s, "repo", "") or "",
                    path=getattr(s, "path", "") or "",
                    line_start=int(getattr(s, "start_line", 1) or 1),
                    line_end=int(getattr(s, "end_line", 1) or 1),
                    content=getattr(s, "content", "") or "",
                    source="sourcebot",
                )
            )
        except Exception:  # noqa: BLE001 — citations are best-effort
            continue
    return out
