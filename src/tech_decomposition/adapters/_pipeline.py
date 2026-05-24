"""Tiered pipeline adapter: router → prefetch → synthesize → structurer → verify.

Default path is cheap: Sourcebot + Serena retrieval, one Flash synthesis call,
Flash structurer for decompose. Falls back to the configured agent adapter
(``pipeline_fallback_adapter``, default ``gemini``) when retrieval coverage is
low or the router marks the task as trace-heavy.
"""
from __future__ import annotations

import time
from typing import Any

from ..core.coverage import assess_coverage, snippet_paths
from ..core.grounding import build_grounded_prompt, retrieve_grounded_context
from ..core.router import (
    RouteDecision,
    TaskTier,
    answer_signals_insufficient,
    classify_task,
    retrieval_boost_terms,
    should_agent_fallback,
    should_escalate_fallback,
    should_escalate_synthesis,
)
from ..core.synthesize import escalation_model_name, synthesize_ask, synthesize_decompose_draft
from ..core.verifier import apply_verifier_warnings, verify_decomposition
from ._prompts import query_blob
from .base import (
    Adapter,
    AdapterAskInput,
    AdapterAskResult,
    AdapterDecomposeInput,
    AdapterMetrics,
    Capability,
    RawDecomposeText,
)


def _synthesis_model_id(
    settings,
    *,
    tags: list[str] | None,
    coverage_sufficient: bool,
) -> str | None:
    """Return Pro model id when synthesis should escalate; else default (Flash)."""
    if should_escalate_synthesis(tags=tags, coverage_sufficient=coverage_sufficient):
        return escalation_model_name(settings)
    return None


def _synthesis_path_label(*, model_id: str | None, escalated: bool) -> str:
    if escalated:
        return "prefetch+pro_synthesis"
    return "prefetch+synthesis"


class PipelineAdapter(Adapter):
    name = "pipeline"
    capabilities: set[Capability] = {"ask", "decompose"}
    description = (
        "Tiered retrieve-first pipeline: Sourcebot+Serena prefetch, Flash "
        "single-shot synthesis, optional bounded agent fallback."
    )

    def health(self) -> dict:
        if not self.settings.gemini_api_key:
            return {"ok": False, "reason": "GEMINI_API_KEY required for synthesis/structurer"}
        if not (self.settings.sourcebot_url and self.settings.sourcebot_api_key):
            return {"ok": False, "reason": "SOURCEBOT_URL and SOURCEBOT_API_KEY required for prefetch"}
        fb = (self.settings.pipeline_fallback_adapter or "gemini").strip()
        try:
            fb_adapter = self._fallback_adapter()
            fb_health = fb_adapter.health()
            if not fb_health.get("ok"):
                return {
                    "ok": False,
                    "reason": f"fallback adapter {fb!r} unhealthy: {fb_health.get('reason', '?')}",
                }
        except Exception as e:
            return {"ok": False, "reason": f"fallback adapter {fb!r}: {e}"}
        return {"ok": True}

    async def ask(self, inp: AdapterAskInput) -> AdapterAskResult:
        t0 = time.monotonic()
        route = classify_task(query=inp.query, job="ask", tags=inp.tags)
        extra: dict[str, Any] = {"pipeline_route": route.tier.value, "pipeline_reason": route.reason}

        broad = route.tier in (TaskTier.ENUMERATION, TaskTier.COMPLEX, TaskTier.TRACE)
        boost = retrieval_boost_terms(inp.query, inp.tags)
        ctx_lines = 10 if route.tier == TaskTier.ENUMERATION else 6

        grounding = await retrieve_grounded_context(
            query=inp.query,
            settings=self.settings,
            repos=inp.repos,
            top_k=max(inp.top_k, route.prefetch_top_k),
            context_lines=ctx_lines,
            retrieval_mode="broad" if broad else "default",
            extra_terms=boost or None,
        )
        extra["grounding"] = grounding.metrics.model_dump()
        extra["grounding_paths"] = sorted(snippet_paths(grounding.snippets))
        coverage = assess_coverage(
            grounding,
            route,
            min_snippets=self.settings.pipeline_min_snippets,
            min_chars=self.settings.pipeline_min_chars,
        )
        extra["coverage"] = {
            "score": coverage.score,
            "sufficient": coverage.sufficient,
            "reason": coverage.reason,
        }

        synth_model = _synthesis_model_id(
            self.settings,
            tags=inp.tags,
            coverage_sufficient=coverage.sufficient,
        )
        synth = await synthesize_ask(
            query=inp.query,
            grounding_block=grounding.grounding_block or "(no snippets retrieved)",
            settings=self.settings,
            model_id=synth_model,
        )
        extra["pipeline_path"] = _synthesis_path_label(
            model_id=synth_model,
            escalated=bool(synth_model),
        )
        extra["synthesis_model"] = synth.model
        extra["synthesis_insufficient"] = answer_signals_insufficient(synth.text)

        if should_agent_fallback(
            route=route,
            tags=inp.tags,
            coverage_sufficient=coverage.sufficient,
            synthesis_text=synth.text,
        ):
            model_id, max_rounds = self._fallback_model_config(route, inp.tags)
            extra["pipeline_path"] = "synthesis→agent_fallback"
            extra["fallback_model"] = model_id
            extra["fallback_max_tool_rounds"] = max_rounds
            fb_query = (
                build_grounded_prompt(
                    grounding_block=grounding.grounding_block, query=inp.query,
                )
                if grounding.grounding_block.strip()
                else inp.query
            )
            fb_inp = inp.model_copy(update={"query": fb_query})
            fb = await self._run_fallback_ask(fb_inp, model_id=model_id, max_tool_rounds=max_rounds)
            if (fb.answer or "").strip():
                fb.metrics.extra = {**(fb.metrics.extra or {}), **extra}
                fb.metrics.duration_ms = int((time.monotonic() - t0) * 1000)
                return fb.model_copy(update={"adapter": self.name})
            extra["fallback_tool_calls"] = fb.metrics.tool_calls
            extra["pipeline_path"] = "agent_fallback_empty→synthesis"

        return AdapterAskResult(
            adapter=self.name,
            answer=synth.text,
            citations=[],
            metrics=AdapterMetrics(
                duration_ms=int((time.monotonic() - t0) * 1000),
                tokens_in=synth.tokens_in,
                tokens_out=synth.tokens_out,
                model=synth.model,
                tool_calls=0,
                extra=extra,
            ),
        )

    async def _decompose_raw_text(self, inp: AdapterDecomposeInput) -> RawDecomposeText:
        t0 = time.monotonic()
        query = query_blob(inp)
        route = classify_task(query=query, job="decompose", tags=inp.tags)
        extra: dict[str, Any] = {"pipeline_route": route.tier.value, "pipeline_reason": route.reason}

        boost = retrieval_boost_terms(query, inp.tags)
        grounding = await retrieve_grounded_context(
            query=query,
            settings=self.settings,
            repos=inp.repos,
            top_k=route.prefetch_top_k,
            context_lines=6,
            retrieval_mode="broad",
            extra_terms=boost or None,
        )
        extra["grounding"] = grounding.metrics.model_dump()
        extra["grounding_paths"] = sorted(snippet_paths(grounding.snippets))
        coverage = assess_coverage(
            grounding,
            route,
            min_snippets=self.settings.pipeline_min_snippets,
            min_chars=self.settings.pipeline_min_chars,
        )
        extra["coverage"] = {
            "score": coverage.score,
            "sufficient": coverage.sufficient,
            "reason": coverage.reason,
        }

        synth_model = _synthesis_model_id(
            self.settings,
            tags=inp.tags,
            coverage_sufficient=coverage.sufficient,
        )
        synth = await synthesize_decompose_draft(
            query=query,
            grounding_block=grounding.grounding_block or "(no snippets retrieved)",
            settings=self.settings,
            model_id=synth_model,
        )
        extra["pipeline_path"] = _synthesis_path_label(
            model_id=synth_model,
            escalated=bool(synth_model),
        )
        extra["synthesis_model"] = synth.model

        if should_agent_fallback(
            route=route,
            tags=inp.tags,
            coverage_sufficient=coverage.sufficient,
            synthesis_text=synth.text,
        ):
            model_id, max_rounds = self._fallback_model_config(route, inp.tags)
            extra["pipeline_path"] = "synthesis→agent_fallback"
            extra["fallback_model"] = model_id
            fb = await self._run_fallback_decompose(inp, model_id=model_id, max_tool_rounds=max_rounds)
            if (fb.text or "").strip():
                fb.metrics.extra = {**(fb.metrics.extra or {}), **extra}
                fb.metrics.duration_ms = int((time.monotonic() - t0) * 1000)
                return fb
            extra["fallback_tool_calls"] = fb.metrics.tool_calls
            extra["pipeline_path"] = "agent_fallback_empty→draft"

        return RawDecomposeText(
            text=synth.text,
            metrics=AdapterMetrics(
                duration_ms=int((time.monotonic() - t0) * 1000),
                tokens_in=synth.tokens_in,
                tokens_out=synth.tokens_out,
                model=synth.model,
                tool_calls=0,
                extra=extra,
            ),
        )

    async def decompose(self, inp: AdapterDecomposeInput):
        """Run base decompose then apply deterministic path verification."""
        from ..core.decomposition_structurer import structure_decomposition

        raw = await self._decompose_raw_text(inp)
        structured = await structure_decomposition(
            text=raw.text,
            query=(inp.query or ""),
            settings=self.settings,
            model_tag=self.name,
        )
        vr = verify_decomposition(self.settings, structured.decomposition)
        decomp = apply_verifier_warnings(structured.decomposition, vr)

        extra = dict(raw.metrics.extra or {})
        extra["verifier_ok"] = vr.ok
        extra["verifier_missing"] = vr.missing_files[:10]
        extra["structurer_notes"] = structured.notes

        from .base import AdapterDecomposeResult

        return AdapterDecomposeResult(
            adapter=self.name,
            decomposition=decomp,
            markdown=decomp.to_markdown(),
            metrics=AdapterMetrics(
                duration_ms=raw.metrics.duration_ms,
                tokens_in=raw.metrics.tokens_in,
                tokens_out=raw.metrics.tokens_out,
                cost_usd=raw.metrics.cost_usd,
                model=raw.metrics.model,
                tool_calls=raw.metrics.tool_calls,
                extra=extra,
            ),
        )

    def _fallback_model_config(
        self, route: RouteDecision, tags: list[str] | None,
    ) -> tuple[str, int]:
        escalate = should_escalate_fallback(route=route, tags=tags)
        if escalate:
            raw = (self.settings.pipeline_escalation_model or "gemini-2.5-pro").strip()
        else:
            raw = (self.settings.pipeline_fallback_model or "gemini-2.5-flash").strip()
        model_id = raw.split(":", 1)[-1].strip() if ":" in raw else raw
        return model_id, int(self.settings.pipeline_max_tool_rounds)

    async def _run_fallback_ask(
        self,
        inp: AdapterAskInput,
        *,
        model_id: str,
        max_tool_rounds: int,
    ) -> AdapterAskResult:
        from ._gemini import GeminiAdapter

        ga = GeminiAdapter(self.settings)
        return await ga.ask_configured(
            inp, model_id=model_id, max_tool_rounds=max_tool_rounds,
        )

    async def _run_fallback_decompose(
        self,
        inp: AdapterDecomposeInput,
        *,
        model_id: str,
        max_tool_rounds: int,
    ) -> RawDecomposeText:
        from ._gemini import GeminiAdapter

        ga = GeminiAdapter(self.settings)
        return await ga.decompose_raw_configured(
            inp, model_id=model_id, max_tool_rounds=max_tool_rounds,
        )

    def _fallback_adapter(self) -> Adapter:
        from .registry import get_adapter

        name = (self.settings.pipeline_fallback_adapter or "gemini").strip()
        fb_model = (self.settings.pipeline_fallback_model or "").strip()
        if name == "gemini" and fb_model:
            fb_settings = self.settings.model_copy(update={"gemini_sdk_model": fb_model})
            return get_adapter(name, fb_settings)
        return get_adapter(name, self.settings)
