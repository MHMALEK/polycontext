"""Baseline adapter: the existing tech-decomposition pipelines, behind the new interface.

This is what gives the bake-off an apples-to-apples reference point — every
other adapter is compared against what tech-decomposition already does today.

``ask`` routes through ``build_ask_pipeline`` (Sourcebot + local fallback).
``decompose`` routes through ``build_decompose_pipeline`` (Gemini Flash enrich
+ Gemini Pro decompose, with auto-escalation to the agentic deep engine).
``implement`` is intentionally not supported — the baseline pipeline only
reads code, never writes it. The bake-off router surfaces this as 501.
"""
from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any

from ..core.context import RunContext
from ..core.factory import build_ask_pipeline, build_decompose_pipeline, build_metrics_observer
from ..models import Decomposition
from .base import (
    Adapter,
    AdapterAskInput,
    AdapterAskResult,
    AdapterDecomposeInput,
    AdapterDecomposeResult,
    AdapterMetrics,
    Capability,
)


def _engine_to_metrics(er, total_seconds: float) -> AdapterMetrics:
    return AdapterMetrics(
        duration_ms=int(total_seconds * 1000),
        tokens_in=er.input_tokens,
        tokens_out=er.output_tokens,
        cost_usd=er.cost_usd,
        model=er.model,
        tool_calls=int(er.extra.get("tool_calls", 0) or 0),
        extra={"transport": er.transport, "inner_engine": er.engine},
    )


class BaselineAdapter(Adapter):
    name = "baseline"
    capabilities: set[Capability] = {"ask", "decompose"}
    description = "Current tech-decomposition pipeline (Sourcebot Q&A + Gemini decompose)."

    async def ask(self, inp: AdapterAskInput) -> AdapterAskResult:
        pipeline = build_ask_pipeline(
            self.settings,
            engine="sourcebot",
            include_cli_sink=False,
            include_file_sink=False,
        )
        obs = build_metrics_observer(self.settings, mode="ask")
        ctx = RunContext(settings=self.settings, mode="ask", metrics=obs)
        t = time.monotonic()
        run = await pipeline.run(inp.query, ctx)
        elapsed = time.monotonic() - t

        er = run.engine_result
        # The pipeline returns citations as plain dicts (Snippet-shaped or not
        # depending on the engine). Coerce via the Snippet model where possible
        # so the API contract is honored.
        from ..models import Snippet

        citations = []
        for c in er.citations or []:
            try:
                citations.append(Snippet.model_validate(c))
            except Exception:  # noqa: BLE001 — engines may emit looser shapes
                continue
        return AdapterAskResult(
            adapter=self.name,
            answer=er.answer_markdown,
            citations=citations,
            metrics=_engine_to_metrics(er, elapsed),
        )

    async def decompose(self, inp: AdapterDecomposeInput) -> AdapterDecomposeResult:
        # Mirror the existing /decompose endpoint's branching on ticket_text vs key/url.
        ref: dict[str, Any] | str
        if inp.ticket_text:
            tmp = Path(tempfile.mkstemp(suffix=".txt")[1])
            tmp.write_text(inp.ticket_text)
            source = "text_file"
            ref = {"path": str(tmp), "key": inp.ticket_key, "url": inp.ticket_url}
        else:
            source = "jira"
            ref = {"key": inp.ticket_key, "url": inp.ticket_url}

        pipeline = build_decompose_pipeline(
            self.settings,
            source=source,  # type: ignore[arg-type]
            mode=inp.mode,
            repos=inp.repos,
            post_to_jira=False,
            include_cli_sink=False,
            include_file_sink=False,
        )
        obs = build_metrics_observer(self.settings, mode="decompose")
        ctx = RunContext(settings=self.settings, mode="decompose", metrics=obs)
        t = time.monotonic()
        run = await pipeline.run(ref, ctx)
        elapsed = time.monotonic() - t

        er = run.engine_result
        decomp_dict = er.payload.get("decomposition") or {}
        try:
            decomp = Decomposition.model_validate(decomp_dict)
        except Exception as e:
            # Pipeline returned an unexpected shape — surface the markdown but
            # raise so the caller knows the contract was violated, rather than
            # silently returning an empty Decomposition.
            raise RuntimeError(
                f"baseline decompose returned no usable Decomposition: {e}"
            ) from e

        return AdapterDecomposeResult(
            adapter=self.name,
            decomposition=decomp,
            markdown=er.answer_markdown,
            metrics=_engine_to_metrics(er, elapsed),
        )
