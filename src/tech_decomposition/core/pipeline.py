"""Pipeline: composes stage strategies and runs them in order.

The pipeline is intentionally dumb. It owns five Protocol slots
(input/enrich/engine/render/sinks), iterates them, times each, forwards
metrics, and returns the final ``EngineResult`` plus the list of
``SinkResult``. Branching by mode (ask vs decompose) is done by picking
*which* strategies you compose, not by adding logic here.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from .context import RunContext, StageResult
from .protocols import (
    EngineResult,
    Enricher,
    EnrichedQuestion,
    Engine,
    InputSource,
    LoadedInput,
    Rendered,
    Renderer,
    Sink,
    SinkResult,
)


@dataclass
class PipelineRun:
    """Aggregate return value of ``Pipeline.run``.

    Carries every stage's payload so callers can inspect intermediate state
    (handy in CLI / tests) without re-running.
    """

    loaded: LoadedInput
    enriched: EnrichedQuestion
    engine_result: EngineResult
    rendered: list[Rendered]
    sink_results: list[SinkResult]
    total_seconds: float


@dataclass
class Pipeline:
    input_source: InputSource
    enricher: Enricher | None
    engine: Engine
    renderers: list[Renderer]
    sinks: list[Sink]

    async def run(self, ref: str | dict[str, Any], ctx: RunContext) -> PipelineRun:
        t_run = time.monotonic()

        ctx.stage_started(stage="input", strategy=self.input_source.__class__.__name__)
        loaded = await self._timed("input", self.input_source, lambda: self.input_source.load(ref, ctx), ctx)
        # Capture a short preview of the user's input so the run row can carry
        # context without bloating the JSONL with the full body.
        if ctx.input_preview is None:
            preview = (loaded.title or loaded.body or "").strip().splitlines()
            ctx.input_preview = (preview[0][:160] if preview else "")

        if self.enricher is None:
            enriched = EnrichedQuestion(question=loaded.body)
            ctx.record_stage(StageResult(stage="enrich", strategy="(none)", seconds=0.0))
        else:
            ctx.stage_started(stage="enrich", strategy=self.enricher.__class__.__name__)
            enriched = await self._timed(
                "enrich", self.enricher, lambda: self.enricher.enrich(loaded, ctx), ctx
            )

        ctx.stage_started(stage="engine", strategy=self.engine.__class__.__name__)
        engine_result = await self._timed(
            "engine", self.engine, lambda: self.engine.run(enriched, ctx), ctx,
            extra_metrics={
                "model": lambda r: r.model,
                "input_tokens": lambda r: r.input_tokens,
                "output_tokens": lambda r: r.output_tokens,
                "cost_usd": lambda r: r.cost_usd,
            },
        )

        rendered_list: list[Rendered] = []
        for renderer in self.renderers:
            ctx.stage_started(stage="render", strategy=renderer.name)
            t = time.monotonic()
            rendered = renderer.render(engine_result, ctx)
            ctx.record_stage(StageResult(
                stage="render", strategy=renderer.name,
                seconds=round(time.monotonic() - t, 4),
            ))
            rendered_list.append(rendered)

        # Sinks consume the first rendered output by default; specialized sinks
        # (e.g. JiraADFSink) can pick a different format from the list.
        primary = rendered_list[0] if rendered_list else Rendered(format="markdown", body=engine_result.answer_markdown)
        sink_results: list[SinkResult] = []
        for sink in self.sinks:
            ctx.stage_started(stage="sink", strategy=sink.name)
            t = time.monotonic()
            chosen = _pick_rendered_for_sink(sink, rendered_list, primary)
            res = await sink.deliver(chosen, ctx)
            ctx.record_stage(StageResult(
                stage="sink", strategy=sink.name,
                seconds=round(time.monotonic() - t, 4),
            ))
            sink_results.append(res)

        total = round(time.monotonic() - t_run, 4)
        if ctx.metrics is not None:
            ctx.metrics.on_run_complete(
                run_id=ctx.run_id,
                total_seconds=total,
                total_cost_usd=engine_result.cost_usd,
                extra={
                    "input_preview": ctx.input_preview,
                    "engine": engine_result.engine,
                    "model": engine_result.model,
                },
            )

        return PipelineRun(
            loaded=loaded,
            enriched=enriched,
            engine_result=engine_result,
            rendered=rendered_list,
            sink_results=sink_results,
            total_seconds=total,
        )

    async def _timed(
        self,
        stage: str,
        strategy: Any,
        thunk,
        ctx: RunContext,
        *,
        extra_metrics: dict[str, Any] | None = None,
    ):
        t = time.monotonic()
        result = await thunk()
        seconds = round(time.monotonic() - t, 4)
        sr = StageResult(stage=stage, strategy=getattr(strategy, "name", strategy.__class__.__name__), seconds=seconds)
        if extra_metrics:
            for k, getter in extra_metrics.items():
                try:
                    setattr(sr, k, getter(result))
                except Exception:
                    pass
        ctx.record_stage(sr)
        return result


def _pick_rendered_for_sink(sink: Sink, all_rendered: list[Rendered], default: Rendered) -> Rendered:
    """Sinks can declare a preferred format via a ``prefers`` attribute."""
    pref = getattr(sink, "prefers", None)
    if pref:
        for r in all_rendered:
            if r.format == pref:
                return r
    return default
