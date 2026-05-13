"""StructuredEngine: composes any Engine with a Structurer to produce typed output.

Pattern:
    inner: Engine  ->  raw markdown answer
    structurer     ->  StructuredAnswer (typed, narration-stripped, cited)

The wrapper's ``run`` calls inner.run, hands the markdown to the structurer,
and emits an EngineResult whose ``answer_markdown`` is a re-rendered clean
version and whose ``citations`` are typed dicts (no regex parsing).

Use the wrapper transparently; the new shape is what downstream stages see.
"""
from __future__ import annotations

from ..core.context import RunContext, StageResult
from ..core.protocols import EngineResult, EnrichedQuestion
from ..core.structurer import PydanticAIStructurer, StructuredAnswer


def _render_structured_md(s: StructuredAnswer) -> str:
    """Render a StructuredAnswer back to the user-facing markdown shape."""
    out = [s.answer.strip()]
    if s.citations:
        out.append("")
        out.append("## Citations")
        for c in s.citations:
            label = f"{c.repo}/{c.path}" if c.repo else c.path
            suf = ""
            if c.start_line and c.end_line and c.end_line != c.start_line:
                suf = f":L{c.start_line}-L{c.end_line}"
            elif c.start_line:
                suf = f":L{c.start_line}"
            if c.url:
                out.append(f"- [{label}{suf}]({c.url})")
            else:
                out.append(f"- `{label}{suf}`")
    if s.open_questions:
        out.append("")
        out.append("## Open questions")
        for q in s.open_questions:
            out.append(f"- {q}")
    return "\n".join(out).rstrip() + "\n"


class StructuredEngine:
    """Wraps any Engine, adds a pydantic-ai structurer pass on the answer."""

    def __init__(self, inner, structurer: PydanticAIStructurer):
        self.inner = inner
        self.structurer = structurer
        self.name = f"{inner.name}+structured"

    async def run(self, q: EnrichedQuestion, ctx: RunContext) -> EngineResult:
        raw = await self.inner.run(q, ctx)
        if not raw.answer_markdown.strip():
            return raw  # nothing to structure

        import time
        t = time.monotonic()
        structured, in_tok, out_tok, cost = await self.structurer.structure(
            raw.answer_markdown, q.question,
        )
        # Record structurer as its own stage so observers see what it cost.
        ctx.record_stage(StageResult(
            stage="structure",
            strategy=self.structurer.name,
            seconds=round(time.monotonic() - t, 4),
            model=self.structurer.model_spec,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=cost,
        ))

        # Roll structurer cost into the engine cost so summary totals are honest.
        combined_in = (raw.input_tokens or 0) + (in_tok or 0) or None
        combined_out = (raw.output_tokens or 0) + (out_tok or 0) or None
        combined_cost = round((raw.cost_usd or 0) + (cost or 0), 6) if (raw.cost_usd or cost) else None

        return EngineResult(
            engine=self.name,
            answer_markdown=_render_structured_md(structured),
            payload={"structured_answer": structured.model_dump(mode="json")},
            citations=[c.model_dump(mode="json") for c in structured.citations],
            model=raw.model,
            transport=raw.transport,
            wall_seconds=raw.wall_seconds,
            input_tokens=combined_in,
            output_tokens=combined_out,
            cost_usd=combined_cost,
            extra={
                **raw.extra,
                "confidence": structured.confidence,
                "open_questions": list(structured.open_questions),
                "raw_answer_markdown": raw.answer_markdown,
            },
        )
