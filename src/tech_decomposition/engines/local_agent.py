"""LocalAgentEngine: wraps ``local_ask`` (the in-process pydantic-ai agent)."""
from __future__ import annotations

from ..core.context import RunContext
from ..core.models import estimate_cost_usd
from ..core.protocols import EngineResult, EnrichedQuestion
from ._local_agent_impl import local_ask
from ..core.usage import usage_from_result


class LocalAgentEngine:
    name = "local_agent"

    async def run(self, q: EnrichedQuestion, ctx: RunContext) -> EngineResult:
        result, deps = await local_ask(q.question, ctx.settings)
        answer = result.output
        in_tok, out_tok = usage_from_result(result)
        model = ctx.settings.decompose_model
        cost = estimate_cost_usd(model, in_tok or 0, out_tok or 0)
        return EngineResult(
            engine=self.name,
            answer_markdown=answer.answer,
            citations=[c.model_dump(mode="json") for c in answer.citations],
            model=model,
            transport="local",
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=cost,
            extra={
                "confidence": answer.confidence,
                "open_questions": list(answer.open_questions),
                "tool_calls": dict(deps.tool_calls),
            },
        )
