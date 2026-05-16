"""CheapEnricher: single Flash-tier LLM call that rewrites a query into an
``EnrichedQuery``. Wraps the existing ``enrich_query`` function.

The native ``EnrichedQuery`` (with summary, entities, code_keywords,
suspected_repos, etc.) is preserved in ``EnrichedQuestion.extra["enriched_query"]``
so downstream stages (DecomposeEngine) can pull the structured fields back out.
"""
from __future__ import annotations

from ..core.context import RunContext
from ..core.protocols import EnrichedQuestion, LoadedInput
from ._cheap_impl import enrich_query
from ..core.models import estimate_cost_usd
from ..core.usage import usage_from_result


class CheapEnricher:
    name = "enrich_cheap"

    async def enrich(self, loaded: LoadedInput, ctx: RunContext) -> EnrichedQuestion:
        query_text = f"{loaded.title}\n{loaded.body}".strip()
        repos = list(ctx.settings.repos)
        result = await enrich_query(query_text, repos, ctx.settings)
        enriched = result.output
        in_tok, out_tok = usage_from_result(result)
        cost = estimate_cost_usd(ctx.settings.enrich_model, in_tok or 0, out_tok or 0)
        return EnrichedQuestion(
            question=query_text,
            intent=enriched.intent,
            code_keywords=list(enriched.code_keywords),
            search_queries=list(enriched.search_queries),
            suspected_repos=list(enriched.suspected_repos),
            confidence=enriched.confidence,
            extra={
                "summary": enriched.summary,
                "enriched_query": enriched.model_dump(mode="json"),
                "query": query_text,
                "stage_metrics": {
                    "model": ctx.settings.enrich_model,
                    "input_tokens": in_tok,
                    "output_tokens": out_tok,
                    "cost_usd": cost,
                },
            },
        )
