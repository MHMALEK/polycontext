"""CheapEnricher: single Flash-tier LLM call that rewrites a ticket into an
``EnrichedQuery``. Wraps the existing ``enrich_ticket`` function.

The native ``EnrichedQuery`` (with summary, entities, code_keywords,
suspected_repos, etc.) is preserved in ``EnrichedQuestion.extra["enriched_query"]``
so downstream stages (DecomposeEngine) can pull the structured fields back out.
"""
from __future__ import annotations

from ..core.context import RunContext
from ..core.protocols import EnrichedQuestion, LoadedInput
from ._cheap_impl import enrich_ticket
from ..core.models import estimate_cost_usd
from ..core.usage import usage_from_result
from ..models import Ticket


class CheapEnricher:
    name = "enrich_cheap"

    async def enrich(self, loaded: LoadedInput, ctx: RunContext) -> EnrichedQuestion:
        ticket = Ticket(**loaded.metadata.get("ticket", {})) if loaded.metadata.get("ticket") else Ticket(
            title=loaded.title or "", body=loaded.body,
            key=loaded.metadata.get("key"), url=loaded.metadata.get("url"),
        )
        repos = list(ctx.settings.repos)
        result = await enrich_ticket(ticket, repos, ctx.settings)
        enriched = result.output
        in_tok, out_tok = usage_from_result(result)
        cost = estimate_cost_usd(ctx.settings.enrich_model, in_tok or 0, out_tok or 0)
        return EnrichedQuestion(
            question=ticket.body,
            intent=enriched.intent,
            code_keywords=list(enriched.code_keywords),
            search_queries=list(enriched.search_queries),
            suspected_repos=list(enriched.suspected_repos),
            confidence=enriched.confidence,
            extra={
                "summary": enriched.summary,
                "enriched_query": enriched.model_dump(mode="json"),
                "ticket": ticket.model_dump(mode="json"),
                "stage_metrics": {
                    "model": ctx.settings.enrich_model,
                    "input_tokens": in_tok,
                    "output_tokens": out_tok,
                    "cost_usd": cost,
                },
            },
        )
