"""DecomposeEngine: cheap-mode ticket → subtasks pipeline as an Engine.

Wraps the existing retrieval → decompose → contradiction-check chain so the
new Pipeline can run a decomposition the same way it runs an ask.

Deep mode lives in ``deep_decompose.py`` engine wrapper since it's the
experimental agentic path.
"""
from __future__ import annotations

from ._contradiction import check_contradictions
from ..core.context import RunContext
from ..core.protocols import EngineResult, EnrichedQuestion
from ._decompose_impl import decompose
from ..core.models import estimate_cost_usd
from ..core.usage import usage_from_result
from ..models import EnrichedQuery
from ._decompose_render import attach_gitlab_links, render_markdown
from ..retrievers import gather_context


class DecomposeEngine:
    name = "decompose_cheap"

    def __init__(self, *, repos: list[str] | None = None):
        self.repos = repos

    async def run(self, q: EnrichedQuestion, ctx: RunContext) -> EngineResult:
        eq_data = q.extra.get("enriched_query")
        if not eq_data:
            raise RuntimeError(
                "DecomposeEngine requires an Enricher (CheapEnricher) that populates "
                "extra['enriched_query']"
            )
        enriched = EnrichedQuery(**eq_data)
        ticket_data = q.extra.get("ticket") or {}
        from ..models import Ticket
        ticket = Ticket(**ticket_data) if ticket_data else None
        repos = self.repos or list(ctx.settings.repos)

        context = await gather_context(repos, enriched, ctx.settings, ticket=ticket)

        result = await decompose(
            ticket=ticket, query=enriched, context=context, settings=ctx.settings,
        )
        decomp = result.output
        in_tok, out_tok = usage_from_result(result)
        cost = estimate_cost_usd(ctx.settings.decompose_model, in_tok or 0, out_tok or 0)

        # Contradiction check is part of the decompose stage's output; if it
        # fails we log but don't abort the run.
        try:
            cc = await check_contradictions(ticket=ticket, decomp=decomp, settings=ctx.settings)
            for c in (cc.output.contradictions or []):
                where = f"subtask #{c.subtask_index}" if c.subtask_index else "overall"
                decomp.risks.append(
                    f"[contradiction-check / {c.severity}] {where}: {c.issue} "
                    f'(ticket says: "{c.ticket_excerpt}")'
                )
        except Exception:
            import logging
            logging.getLogger(__name__).warning("contradiction check failed", exc_info=True)

        decomp = attach_gitlab_links(decomp, context, ctx.settings)
        markdown = render_markdown(decomp=decomp, enriched=enriched, ctx=context, settings=ctx.settings)

        return EngineResult(
            engine=self.name,
            answer_markdown=markdown,
            payload={"decomposition": decomp.model_dump(mode="json")},
            model=ctx.settings.decompose_model,
            transport="local-llm",
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=cost,
            extra={
                "ticket_key": getattr(ticket, "key", None),
                "ticket_url": getattr(ticket, "url", None),
                "affected_repos": list(decomp.affected_repos),
                "subtask_count": len(decomp.subtasks),
                "retrieval": {
                    "total_snippets": context.total_snippets,
                    "total_chars": context.total_chars,
                },
            },
        )
