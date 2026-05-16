"""DeepDecomposeEngine: agentic deep mode ticket → subtasks. Experimental.

Wraps the existing ``deep_decompose`` agentic loop. Use sparingly — Pro-tier
with tools costs significantly more than the cheap path.
"""
from __future__ import annotations

from ._contradiction import check_contradictions
from ..core.context import RunContext
from ..core.protocols import EngineResult, EnrichedQuestion
from ._deep_decompose_impl import deep_decompose
from ..core.llm_registry import estimate_cost_usd
from ..core.usage import usage_from_result
from ..models import EnrichedQuery, RepoContext, RetrievedContext
from ._decompose_render import attach_gitlab_links, render_markdown
from ..retrievers import gather_anchor_snippets


class DeepDecomposeEngine:
    name = "decompose_deep"

    def __init__(self, *, repos: list[str] | None = None):
        self.repos = repos

    async def run(self, q: EnrichedQuestion, ctx: RunContext) -> EngineResult:
        eq_data = q.extra.get("enriched_query")
        ticket_data = q.extra.get("ticket") or {}
        if not eq_data or not ticket_data:
            raise RuntimeError(
                "DeepDecomposeEngine requires Enricher output with enriched_query + ticket"
            )
        enriched = EnrichedQuery(**eq_data)
        ticket = Ticket(**ticket_data)
        repos = self.repos or list(ctx.settings.repos)

        # Anchors-only context (Pro retrieves the rest via tools).
        anchor_ctx_map = gather_anchor_snippets(ticket, ctx.settings)
        repo_ctxs = [
            anchor_ctx_map.get(r) or RepoContext(repo=r, head_sha="HEAD", snippets=[])
            for r in repos
        ]
        context = RetrievedContext(
            repos=repo_ctxs,
            total_snippets=sum(len(rc.snippets) for rc in repo_ctxs),
            total_chars=sum(len(s.content) for rc in repo_ctxs for s in rc.snippets),
        )

        result, deep_deps = await deep_decompose(
            ticket=ticket, enriched=enriched, settings=ctx.settings,
        )
        decomp = result.output
        in_tok, out_tok = usage_from_result(result)
        cost = estimate_cost_usd(ctx.settings.decompose_model, in_tok or 0, out_tok or 0)

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
            transport="local-agent",
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=cost,
            extra={
                "ticket_key": ticket.key,
                "ticket_url": ticket.url,
                "affected_repos": list(decomp.affected_repos),
                "subtask_count": len(decomp.subtasks),
                "tool_calls": dict(deep_deps.tool_calls),
            },
        )
