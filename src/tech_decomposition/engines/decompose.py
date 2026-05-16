"""DecomposeEngine: ticket → Sourcebot-grounded exploration → structured decomposition.

Single path: the same Sourcebot blocking chat used for ``ask`` furnishes grounded
context; a dedicated post-process LLM pass turns that answer into ``Decomposition``.
"""
from __future__ import annotations

from ..clients.sourcebot import AskResult
from ..models import EnrichedQuery, RepoContext, RetrievedContext, Ticket
from ._contradiction import check_contradictions
from ..core.context import RunContext
from ..core.protocols import EngineResult, EnrichedQuestion
from ._decompose_impl import run_decompose_pipeline
from ..core.llm_registry import estimate_cost_usd
from ..core.usage import usage_from_result
from ._decompose_render import attach_gitlab_links, render_markdown


def _ticket_from_question(q: EnrichedQuestion) -> Ticket:
    text = (q.question or "").strip()
    return Ticket(title=text[:500] or "(ticket)", body=text)


def _repo_keys_for_context(enriched: EnrichedQuery, fallback: list[str]) -> list[str]:
    seen: list[str] = []
    for r in (enriched.suspected_repos or []) or fallback:
        if r and r not in seen:
            seen.append(r)
    if not seen:
        return list(fallback)
    return seen


def _empty_sourcebot_context(repo_keys: list[str]) -> RetrievedContext:
    return RetrievedContext(
        repos=[RepoContext(repo=r, head_sha="HEAD", snippets=[]) for r in repo_keys],
        total_snippets=0,
        total_chars=0,
        grounding="sourcebot_chat",
    )


class DecomposeEngine:
    name = "decompose"

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
        ticket = Ticket(**ticket_data) if ticket_data else _ticket_from_question(q)
        if not (ticket.body or "").strip() and (q.question or "").strip():
            ticket = ticket.model_copy(update={"body": q.question})

        repos = self.repos or list(ctx.settings.repos)
        ask_repos = enriched.suspected_repos or repos

        if ctx.settings.grounding_require_sourcebot and not (
            ctx.settings.sourcebot_url and ctx.settings.sourcebot_api_key
        ):
            raise RuntimeError(
                "Decomposition requires Sourcebot: set SOURCEBOT_URL and SOURCEBOT_API_KEY "
                "(or set GROUNDING_REQUIRE_SOURCEBOT=0 for local-only experiments)."
            )

        pipe = await run_decompose_pipeline(
            ticket=ticket,
            query=enriched,
            query_text=q.question,
            settings=ctx.settings,
            repos=ask_repos,
            max_sourcebot_steps=None,
        )
        decomp = pipe.struct_result.output
        in_tok, out_tok = usage_from_result(pipe.struct_result)
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

        ctx_keys = _repo_keys_for_context(enriched, repos)
        context = _empty_sourcebot_context(ctx_keys)
        decomp = attach_gitlab_links(decomp, context, ctx.settings)
        markdown = render_markdown(decomp=decomp, enriched=enriched, ctx=context, settings=ctx.settings)

        ask = pipe.ask
        meta = ask.metadata
        return EngineResult(
            engine=self.name,
            answer_markdown=markdown,
            payload={"decomposition": decomp.model_dump(mode="json")},
            model=ctx.settings.decompose_model,
            transport="sourcebot+local-llm",
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=cost,
            wall_seconds=ask.wall_seconds,
            extra={
                "ticket_key": ticket.key,
                "ticket_url": ticket.url,
                "affected_repos": list(decomp.affected_repos),
                "subtask_count": len(decomp.subtasks),
                "grounding": "sourcebot_chat",
                "sourcebot": _sourcebot_extra(ask),
                "retrieval": {
                    "total_snippets": 0,
                    "total_chars": 0,
                    "via": "sourcebot",
                },
            },
        )


def _sourcebot_extra(ask: AskResult) -> dict:
    m = ask.metadata
    if not m:
        return {"wall_seconds": ask.wall_seconds}
    return {
        "wall_seconds": ask.wall_seconds,
        "transport": m.transport,
        "model_name": m.model_name,
        "chat_id": m.chat_id,
        "chat_url": m.chat_url,
        "total_input_tokens": m.total_input_tokens,
        "total_output_tokens": m.total_output_tokens,
    }
