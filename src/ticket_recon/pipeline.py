from __future__ import annotations

import time
import uuid
from pathlib import Path

from .config import Settings
from .contradiction import check_contradictions
from .decompose import decompose
from .deep_decompose import deep_decompose
from .enrich import enrich_ticket
from .jira import fetch_ticket, post_comment, ticket_from_text
from .markdown_to_adf import markdown_to_adf
from .metrics import (
    RetrievalMetrics,
    RunMetrics,
    StageMetrics,
    estimate_cost_usd,
    now_utc,
    retrieval_metrics,
    usage_from_result,
    write_metrics,
)
from .models import (
    Decomposition,
    DecomposeRequest,
    DecomposeResponse,
    EnrichedQuery,
    RepoContext,
    RetrievedContext,
    Ticket,
)
from .output import attach_gitlab_links, render_markdown, write_markdown
from .retrievers import gather_anchor_snippets, gather_context


def _decide_mode(req: DecomposeRequest, enriched: EnrichedQuery, ticket: Ticket) -> str:
    """Resolve `auto` to either `cheap` or `deep` based on signal."""
    if req.mode in ("cheap", "deep"):
        return req.mode
    # Auto-escalate heuristics:
    if enriched.confidence == "low":
        return "deep"
    body = (ticket.body or "").lower()
    title = ticket.title.lower()
    text = f"{title} {body}"
    if any(kw in text for kw in ("migrate", "migration", "port to", "move to ", "refactor across")):
        return "deep"
    if len(enriched.suspected_repos) >= 3:
        return "deep"
    return "cheap"


async def resolve_ticket(req: DecomposeRequest, settings: Settings) -> Ticket:
    if req.ticket_text:
        return ticket_from_text(req.ticket_text, key=req.ticket_key, url=req.ticket_url)
    return await fetch_ticket(settings=settings, key=req.ticket_key, url=req.ticket_url)


async def run_pipeline(req: DecomposeRequest, settings: Settings) -> DecomposeResponse:
    repos = req.repos or settings.repos
    run_id = f"{now_utc().strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:6]}"
    started = now_utc()
    t0 = time.monotonic()

    ticket = await resolve_ticket(req, settings)

    t_enrich = time.monotonic()
    enrich_result = await enrich_ticket(ticket, repos, settings)
    enrich_secs = time.monotonic() - t_enrich
    enriched: EnrichedQuery = enrich_result.output
    enrich_in, enrich_out = usage_from_result(enrich_result)
    enrich_cost = estimate_cost_usd(settings.enrich_model, enrich_in or 0, enrich_out or 0)
    enrich_stage = StageMetrics(
        seconds=round(enrich_secs, 3),
        model=settings.enrich_model,
        input_tokens=enrich_in,
        output_tokens=enrich_out,
        cost_usd=enrich_cost,
    )

    mode = _decide_mode(req, enriched, ticket)
    escalated = (req.mode == "auto" and mode == "deep")

    deep_iterations: int | None = None
    deep_tool_calls: dict[str, int] = {}

    if mode == "deep":
        # Build anchors-only context for permalinks + metrics. Pro retrieves
        # the rest itself via tools, so we skip the keyword retriever fan-out.
        t_ret = time.monotonic()
        anchor_ctx_map = gather_anchor_snippets(ticket, settings)
        repo_ctxs = [
            anchor_ctx_map.get(r) or RepoContext(repo=r, head_sha="HEAD", snippets=[])
            for r in repos
        ]
        context = RetrievedContext(
            repos=repo_ctxs,
            total_snippets=sum(len(rc.snippets) for rc in repo_ctxs),
            total_chars=sum(len(s.content) for rc in repo_ctxs for s in rc.snippets),
        )
        ret_secs = time.monotonic() - t_ret
        ret_stage = retrieval_metrics(context, ret_secs)

        t_dec = time.monotonic()
        decompose_result, deep_deps = await deep_decompose(
            ticket=ticket, enriched=enriched, settings=settings,
        )
        dec_secs = time.monotonic() - t_dec
        decomp = decompose_result.output
        dec_in, dec_out = usage_from_result(decompose_result)
        dec_cost = estimate_cost_usd(settings.decompose_model, dec_in or 0, dec_out or 0)
        decompose_stage = StageMetrics(
            seconds=round(dec_secs, 3),
            model=settings.decompose_model,
            input_tokens=dec_in,
            output_tokens=dec_out,
            cost_usd=dec_cost,
        )
        deep_tool_calls = dict(deep_deps.tool_calls)
        deep_iterations = sum(deep_tool_calls.values())
    else:
        t_ret = time.monotonic()
        context = await gather_context(repos, enriched, settings, ticket=ticket)
        ret_secs = time.monotonic() - t_ret
        ret_stage = retrieval_metrics(context, ret_secs)

        t_dec = time.monotonic()
        decompose_result = await decompose(
            ticket=ticket, query=enriched, context=context, settings=settings,
        )
        dec_secs = time.monotonic() - t_dec
        decomp = decompose_result.output
        dec_in, dec_out = usage_from_result(decompose_result)
        dec_cost = estimate_cost_usd(settings.decompose_model, dec_in or 0, dec_out or 0)
        decompose_stage = StageMetrics(
            seconds=round(dec_secs, 3),
            model=settings.decompose_model,
            input_tokens=dec_in,
            output_tokens=dec_out,
            cost_usd=dec_cost,
        )

    # Contradiction check (cheap Flash call). Folds any findings into the
    # decomposition's risks section before rendering, so the markdown surfaces
    # them and downstream agents see them.
    try:
        contradiction_result = await check_contradictions(
            ticket=ticket, decomp=decomp, settings=settings,
        )
        report = contradiction_result.output
        if report.contradictions:
            for c in report.contradictions:
                where = f"subtask #{c.subtask_index}" if c.subtask_index else "overall"
                decomp.risks.append(
                    f"[contradiction-check / {c.severity}] {where}: {c.issue} "
                    f"(ticket says: \"{c.ticket_excerpt}\")"
                )
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("contradiction check failed: %s", e)

    decomp = attach_gitlab_links(decomp, context, settings)
    markdown = render_markdown(decomp=decomp, enriched=enriched, ctx=context, settings=settings)
    md_path = write_markdown(markdown, decomp, settings)

    comment_id: str | None = None
    if req.post_to_jira and (ticket.key or req.ticket_key):
        try:
            adf = markdown_to_adf(markdown)
            comment_id = await post_comment(
                settings=settings, key=(ticket.key or req.ticket_key or ""), adf_body=adf,
            )
        except Exception as e:
            # Don't fail the whole run if Jira posting fails.
            import logging
            logging.getLogger(__name__).warning("post_to_jira failed: %s", e)

    total_seconds = time.monotonic() - t0
    total_cost = sum(
        c for c in (enrich_cost, dec_cost) if c is not None
    ) or None

    metrics = RunMetrics(
        run_id=run_id,
        started_at=started,
        ticket_key=ticket.key,
        ticket_title=ticket.title,
        mode=mode,
        escalated_to_deep=escalated,
        total_seconds=round(total_seconds, 3),
        enrich=enrich_stage,
        retrieval=ret_stage,
        decompose=decompose_stage,
        deep_iterations=deep_iterations,
        deep_tool_calls=deep_tool_calls,
        total_cost_usd=round(total_cost, 6) if total_cost is not None else None,
        posted_to_jira=comment_id is not None,
        markdown_path=str(md_path),
        subtask_count=len(decomp.subtasks),
        affected_repos=decomp.affected_repos,
    )
    write_metrics(metrics, settings)

    return DecomposeResponse(
        decomposition=decomp,
        enriched_query=enriched,
        markdown_path=str(md_path),
        markdown=markdown,
        jira_comment_id=comment_id,
        metrics=metrics.model_dump(mode="json"),
    )
