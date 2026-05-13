from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from rich.console import Console

from .ask import AskMetadata, AskResult, Citation as SourcebotCitation, SourcebotAskError, ask_sourcebot, render_ask_markdown, write_ask_markdown
from .config import get_settings
from .local_ask import Answer, local_ask
from .metrics import estimate_cost_usd, usage_from_result
from .models import DecomposeRequest
from .pipeline import run_pipeline

console = Console()


def _run_ask(args, settings) -> int:
    """Handle --ask. Tries Sourcebot ``POST /api/ask`` (SSE, Sentinel-style),
    then MCP ``ask_codebase`` on 404 (unless disabled), else local agent only if ask_via=auto."""
    import time
    question = args.ask
    repos = [s.strip() for s in args.repos.split(",")] if args.repos else None

    sourcebot_result: AskResult | None = None

    if args.ask_via in ("auto", "sourcebot"):
        try:
            sourcebot_result = asyncio.run(ask_sourcebot(
                question,
                settings=settings,
                repos=repos,
                max_steps=args.ask_max_steps,
                disable_mcp_fallback=True if args.ask_sse_only else None,
            ))
        except SourcebotAskError as e:
            if args.ask_via == "sourcebot":
                console.print(f"[red]Sourcebot error:[/] {e}")
                return 1
            console.print(f"[yellow]Sourcebot ask unavailable ({e}); falling back to local agent.[/]")
            sourcebot_result = None

    if sourcebot_result is not None and sourcebot_result.answer.strip():
        path = write_ask_markdown(question, sourcebot_result, settings)
        console.print(f"\n[green]✓[/] answer written to: [bold]{path}[/]")
        meta = sourcebot_result.metadata
        tr = meta.transport if meta else "sse"
        via_label = {
            "sse": "Sourcebot SSE /api/ask",
            "mcp": "Sourcebot MCP ask_codebase",
            "local": "local agent",
        }[tr]
        bits = [f"via: [bold]{via_label}[/]"]
        if meta:
            if meta.model_name: bits.append(f"model: {meta.model_name}")
            if meta.total_tokens: bits.append(f"tokens: {meta.total_tokens}")
        bits.append(f"wall: {sourcebot_result.wall_seconds}s")
        bits.append(f"citations: {len(sourcebot_result.citations)}")
        console.print("  " + " · ".join(bits))
        return 0

    if args.ask_via == "sourcebot":
        msg = (
            "Sourcebot returned no usable answer."
            if sourcebot_result is not None
            else "Sourcebot did not return a result."
        )
        console.print(f"[red]{msg}[/]")
        return 1

    # Local fallback (ask_via=auto only)
    t0 = time.monotonic()
    result, deps = asyncio.run(local_ask(question, settings))
    wall = round(time.monotonic() - t0, 2)
    answer: Answer = result.output
    in_tok, out_tok = usage_from_result(result)
    cost = estimate_cost_usd(settings.decompose_model, in_tok or 0, out_tok or 0)

    # Render to a markdown file using the same renderer as Sourcebot path.
    converted = AskResult(
        answer=answer.answer,
        citations=[
            SourcebotCitation(
                repo=c.repo, path=c.path,
                start_line=c.line_start, end_line=c.line_end,
            ) for c in answer.citations
        ],
        metadata=AskMetadata(
            total_input_tokens=in_tok, total_output_tokens=out_tok,
            total_tokens=(in_tok or 0) + (out_tok or 0) if (in_tok or out_tok) else None,
            model_name=settings.decompose_model,
            sources_seen=[],
            transport="local",
        ),
        wall_seconds=wall,
    )
    path = write_ask_markdown(question, converted, settings)
    console.print(f"\n[green]✓[/] answer written to: [bold]{path}[/]")
    bits = [f"via: [bold]local agent[/]"]
    bits.append(f"model: {settings.decompose_model}")
    if in_tok and out_tok:
        bits.append(f"tokens: {in_tok}/{out_tok}")
    if cost is not None:
        bits.append(f"cost: [yellow]${cost:.4f}[/]")
    bits.append(f"wall: {wall}s")
    bits.append(f"citations: {len(answer.citations)}")
    bits.append(f"confidence: {answer.confidence}")
    console.print("  " + " · ".join(bits))
    tc = dict(deps.tool_calls)
    if tc:
        console.print(f"  tools: {sum(tc.values())} calls "
                      f"({', '.join(f'{k}={v}' for k,v in sorted(tc.items()))})")
    return 0


def main() -> int:
    settings = get_settings()
    p = argparse.ArgumentParser(description="ticket-recon CLI")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--ticket-url", help="Jira ticket URL")
    src.add_argument("--ticket-key", help="Jira ticket key (e.g. DEV-7543)")
    src.add_argument("--ticket-text-file", help="Path to a .txt/.md file with ticket title+body")
    src.add_argument("--ask", metavar="QUESTION",
                     help="Ask a code question via Sourcebot (SSE POST /api/ask, MCP on 404) "
                          "or local agent when --ask-via allows it (Q&A mode, skips ticket decomposition).")
    p.add_argument(
        "--repos",
        help="Comma-separated repo dir names to override the configured set",
        default=None,
    )
    p.add_argument("--print-json", action="store_true",
                   help="Print the structured Decomposition JSON to stdout in addition to writing markdown")
    p.add_argument("--post-to-jira", action="store_true",
                   help="Post the decomposition as a Jira comment on the ticket (requires a ticket key)")
    p.add_argument("--mode", choices=["cheap", "deep", "auto"], default="auto",
                   help="cheap = single-pass static retrieval (default ~$0.06, ~50s). "
                        "deep = agentic Pro with tools (~$0.30+, ~3-5min). "
                        "auto = cheap, escalate to deep on low-confidence or migration cues.")
    p.add_argument("--ask-max-steps", type=int, default=None,
                   help="When using --ask, max reasoning steps Sourcebot takes (1-50, default 20). "
                        "Ignored for local fallback.")
    p.add_argument(
        "--ask-via",
        choices=["auto", "sourcebot", "local"],
        default=settings.ask_via,
        help="auto = try Sourcebot SSE /api/ask, MCP if 404, else local agent. "
             "sourcebot = require Sourcebot only (no local fallback). "
             "local = local Pydantic AI agent. "
             "Default: ASK_VIA env or 'auto'.",
    )
    p.add_argument(
        "--ask-sse-only",
        action="store_true",
        help="With --ask, use only Sourcebot chat SSE POST /api/ask (no MCP fallback). "
             "Same as SOURCEBOT_DISABLE_MCP_FALLBACK=1 for this run.",
    )
    args = p.parse_args()

    if args.ask:
        return _run_ask(args, settings)

    text = None
    if args.ticket_text_file:
        text = Path(args.ticket_text_file).read_text()

    req = DecomposeRequest(
        ticket_url=args.ticket_url,
        ticket_key=args.ticket_key,
        ticket_text=text,
        repos=[s.strip() for s in args.repos.split(",")] if args.repos else None,
        post_to_jira=args.post_to_jira,
        mode=args.mode,
    )

    resp = asyncio.run(run_pipeline(req, settings))

    m = resp.metrics or {}
    cost = m.get("total_cost_usd")
    mode_label = m.get("mode", "?")
    if m.get("escalated_to_deep"):
        mode_label = f"{mode_label} (auto-escalated)"
    console.print(f"\n[green]✓[/] decomposition written to: [bold]{resp.markdown_path}[/]")
    console.print(f"  mode: [bold]{mode_label}[/]")
    console.print(f"  enriched intent: [cyan]{resp.enriched_query.intent}[/] "
                  f"(confidence: {resp.enriched_query.confidence})")
    console.print(f"  affected repos: {', '.join(resp.decomposition.affected_repos) or '(none)'}")
    console.print(f"  subtasks: {len(resp.decomposition.subtasks)}")
    if m:
        console.print(f"  timing: total {m.get('total_seconds')}s "
                      f"(enrich {m['enrich']['seconds']}s · retrieve {m['retrieval']['seconds']}s · "
                      f"decompose {m['decompose']['seconds']}s)")
        console.print(f"  retrieval: {m['retrieval']['total_snippets']} snippets, "
                      f"{m['retrieval']['total_chars']:,} chars "
                      f"({', '.join(f'{k}={v}' for k,v in m['retrieval']['by_source'].items()) or '—'})")
        tc = m.get("deep_tool_calls") or {}
        if tc:
            console.print(f"  deep tools: {sum(tc.values())} calls "
                          f"({', '.join(f'{k}={v}' for k,v in sorted(tc.items()))})")
        if cost is not None:
            console.print(f"  cost: [yellow]${cost:.4f}[/]")
    if resp.jira_comment_id:
        console.print(f"  posted Jira comment id: [magenta]{resp.jira_comment_id}[/]")

    if args.print_json:
        print(json.dumps(resp.decomposition.model_dump(mode="json"), indent=2, default=str))

    return 0


if __name__ == "__main__":
    sys.exit(main())
