"""ticket-recon CLI — subcommand-based.

Subcommands:
    ask        - ask a code question via Sourcebot (or local agent, experimental)
    decompose  - decompose a Jira ticket / text file into subtasks
    compare    - run a TOML batch of questions through engines side-by-side
    serve      - run the FastAPI server
    analyze    - print a summary of outputs/metrics/runs.jsonl

The body of each subcommand composes a Pipeline via ``core.factory`` and runs
it; the CLI is glue, not logic.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from rich.console import Console

from .compare import run_compare
from .config import get_settings
from .core.context import RunContext
from .core.factory import (
    build_ask_pipeline,
    build_decompose_pipeline,
    build_metrics_observer,
)

console = Console()


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------


def _cmd_ask(args, settings) -> int:
    pipeline = build_ask_pipeline(
        settings,
        engine=args.engine,
        max_steps=args.max_steps,
        include_cli_sink=False,  # we print our own status line; the markdown body is in the file
        include_file_sink=not args.no_file,
    )
    obs = build_metrics_observer(settings, mode="ask")
    ctx = RunContext(settings=settings, mode="ask", metrics=obs)
    run = asyncio.run(pipeline.run(args.question, ctx))
    er = run.engine_result

    console.print(er.answer_markdown.strip())
    bits = [f"via: [bold]{er.transport or er.engine}[/]"]
    if er.model:
        bits.append(f"model: {er.model}")
    if er.wall_seconds is not None:
        bits.append(f"wall: {er.wall_seconds}s")
    if er.input_tokens or er.output_tokens:
        bits.append(f"tokens: {er.input_tokens or 0}/{er.output_tokens or 0}")
    if er.cost_usd is not None:
        bits.append(f"cost: [yellow]${er.cost_usd:.4f}[/]")
    if er.citations:
        bits.append(f"citations: {len(er.citations)}")
    console.print("\n  " + " · ".join(bits))
    file_locs = [sr.location for sr in run.sink_results if sr.sink == "markdown_file" and sr.location]
    if file_locs:
        console.print(f"  written: [bold]{file_locs[0]}[/]")
    return 0


def _cmd_decompose(args, settings) -> int:
    if args.ticket_text_file:
        source = "text_file"
        ref = {"path": args.ticket_text_file, "key": args.ticket_key, "url": args.ticket_url}
    elif args.ticket_key:
        source = "jira"
        ref = {"key": args.ticket_key}
    elif args.ticket_url:
        source = "jira"
        ref = {"url": args.ticket_url}
    else:
        console.print("[red]decompose: one of --ticket-key / --ticket-url / --ticket-text-file required[/]")
        return 2

    repos = [s.strip() for s in args.repos.split(",")] if args.repos else None
    pipeline = build_decompose_pipeline(
        settings,
        source=source,
        mode=args.mode,
        repos=repos,
        post_to_jira=args.post_to_jira,
        include_cli_sink=False,  # we print our own summary
        include_file_sink=True,
    )
    obs = build_metrics_observer(settings, mode="decompose")
    ctx = RunContext(settings=settings, mode="decompose", metrics=obs)
    run = asyncio.run(pipeline.run(ref, ctx))
    er = run.engine_result

    md_path = next(
        (sr.location for sr in run.sink_results if sr.sink == "markdown_file" and sr.location),
        None,
    )
    if md_path:
        console.print(f"\n[green]✓[/] decomposition written to: [bold]{md_path}[/]")
    console.print(f"  engine: [bold]{er.engine}[/]")
    affected = er.extra.get("affected_repos") or []
    console.print(f"  affected repos: {', '.join(affected) or '(none)'}")
    console.print(f"  subtasks: {er.extra.get('subtask_count', 0)}")
    bits = [f"wall: {run.total_seconds}s"]
    if er.cost_usd is not None:
        bits.append(f"cost: [yellow]${er.cost_usd:.4f}[/]")
    if er.input_tokens or er.output_tokens:
        bits.append(f"tokens: {er.input_tokens or 0}/{er.output_tokens or 0}")
    console.print("  " + " · ".join(bits))
    jira_sr = next((sr for sr in run.sink_results if sr.sink == "jira_comment"), None)
    if jira_sr and jira_sr.location:
        console.print(f"  posted jira: [magenta]{jira_sr.location}[/]")
    if args.print_json:
        decomp = er.payload.get("decomposition") or {}
        print(json.dumps(decomp, indent=2, default=str))
    return 0


def _cmd_compare(args, settings) -> int:
    path = Path(args.questions)
    if not path.is_file():
        console.print(f"[red]questions file not found:[/] {path}")
        return 1
    out_dir = asyncio.run(run_compare(
        path, settings,
        only_ids=args.only,
        skip_engine=args.skip,
    ))
    console.print(f"\n[green]✓[/] compare run dir: [bold]{out_dir}[/]")
    return 0


def _cmd_serve(args, settings) -> int:
    import uvicorn  # local import — only loaded if serve is invoked

    from .api import app  # FastAPI app

    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


def _cmd_analyze(args, settings) -> int:
    from .analyze import main as analyze_main

    forward: list[str] = []
    if args.since:
        forward += ["--since", args.since]
    if args.by_engine:
        forward += ["--by-engine", args.by_engine]
    if args.mode_filter:
        forward += ["--mode", args.mode_filter]
    analyze_main(forward)
    return 0


# ---------------------------------------------------------------------------
# Argparse wiring
# ---------------------------------------------------------------------------


def _build_parser(settings) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ticket-recon",
        description="ticket-recon: multi-repo code Q&A and Jira ticket decomposition",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # ask --------------------------------------------------------------------
    p_ask = sub.add_parser(
        "ask", help="Ask a code question via Sourcebot (or experimental local agent)"
    )
    p_ask.add_argument("question", help="The question to ask")
    p_ask.add_argument(
        "--engine",
        choices=["sourcebot", "local"],
        default="sourcebot",
        help="sourcebot (default) = remote /api/chat/blocking + MCP fallback. "
             "local (EXPERIMENTAL) = in-process pydantic-ai agent against local checkouts.",
    )
    p_ask.add_argument(
        "--max-steps", type=int, default=None,
        help="Sourcebot maxSteps (1-50). Ignored for local engine.",
    )
    p_ask.add_argument("--repos", default=None, help="Comma-separated repo dir names")
    p_ask.add_argument("--no-file", action="store_true", help="Skip writing the markdown file")
    p_ask.set_defaults(func=_cmd_ask)

    # decompose --------------------------------------------------------------
    p_dec = sub.add_parser("decompose", help="Decompose a Jira ticket into subtasks")
    src = p_dec.add_mutually_exclusive_group(required=True)
    src.add_argument("--ticket-key", help="Jira ticket key (e.g. DEV-7543)")
    src.add_argument("--ticket-url", help="Jira ticket URL")
    src.add_argument("--ticket-text-file", help="Path to a .txt/.md file with ticket title+body")
    p_dec.add_argument(
        "--mode",
        choices=["cheap", "deep", "auto"],
        default="auto",
        help="cheap = single-pass retrieval + LLM. "
             "deep (EXPERIMENTAL) = agentic Pro with tools. "
             "auto = cheap, escalate to deep on low-confidence / migration cues.",
    )
    p_dec.add_argument("--repos", default=None, help="Comma-separated repo dir names")
    p_dec.add_argument("--post-to-jira", action="store_true",
                       help="Post the decomposition as a Jira comment (requires ticket key)")
    p_dec.add_argument("--print-json", action="store_true",
                       help="Print the structured Decomposition JSON to stdout too")
    p_dec.set_defaults(func=_cmd_decompose)

    # compare ----------------------------------------------------------------
    p_cmp = sub.add_parser("compare", help="Batch-run questions.toml side-by-side across engines")
    p_cmp.add_argument("questions", help="Path to questions.toml")
    p_cmp.add_argument("--only", action="append", default=None,
                       help="Run only the question with this id (repeatable)")
    p_cmp.add_argument("--skip", choices=["sourcebot", "local"], default=None,
                       help="Skip this engine for this run")
    p_cmp.set_defaults(func=_cmd_compare)

    # serve ------------------------------------------------------------------
    p_srv = sub.add_parser("serve", help="Run the FastAPI server")
    p_srv.add_argument("--host", default="127.0.0.1")
    p_srv.add_argument("--port", type=int, default=8000)
    p_srv.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"])
    p_srv.set_defaults(func=_cmd_serve)

    # analyze ----------------------------------------------------------------
    p_ana = sub.add_parser("analyze", help="Summarize outputs/metrics/runs.jsonl")
    p_ana.add_argument("--since", default=None,
                       help="Time window: '24h', '7d', '15m'. Default: all runs.")
    p_ana.add_argument("--by-engine", default=None,
                       help="Substring match on engine name (e.g. 'sourcebot').")
    p_ana.add_argument("--mode-filter", default=None,
                       help="Filter by mode: ask | decompose | compare.")
    p_ana.set_defaults(func=_cmd_analyze)

    return p


def main() -> int:
    settings = get_settings()
    parser = _build_parser(settings)
    args = parser.parse_args()
    return args.func(args, settings)


if __name__ == "__main__":
    sys.exit(main())
