"""tech-decomposition CLI — subcommand-based.

Subcommands:
    ask        - ask a code question via Sourcebot
    decompose  - decompose a query into subtasks
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

from .config import get_settings
from .core.context import RunContext
from .core.factory import (
    build_ask_pipeline,
    build_decompose_pipeline,
    build_metrics_observer,
)
from .core.runstore import open_store

console = Console()


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------


def _cmd_ask(args, settings) -> int:
    pipeline = build_ask_pipeline(
        settings,
        max_steps=args.max_steps,
        output_format=args.format,
        include_cli_sink=False,  # we print our own status line; the rendered body is in the file
        include_file_sink=not args.no_file,
    )
    obs = build_metrics_observer(settings, mode="ask")
    ctx = RunContext(settings=settings, mode="ask", metrics=obs)
    try:
        run = asyncio.run(pipeline.run(args.question, ctx))
    except Exception as e:
        with open_store(settings) as store:
            store.record(run_id=ctx.run_id, mode="ask", status="failed",
                         input_ref=args.question, output_format=args.format,
                         error=f"{type(e).__name__}: {e}")
        console.print(f"[red]ask failed:[/] {type(e).__name__}: {e}")
        return 1
    with open_store(settings) as store:
        store.record(run_id=ctx.run_id, mode="ask", status="completed",
                     input_ref=args.question, output_format=args.format,
                     pipeline_run=run)
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
    file_locs = [sr.location for sr in run.sink_results if sr.sink == "file" and sr.location]
    if file_locs:
        console.print(f"  written: [bold]{file_locs[0]}[/]")
    return 0


def _cmd_decompose(args, settings) -> int:
    if args.query:
        ref = {"query": args.query}
    elif args.query_file:
        ref = {"path": args.query_file}
    else:
        console.print("[red]decompose: --query or --query-file required[/]")
        return 2

    repos = [s.strip() for s in args.repos.split(",")] if args.repos else None
    pipeline = build_decompose_pipeline(
        settings,
        source="text_file" if args.query_file else "raw",
        mode=args.mode,
        repos=repos,
        output_format=args.format,
        include_cli_sink=False,  # we print our own summary
        include_file_sink=True,
    )
    obs = build_metrics_observer(settings, mode="decompose")
    ctx = RunContext(settings=settings, mode="decompose", metrics=obs)
    try:
        run = asyncio.run(pipeline.run(ref, ctx))
    except Exception as e:
        with open_store(settings) as store:
            store.record(run_id=ctx.run_id, mode="decompose", status="failed",
                         input_ref=ref, output_format=args.format,
                         error=f"{type(e).__name__}: {e}")
        console.print(f"[red]decompose failed:[/] {type(e).__name__}: {e}")
        return 1
    with open_store(settings) as store:
        store.record(run_id=ctx.run_id, mode="decompose", status="completed",
                     input_ref=ref, output_format=args.format, pipeline_run=run)
    er = run.engine_result

    md_path = next(
        (sr.location for sr in run.sink_results if sr.sink == "file" and sr.location),
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
    if args.print_json:
        decomp = er.payload.get("decomposition") or {}
        print(json.dumps(decomp, indent=2, default=str))
    return 0


def _cmd_serve(args, settings) -> int:
    import uvicorn  # local import — only loaded if serve is invoked

    from .api import app  # FastAPI app

    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


def _cmd_history(args, settings) -> int:
    if args.history_cmd == "show":
        return _history_show(args, settings)
    if args.history_cmd == "replay":
        return _history_replay(args, settings)
    return _history_list(args, settings)


def _history_list(args, settings) -> int:
    since_iso = None
    if args.since:
        from datetime import datetime, timedelta, timezone
        unit_map = {"s": "seconds", "m": "minutes", "h": "hours", "d": "days", "w": "weeks"}
        import re
        m = re.fullmatch(r"(\d+)([smhdw])", args.since)
        if not m:
            console.print(f"[red]invalid --since:[/] {args.since}")
            return 2
        delta = timedelta(**{unit_map[m.group(2)]: int(m.group(1))})
        since_iso = (datetime.now(timezone.utc) - delta).isoformat()

    with open_store(settings) as store:
        rows = store.list(
            mode=args.history_mode,
            status=args.status,
            engine_contains=args.engine,
            since_iso=since_iso,
            limit=args.limit,
        )
    if not rows:
        console.print("[dim]no matching runs[/]")
        return 0
    console.print(
        f"{'id':<14} {'mode':<10} {'status':<10} {'engine':<22} "
        f"{'wall_s':>7} {'cost':>9}  preview"
    )
    console.print("-" * 120)
    for r in rows:
        rid = (r.get("id") or "")[:14]
        mode = (r.get("mode") or "?")[:10]
        st = (r.get("status") or "?")[:10]
        eng = (r.get("engine") or "?")[:22]
        wall = r.get("total_seconds") or 0.0
        cost = r.get("total_cost_usd")
        cost_str = f"${cost:.4f}" if cost is not None else "—"
        preview = (r.get("input_preview") or "").strip()[:60]
        console.print(f"{rid:<14} {mode:<10} {st:<10} {eng:<22} {wall:>7.2f} {cost_str:>9}  {preview}")
    return 0


def _history_show(args, settings) -> int:
    with open_store(settings) as store:
        row = store.get(args.run_id)
    if not row:
        console.print(f"[red]no run with id:[/] {args.run_id}")
        return 1
    if args.json:
        print(json.dumps(row, indent=2, default=str))
        return 0
    console.print(f"[bold]{row['id']}[/]  mode=[bold]{row['mode']}[/]  status=[bold]{row['status']}[/]")
    if row.get("engine"):
        console.print(f"  engine: {row['engine']} · model: {row.get('model') or '—'}")
    if row.get("total_seconds") is not None:
        bits = [f"wall: {row['total_seconds']}s"]
        if row.get("total_cost_usd") is not None:
            bits.append(f"cost: ${row['total_cost_usd']:.4f}")
        if row.get("input_tokens") or row.get("output_tokens"):
            bits.append(f"tokens: {row.get('input_tokens') or 0}/{row.get('output_tokens') or 0}")
        console.print("  " + " · ".join(bits))
    if row.get("input_preview"):
        console.print(f"\n[bold]Input:[/]\n  {row['input_preview']}")
    if row.get("error"):
        console.print(f"\n[red]Error:[/]\n{row['error']}")
        return 0
    if row.get("answer"):
        console.print("\n[bold]Answer:[/]\n")
        console.print(row["answer"])
    return 0


def _history_replay(args, settings) -> int:
    with open_store(settings) as store:
        row = store.get(args.run_id)
    if not row:
        console.print(f"[red]no run with id:[/] {args.run_id}")
        return 1
    mode = row["mode"]
    ref = row.get("input_ref") or json.loads(row.get("input_ref_json") or "null")
    fmt = row.get("output_format") or "markdown"
    if mode == "ask":
        # ref is the question string
        question = ref if isinstance(ref, str) else (ref.get("question") if isinstance(ref, dict) else "")
        # Build args-like namespace and reuse _cmd_ask
        ns = argparse.Namespace(
            question=question, engine="sourcebot", max_steps=None,
            repos=None, format=fmt, no_file=False,
        )
        return _cmd_ask(ns, settings)
    if mode == "decompose":
        ns = argparse.Namespace(
            query=(ref or {}).get("query") if isinstance(ref, dict) else None,
            query_file=(ref or {}).get("path") if isinstance(ref, dict) else None,
            mode="auto", repos=None,
            print_json=False, format=fmt,
        )
        return _cmd_decompose(ns, settings)
    console.print(f"[red]replay not supported for mode:[/] {mode}")
    return 2


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
        prog="tech-decomposition",
        description="tech-decomposition: multi-repo code Q&A and query decomposition",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # ask --------------------------------------------------------------------
    p_ask = sub.add_parser(
        "ask", help="Ask a code question via Sourcebot"
    )
    p_ask.add_argument("question", help="The question to ask")
    p_ask.add_argument(
        "--max-steps", type=int, default=None,
        help="Sourcebot maxSteps (1-50).",
    )
    p_ask.add_argument("--repos", default=None, help="Comma-separated repo dir names")
    p_ask.add_argument(
        "--format",
        choices=["markdown", "html", "text"],
        default="markdown",
        help="Output format. Affects both the printed answer and the saved file extension.",
    )
    p_ask.add_argument("--no-file", action="store_true", help="Skip writing the answer file")
    p_ask.set_defaults(func=_cmd_ask)

    # decompose --------------------------------------------------------------
    p_dec = sub.add_parser("decompose", help="Decompose a query into subtasks")
    src = p_dec.add_mutually_exclusive_group(required=True)
    src.add_argument("--query", help="Raw question/query string")
    src.add_argument("--query-file", help="Path to a .txt/.md file with query")
    p_dec.add_argument(
        "--mode",
        choices=["cheap", "deep", "auto"],
        default="auto",
        help="cheap = single-pass retrieval + LLM. "
             "deep (EXPERIMENTAL) = agentic Pro with tools. "
             "auto = cheap, escalate to deep on low-confidence / migration cues.",
    )
    p_dec.add_argument("--repos", default=None, help="Comma-separated repo dir names")
    p_dec.add_argument("--print-json", action="store_true",
                       help="Print the structured Decomposition JSON to stdout too")
    p_dec.add_argument(
        "--format",
        choices=["markdown", "html", "text"],
        default="markdown",
        help="Output format. Affects the saved file extension.",
    )
    p_dec.set_defaults(func=_cmd_decompose)

    # serve ------------------------------------------------------------------
    p_srv = sub.add_parser("serve", help="Run the FastAPI server")
    p_srv.add_argument("--host", default="127.0.0.1")
    p_srv.add_argument("--port", type=int, default=8000)
    p_srv.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"])
    p_srv.set_defaults(func=_cmd_serve)

    # history ----------------------------------------------------------------
    p_hist = sub.add_parser("history", help="List, show, or replay prior runs (from outputs/runstore.db)")
    hsub = p_hist.add_subparsers(dest="history_cmd")
    # tech-decomposition history (list, default)
    p_hist.add_argument("--limit", type=int, default=20)
    p_hist.add_argument("--since", default=None, help="Window: '24h', '7d', '15m'.")
    p_hist.add_argument("--engine", default=None, help="Filter by engine substring.")
    p_hist.add_argument("--status", default=None, choices=[None, "completed", "failed"],
                        help="Filter by status.")
    p_hist.add_argument("--history-mode", default=None, choices=[None, "ask", "decompose"],
                        help="Filter by run mode.")
    # tech-decomposition history show <id>
    p_show = hsub.add_parser("show", help="Print a full prior run by id")
    p_show.add_argument("run_id")
    p_show.add_argument("--json", action="store_true", help="Print as JSON.")
    # tech-decomposition history replay <id>
    p_rep = hsub.add_parser("replay", help="Re-run a prior request by id")
    p_rep.add_argument("run_id")
    p_hist.set_defaults(func=_cmd_history)

    # analyze ----------------------------------------------------------------
    p_ana = sub.add_parser("analyze", help="Summarize outputs/metrics/runs.jsonl")
    p_ana.add_argument("--since", default=None,
                       help="Time window: '24h', '7d', '15m'. Default: all runs.")
    p_ana.add_argument("--by-engine", default=None,
                       help="Substring match on engine name (e.g. 'sourcebot').")
    p_ana.add_argument("--mode-filter", default=None,
                       help="Filter by mode: ask | decompose.")
    p_ana.set_defaults(func=_cmd_analyze)

    return p


def main() -> int:
    settings = get_settings()
    parser = _build_parser(settings)
    args = parser.parse_args()
    return args.func(args, settings)


if __name__ == "__main__":
    sys.exit(main())
