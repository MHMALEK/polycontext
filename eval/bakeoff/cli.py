"""Bake-off CLI.

Run via:
    uv run python -m eval.bakeoff.cli list-cases
    uv run python -m eval.bakeoff.cli list-adapters
    uv run python -m eval.bakeoff.cli run --job ask --adapters cline_sdk,opencode
    uv run python -m eval.bakeoff.cli report eval/outputs/eval-20260514T...

The runner uses an in-process FastAPI ``TestClient`` by default (no need to
start the server). Pass ``--base-url http://localhost:8000`` to hit a real
running stack instead — useful when an adapter needs Docker sidecars.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path


def _load_dotenv_into_environ() -> None:
    """Make ``.env`` available to subprocess-level libraries.

    ``pydantic-settings`` reads ``.env`` into the ``Settings`` model fields,
    but the adapter CLIs (Cline, OpenCode, cursor-agent) read ``os.environ``
    directly for provider keys. Without this shim, you get the surprising
    ``API_KEY_INVALID`` errors when an adapter inside the FastAPI process
    can't see keys that the parent app loaded.

    Done before any other imports so the env is set when ``Settings()`` and
    adapter modules first import.
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    env_file = project_root / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        # Don't overwrite anything already set in the shell — that lets
        # users override .env from the command line.
        os.environ.setdefault(key, value)


_load_dotenv_into_environ()

from .cases import discover_cases, filter_cases  # noqa: E402 — load env first
from .client import build_client  # noqa: E402
from .report import render_report  # noqa: E402
from .runner import new_output_dir, run_bakeoff  # noqa: E402

log = logging.getLogger(__name__)


def _project_root() -> Path:
    """The repo root is two ``parents`` up from this file (eval/bakeoff/cli.py)."""
    return Path(__file__).resolve().parent.parent.parent


def _eval_dir() -> Path:
    return _project_root() / "eval"


def _outputs_dir() -> Path:
    return _eval_dir() / "outputs"


# ---------------------------------------------------------------------------
# subcommands
# ---------------------------------------------------------------------------


async def _cmd_list_adapters(args) -> int:
    client = build_client(args.base_url)
    try:
        adapters = await client.list_adapters()
    finally:
        await client.aclose()
    for a in adapters:
        h = "ok" if a["health"]["ok"] else f"NO  ({a['health'].get('reason', '?')[:80]})"
        caps = ",".join(a["capabilities"])
        print(f"  {a['name']:14}  cap=[{caps}]  health={h}")
    return 0


def _cmd_list_cases(args) -> int:
    cases = discover_cases(_eval_dir())
    cases = filter_cases(cases, job=args.job, ids=args.ids, tags=args.tags)
    for c in cases:
        tag_str = ",".join(c.tags) if c.tags else "-"
        print(f"  [{c.job:9}] {c.id:38} tags={tag_str}")
    return 0


async def _cmd_run(args) -> int:
    cases = filter_cases(
        discover_cases(_eval_dir()),
        job=args.job,
        ids=args.ids,
        tags=args.tags,
    )
    if not cases:
        print("no cases match filter", file=sys.stderr)
        return 1
    if not args.adapters:
        print("--adapters required", file=sys.stderr)
        return 1

    client = build_client(args.base_url)
    out_dir = new_output_dir(_outputs_dir())
    print(f"output: {out_dir}", file=sys.stderr)

    def _progress(done, total, case_id, adapter):
        bar = f"[{done:>3}/{total}]"
        print(f"{bar} {adapter:14} {case_id}", file=sys.stderr)

    try:
        await run_bakeoff(
            client=client,
            cases=cases,
            adapters=args.adapters,
            output_dir=out_dir,
            timeout_seconds=args.timeout,
            on_progress=_progress,
        )
    finally:
        await client.aclose()

    report_path, summary_path = render_report(out_dir)
    print(f"report:  {report_path}", file=sys.stderr)
    print(f"summary: {summary_path}", file=sys.stderr)

    # Final leaderboard to stdout so it's pipeable.
    summary = json.loads(summary_path.read_text())
    print(json.dumps(summary["leaderboard"], indent=2))
    return 0


def _cmd_report(args) -> int:
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = _outputs_dir() / run_dir
    report_path, summary_path = render_report(run_dir)
    print(f"report:  {report_path}")
    print(f"summary: {summary_path}")
    return 0


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="eval.bakeoff", description="adapter bake-off harness")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    def _common_run_filters(sp):
        sp.add_argument("--job", choices=["ask", "decompose", "implement"], default=None)
        sp.add_argument("--ids", nargs="+", default=None, help="case id allowlist")
        sp.add_argument("--tags", nargs="+", default=None, help="tag intersection filter")

    la = sub.add_parser("list-adapters", help="list registered adapters and health")
    la.add_argument("--base-url", default=None)

    lc = sub.add_parser("list-cases", help="list discovered cases (filterable)")
    _common_run_filters(lc)

    r = sub.add_parser("run", help="fan cases × adapters and write a run dir")
    r.add_argument("--adapters", type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
                   required=True, help="comma-separated adapter names")
    r.add_argument("--base-url", default=None, help="omit to use in-process TestClient")
    r.add_argument("--timeout", type=float, default=900.0, help="per-case timeout seconds")
    _common_run_filters(r)

    rp = sub.add_parser("report", help="regenerate report.md/summary.json for a run dir")
    rp.add_argument("run_dir", help="path under eval/outputs/ or absolute")

    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.cmd == "list-adapters":
        return asyncio.run(_cmd_list_adapters(args))
    if args.cmd == "list-cases":
        return _cmd_list_cases(args)
    if args.cmd == "run":
        return asyncio.run(_cmd_run(args))
    if args.cmd == "report":
        return _cmd_report(args)
    parser.error(f"unknown command: {args.cmd}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
