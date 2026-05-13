"""Summarize ``outputs/metrics/runs.jsonl`` — per-run cost / latency / engine mix.

Reads JSONL emitted by ``core.metrics.JsonlMetricsObserver`` and prints a
human-readable table plus aggregates. Supports filtering by time window and
engine name so you can answer "what did the last 24h cost" or "how is the
local agent doing this week" without grepping JSONL by hand.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

from .config import get_settings


_SINCE_RX = re.compile(r"^(\d+)([smhdw])$")
_UNITS = {"s": "seconds", "m": "minutes", "h": "hours", "d": "days", "w": "weeks"}


def _parse_since(spec: str) -> datetime | None:
    """Parse '24h' / '7d' / '15m' into an absolute UTC threshold."""
    m = _SINCE_RX.fullmatch(spec.strip())
    if not m:
        raise SystemExit(f"--since: expected '<n><smhdw>' (e.g. '24h', '7d'), got {spec!r}")
    n, unit = int(m.group(1)), m.group(2)
    delta = timedelta(**{_UNITS[unit]: n})
    return datetime.now(timezone.utc) - delta


def _row_ts(r: dict[str, Any]) -> datetime | None:
    ts = r.get("ts")
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group stage + run rows by run_id; return one summary per run."""
    by_run: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "stages": [],
            "total_seconds": None,
            "total_cost_usd": 0.0,
            "mode": "?",
            "input_preview": None,
            "engine": None,
            "model": None,
            "ts": None,
        }
    )
    for r in rows:
        rt = r.get("type")
        rid = r.get("run_id")
        if not rid or not rt:
            continue
        bucket = by_run[rid]
        bucket["mode"] = r.get("mode", bucket["mode"])
        ts = _row_ts(r)
        if ts and (bucket["ts"] is None or ts < bucket["ts"]):
            bucket["ts"] = ts
        if rt == "stage":
            bucket["stages"].append(r)
            cost = r.get("cost_usd")
            if cost is not None:
                bucket["total_cost_usd"] += cost
        elif rt == "run":
            bucket["total_seconds"] = r.get("total_seconds")
            if r.get("total_cost_usd") is not None:
                bucket["total_cost_usd"] = r["total_cost_usd"]
            extra = r.get("extra") or {}
            if extra.get("input_preview"):
                bucket["input_preview"] = extra["input_preview"]
            if extra.get("engine"):
                bucket["engine"] = extra["engine"]
            if extra.get("model"):
                bucket["model"] = extra["model"]
    summaries: list[dict[str, Any]] = []
    for rid, b in by_run.items():
        stages = b["stages"]
        engine_stage = next((s for s in stages if s.get("stage") == "engine"), None)
        eng_name = b["engine"] or (engine_stage.get("strategy") if engine_stage else "?")
        model = b["model"] or (engine_stage.get("model") if engine_stage else None)
        in_tok = sum((s.get("input_tokens") or 0) for s in stages) or None
        out_tok = sum((s.get("output_tokens") or 0) for s in stages) or None
        total_seconds = b["total_seconds"] or sum((s.get("seconds") or 0) for s in stages)
        summaries.append({
            "run_id": rid,
            "mode": b["mode"],
            "engine": eng_name,
            "model": model,
            "total_seconds": total_seconds,
            "total_cost_usd": b["total_cost_usd"] or None,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "n_stages": len(stages),
            "input_preview": b["input_preview"],
            "ts": b["ts"],
        })
    return summaries


def _format_preview(s: str | None, n: int = 50) -> str:
    if not s:
        return "—"
    s = s.strip()
    return (s[: n - 1] + "…") if len(s) > n else s


def _print_table(summaries: list[dict[str, Any]]) -> None:
    summaries.sort(key=lambda r: r["ts"] or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
    print(
        f"{'run_id':<14} {'mode':<10} {'engine':<22} {'wall_s':>7} "
        f"{'in_tok':>8} {'out_tok':>8} {'cost':>9}  preview"
    )
    print("-" * 130)
    for r in summaries:
        rid = r["run_id"][:14]
        mode = (r["mode"] or "?")[:10]
        eng = (r["engine"] or "?")[:22]
        wall = r["total_seconds"] or 0.0
        in_t = r["input_tokens"] or "—"
        out_t = r["output_tokens"] or "—"
        cost = r["total_cost_usd"]
        cost_str = f"${cost:.4f}" if cost is not None else "—"
        preview = _format_preview(r["input_preview"], 60)
        print(
            f"{rid:<14} {mode:<10} {eng:<22} {wall:>7.2f} "
            f"{str(in_t):>8} {str(out_t):>8} {cost_str:>9}  {preview}"
        )


def _print_aggregates(summaries: list[dict[str, Any]]) -> None:
    if not summaries:
        return
    print("-" * 130)
    total_cost = sum((r["total_cost_usd"] or 0) for r in summaries)
    walls = [r["total_seconds"] for r in summaries if r["total_seconds"]]
    print(f"runs: {len(summaries)}   total cost: ${total_cost:.4f}")
    if walls:
        p50 = statistics.median(walls)
        p95 = sorted(walls)[max(0, int(0.95 * len(walls)) - 1)]
        print(f"latency p50/p95: {p50:.1f}s / {p95:.1f}s")
    # Engine + model mix.
    by_engine: defaultdict[str, list[float]] = defaultdict(list)
    for r in summaries:
        by_engine[r["engine"] or "?"].append(r["total_cost_usd"] or 0)
    print("by engine:")
    for eng, costs in sorted(by_engine.items(), key=lambda kv: -len(kv[1])):
        print(f"  {eng:<28} runs={len(costs):>4}  sum_cost=${sum(costs):.4f}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="tech-decomposition analyze",
        description="Summarize outputs/metrics/runs.jsonl.",
    )
    p.add_argument("--since", help="Window like '24h', '7d', '15m'. Default: all.")
    p.add_argument("--by-engine", help="Substring match on engine name (e.g. 'sourcebot').")
    p.add_argument("--mode", help="Filter by mode (e.g. 'ask', 'decompose', 'compare').")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    s = get_settings()
    jsonl = s.output_dir / "metrics" / "runs.jsonl"
    if not jsonl.exists():
        print(f"no runs yet at {jsonl}")
        return
    rows = [json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()]

    summaries = _aggregate(rows)

    if args.since:
        threshold = _parse_since(args.since)
        summaries = [r for r in summaries if r["ts"] and r["ts"] >= threshold]
    if args.by_engine:
        needle = args.by_engine.lower()
        summaries = [r for r in summaries if (r["engine"] or "").lower().find(needle) != -1]
    if args.mode:
        summaries = [r for r in summaries if r["mode"] == args.mode]

    if not summaries:
        print(f"no rows match the filters in {jsonl}")
        return

    _print_table(summaries)
    _print_aggregates(summaries)


if __name__ == "__main__":
    main()
