"""Summarize ``outputs/metrics/runs.jsonl`` — per-run cost / latency / stage mix.

Reads the JSONL written by ``core.metrics.JsonlMetricsObserver`` (``{type: stage}``
and ``{type: run}`` rows). Older rows from the v0.1 pipeline (no ``type`` field,
``mode``+``enrich``+``decompose`` nested) are skipped — they predate the new
schema and aren't worth back-translating.
"""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from typing import Any

from .config import get_settings


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group stage + run rows by run_id; return one summary per run."""
    by_run: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"stages": [], "total_seconds": None, "total_cost_usd": 0.0, "mode": "?"}
    )
    for r in rows:
        rt = r.get("type")
        rid = r.get("run_id")
        if not rid or not rt:
            continue
        bucket = by_run[rid]
        bucket["mode"] = r.get("mode", bucket["mode"])
        if rt == "stage":
            bucket["stages"].append(r)
            cost = r.get("cost_usd")
            if cost is not None:
                bucket["total_cost_usd"] += cost
        elif rt == "run":
            bucket["total_seconds"] = r.get("total_seconds")
            if r.get("total_cost_usd") is not None:
                bucket["total_cost_usd"] = r["total_cost_usd"]
    summaries: list[dict[str, Any]] = []
    for rid, b in by_run.items():
        stages = b["stages"]
        # Pick engine model + tokens from the engine stage if present.
        engine_stage = next((s for s in stages if s.get("stage") == "engine"), None)
        eng_name = engine_stage.get("strategy") if engine_stage else "?"
        model = engine_stage.get("model") if engine_stage else None
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
        })
    return summaries


def main() -> None:
    s = get_settings()
    p = s.output_dir / "metrics" / "runs.jsonl"
    if not p.exists():
        print(f"no runs yet at {p}")
        return
    rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    summaries = _aggregate(rows)
    if not summaries:
        print(f"no v2 metrics in {p} yet")
        return

    print(
        f"{'run_id':<14} {'mode':<10} {'engine':<24} {'model':<24} "
        f"{'wall_s':>7} {'in_tok':>8} {'out_tok':>8} {'cost':>9}"
    )
    print("-" * 110)
    for r in summaries:
        rid = r["run_id"][:14]
        mode = (r["mode"] or "?")[:10]
        eng = (r["engine"] or "?")[:24]
        model = (r["model"] or "—")[:24]
        wall = r["total_seconds"] or 0.0
        in_t = r["input_tokens"] or "—"
        out_t = r["output_tokens"] or "—"
        cost = r["total_cost_usd"]
        cost_str = f"${cost:.4f}" if cost is not None else "—"
        print(f"{rid:<14} {mode:<10} {eng:<24} {model:<24} {wall:>7.2f} {str(in_t):>8} {str(out_t):>8} {cost_str:>9}")
    print("-" * 110)
    total_cost = sum((r["total_cost_usd"] or 0) for r in summaries)
    total_wall = [r["total_seconds"] for r in summaries if r["total_seconds"]]
    print(f"runs: {len(summaries)}   total cost: ${total_cost:.4f}")
    if total_wall:
        p50 = statistics.median(total_wall)
        p95 = sorted(total_wall)[max(0, int(0.95 * len(total_wall)) - 1)]
        print(f"latency p50/p95: {p50:.1f}s / {p95:.1f}s")


if __name__ == "__main__":
    main()
