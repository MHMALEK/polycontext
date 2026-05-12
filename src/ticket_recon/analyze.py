"""Summarize outputs/metrics/runs.jsonl: per-ticket cost, latency, retrieval breakdown."""
from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path

from .config import get_settings


def main() -> None:
    s = get_settings()
    p = s.output_dir / "metrics" / "runs.jsonl"
    if not p.exists():
        print(f"no runs yet at {p}")
        return
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    if not rows:
        print("no runs yet"); return

    print(f"{'run_id':<24} {'ticket':<12} {'mode':<6} {'subs':>4} {'tot_s':>7} {'enr_s':>7} "
          f"{'ret_s':>7} {'dec_s':>7} {'in_tok':>7} {'out_tok':>7} {'cost':>9} jira tools")
    print("-" * 130)
    src_total: Counter = Counter()
    total_cost = 0.0
    for r in rows:
        rid = r["run_id"][:22]
        tk = (r.get("ticket_key") or "—")[:11]
        in_tok = (r["enrich"].get("input_tokens") or 0) + (r["decompose"].get("input_tokens") or 0)
        out_tok = (r["enrich"].get("output_tokens") or 0) + (r["decompose"].get("output_tokens") or 0)
        cost = r.get("total_cost_usd")
        jira = "✓" if r.get("posted_to_jira") else "—"
        cost_str = f"${cost:.4f}" if cost is not None else "—"
        if cost: total_cost += cost
        src = r.get("retrieval", {}).get("by_source", {})
        for k, v in src.items():
            src_total[k] += v
        mode = (r.get("mode") or "?")[:5]
        if r.get("escalated_to_deep"):
            mode = f"{mode}*"
        tools_total = sum((r.get("deep_tool_calls") or {}).values())
        tools_col = str(tools_total) if tools_total else "—"
        print(f"{rid:<24} {tk:<12} {mode:<6} {r.get('subtask_count',0):>4} "
              f"{r['total_seconds']:>7.2f} {r['enrich']['seconds']:>7.2f} "
              f"{r['retrieval']['seconds']:>7.2f} {r['decompose']['seconds']:>7.2f} "
              f"{in_tok:>7} {out_tok:>7} {cost_str:>9} {jira:<4} {tools_col}")
    print("-" * 130)
    print(f"runs: {len(rows)}   total cost: ${total_cost:.4f}   "
          f"retrieval mix: {', '.join(f'{k}={v}' for k,v in src_total.most_common())}")
    if rows:
        totals = [r["total_seconds"] for r in rows]
        print(f"latency p50/p95: {statistics.median(totals):.1f}s / "
              f"{sorted(totals)[max(0,int(0.95*len(totals))-1)]:.1f}s")


if __name__ == "__main__":
    main()
