"""Report writer.

Reads a run directory produced by ``runner.run_bakeoff`` and emits:

  * ``report.md``    — human-readable markdown, side-by-side per case
  * ``summary.json`` — machine-readable leaderboard for further analysis

The summary is a flat list — one row per (case, adapter) — so it's trivial
to load into pandas / a spreadsheet / a quick chart later.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class _AdapterTotals:
    runs: int = 0
    successes: int = 0
    score_sum: float = 0.0
    cost_sum: float = 0.0
    duration_ms_sum: int = 0


def render_report(run_dir: Path) -> tuple[Path, Path]:
    """Write ``report.md`` and ``summary.json`` next to the runs/ folder.

    Returns ``(report_path, summary_path)``.
    """
    manifest = json.loads((run_dir / "manifest.json").read_text())
    cases_meta = {c["id"]: c for c in manifest["cases"]}
    runs_dir = run_dir / "runs"
    if not runs_dir.is_dir():
        raise FileNotFoundError(f"no runs/ under {run_dir}")

    summary: list[dict[str, Any]] = []
    totals: dict[str, _AdapterTotals] = defaultdict(_AdapterTotals)
    rows_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for case_dir in sorted(runs_dir.iterdir()):
        if not case_dir.is_dir():
            continue
        case_id = case_dir.name
        for run_file in sorted(case_dir.glob("*.json")):
            rec = json.loads(run_file.read_text())
            adapter = rec.get("adapter") or run_file.stem
            cost = _safe_float(rec.get("response", {}).get("metrics", {}).get("cost_usd"))
            row = {
                "case_id": case_id,
                "adapter": adapter,
                "job": rec.get("job"),
                "ok": rec.get("ok"),
                "status": rec.get("status"),
                "score": (rec.get("score") or {}).get("overall", 0.0),
                "duration_ms": rec.get("duration_ms", 0),
                "cost_usd": cost,
                "error": rec.get("error"),
            }
            summary.append(row)
            rows_by_case[case_id].append(rec)
            t = totals[adapter]
            t.runs += 1
            if rec.get("ok"):
                t.successes += 1
                t.score_sum += row["score"]
            t.cost_sum += cost or 0.0
            t.duration_ms_sum += rec.get("duration_ms", 0) or 0

    leaderboard = _leaderboard(totals)

    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps({
        "manifest": manifest,
        "leaderboard": leaderboard,
        "rows": summary,
    }, indent=2))

    report_path = run_dir / "report.md"
    report_path.write_text(_markdown(manifest, leaderboard, rows_by_case, cases_meta))
    return report_path, summary_path


def _leaderboard(totals: dict[str, _AdapterTotals]) -> list[dict[str, Any]]:
    out = []
    for name, t in totals.items():
        succ_rate = (t.successes / t.runs) if t.runs else 0.0
        avg_score = (t.score_sum / t.successes) if t.successes else 0.0
        avg_ms = (t.duration_ms_sum / t.runs) if t.runs else 0.0
        out.append({
            "adapter": name,
            "runs": t.runs,
            "successes": t.successes,
            "success_rate": round(succ_rate, 3),
            "avg_score": round(avg_score, 3),
            "avg_duration_ms": int(avg_ms),
            "total_cost_usd": round(t.cost_sum, 4),
        })
    out.sort(key=lambda r: (-r["avg_score"], r["avg_duration_ms"]))
    return out


def _markdown(manifest, leaderboard, rows_by_case, cases_meta) -> str:
    lines: list[str] = []
    lines.append("# Adapter bake-off report")
    lines.append("")
    lines.append(f"- Started: `{manifest.get('started_at', '?')}`")
    lines.append(f"- Finished: `{manifest.get('finished_at', '(in progress)')}`")
    lines.append(f"- Adapters: {', '.join(f'`{a}`' for a in manifest.get('adapters', []))}")
    lines.append(f"- Cases: {len(manifest.get('cases', []))}")
    lines.append("")

    lines.append("## Leaderboard")
    lines.append("")
    lines.append("| Adapter | runs | success | avg score | avg ms | total $ |")
    lines.append("|---|---|---|---|---|---|")
    for row in leaderboard:
        lines.append(
            f"| `{row['adapter']}` | {row['runs']} | "
            f"{row['successes']}/{row['runs']} ({int(row['success_rate']*100)}%) | "
            f"{row['avg_score']:.2f} | {row['avg_duration_ms']} | "
            f"${row['total_cost_usd']:.4f} |"
        )
    lines.append("")

    lines.append("## Per-case results")
    lines.append("")
    for case_id in sorted(rows_by_case.keys()):
        meta = cases_meta.get(case_id, {})
        recs = rows_by_case[case_id]
        lines.append(f"### `{case_id}`")
        lines.append("")
        if meta.get("tags"):
            lines.append(f"_tags:_ {', '.join('`' + t + '`' for t in meta['tags'])}")
            lines.append("")
        lines.append("| Adapter | ok | score | ms | response preview |")
        lines.append("|---|---|---|---|---|")
        for rec in sorted(recs, key=lambda r: -(r.get("score") or {}).get("overall", 0.0)):
            adapter = rec.get("adapter")
            ok = "✅" if rec.get("ok") else f"❌ ({rec.get('status')})"
            score = (rec.get("score") or {}).get("overall", 0.0)
            ms = rec.get("duration_ms", 0)
            preview = _response_preview(rec)
            lines.append(f"| `{adapter}` | {ok} | {score:.2f} | {ms} | {preview} |")
        lines.append("")
        # Per-adapter check breakdown lives in details blocks so the top-level
        # table stays scannable.
        for rec in recs:
            if not rec.get("ok"):
                lines.append(f"<details><summary>`{rec['adapter']}` error</summary>")
                lines.append("")
                lines.append("```")
                lines.append(str(rec.get("error", "(no detail)"))[:2000])
                lines.append("```")
                lines.append("</details>")
                lines.append("")
                continue
            score = rec.get("score") or {}
            checks = score.get("checks") or []
            if not checks:
                continue
            lines.append(f"<details><summary>`{rec['adapter']}` checks ({score.get('overall', 0):.2f})</summary>")
            lines.append("")
            for c in checks:
                mark = "✅" if c.get("ok") else "❌"
                w = c.get("weight", 1)
                lines.append(f"- {mark} `{c.get('name')}` (w={w}) — {c.get('detail', '')}")
            lines.append("")
            lines.append("</details>")
            lines.append("")

    return "\n".join(lines) + "\n"


def _response_preview(rec: dict[str, Any]) -> str:
    """One short cell of the per-case table."""
    if not rec.get("ok"):
        return "—"
    job = rec.get("job")
    r = rec.get("response") or {}
    if job == "ask":
        ans = (r.get("answer") or "").strip().splitlines()
        return _esc(ans[0][:120]) if ans else "—"
    if job == "decompose":
        d = r.get("decomposition") or {}
        st = d.get("subtasks") or []
        return f"{len(st)} subtasks, repos: {', '.join((d.get('affected_repos') or [])[:3])}"
    if job == "implement":
        return f"branch=`{r.get('branch', '?')}` files={len(r.get('files_changed') or [])} mr={r.get('mr_url') or '—'}"
    return ""


def _safe_float(v) -> float | None:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _esc(s: str) -> str:
    return s.replace("|", "\\|").replace("\n", " ")
