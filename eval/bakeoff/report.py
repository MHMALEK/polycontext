"""Report writer.

Reads a run directory produced by ``runner.run_bakeoff`` and emits:

  * ``report.md``    — leaderboard, per-case previews, metrics matrix, **full answers**
    (especially for ``ask``), and scoring check breakdowns
  * ``summary.json`` — flattened rows suitable for spreadsheets and scripts

The summary is one row per (case, adapter) with adapter-reported telemetry
(tokens, tool calls, cost) when present in ``response.metrics``.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .cases import discover_cases


def _fence_block(body: str, lang: str = "markdown") -> str:
    """Use a Markdown fence tall enough so ``body`` cannot break out."""
    body = body.rstrip() + ("\n" if body and not body.endswith("\n") else "")
    n = 3
    while True:
        fence = "`" * n
        if fence not in body:
            return f"{fence}{lang}\n{body}{fence}\n"
        n += 1


_WS = re.compile(r"\s+")


def _one_line_answer_peek(answer: str, max_len: int = 160) -> str:
    s = _WS.sub(" ", answer.strip())
    return s[:max_len] + ("…" if len(s) > max_len else "")


def _safe_float(v) -> float | None:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _esc(s: str) -> str:
    return s.replace("|", "\\|").replace("\n", " ")


def _flatten_ask_observables(rec: dict[str, Any]) -> dict[str, Any]:
    """Flatten ``ask`` payloads for CSV/summary rows and leaderboard."""
    raw: dict[str, Any] = {
        "answer_chars": None,
        "answer_words": None,
        "citation_count": None,
        "tokens_in": None,
        "tokens_out": None,
        "cost_usd": None,
        "tool_calls": None,
        "model": None,
    }
    if not rec.get("ok") or rec.get("job") != "ask":
        return raw
    r = rec.get("response") or {}
    ans = r.get("answer") or ""
    cits = r.get("citations") or []
    m = r.get("metrics") or {}

    tin = m.get("tokens_in")
    tout = m.get("tokens_out")
    tc = m.get("tool_calls")
    raw.update({
        "answer_chars": len(ans),
        "answer_words": len(ans.split()),
        "citation_count": len(cits),
        "tokens_in": int(tin) if tin is not None else None,
        "tokens_out": int(tout) if tout is not None else None,
        "cost_usd": _safe_float(m.get("cost_usd")),
        "tool_calls": int(tc) if tc is not None else None,
        "model": m.get("model"),
    })
    extra = m.get("extra") or {}
    # Surface common SDK extras without nesting in summary.json
    for k in ("adapter_duration_ms", "turns", "provider_latency_ms"):
        if k in extra and extra[k] is not None:
            raw[f"metric_extra_{k}"] = extra[k]
    return raw


def _citations_lines(resp: dict[str, Any], limit: int = 80) -> str:
    cites = resp.get("citations") or []
    if not cites:
        return "(no structured citations)"
    lines: list[str] = []
    for c in cites[:limit]:
        if not isinstance(c, dict):
            lines.append(str(c))
            continue
        repo = c.get("repo", "")
        path = c.get("path", "")
        ls = c.get("line_start", "?")
        le = c.get("line_end", "?")
        lines.append(f"- `{repo}` `{path}` L{ls}–{le}")
    if len(cites) > limit:
        lines.append(f"_… plus {len(cites) - limit} more citation(s)._")
    return "\n".join(lines)


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return ""
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join("---" for _ in headers) + " |")
    out.append("\n".join("| " + " | ".join(cells) + " |" for cells in rows))
    out.append("")
    return "\n".join(out)


def _response_preview(rec: dict[str, Any]) -> str:
    """One short cell of the per-case overview table."""
    if not rec.get("ok"):
        return "—"
    job = rec.get("job")
    r = rec.get("response") or {}
    if job == "ask":
        peek = _one_line_answer_peek(r.get("answer") or "", 140)
        n = len((r.get("citations") or []))
        return _esc(f"{peek[:120]}  · chars={len(r.get('answer') or '')} cit={n}")
    if job == "decompose":
        d = r.get("decomposition") or {}
        st = d.get("subtasks") or []
        return f"{len(st)} subtasks, repos: {', '.join((d.get('affected_repos') or [])[:3])}"
    return ""


def _gold_lookup(eval_dir: Path | None) -> dict[str, str]:
    """case_id → golden reference text (ask or decompose)."""
    if eval_dir is None or not eval_dir.is_dir():
        return {}
    out: dict[str, str] = {}
    for c in discover_cases(eval_dir):
        exp = c.expected or {}
        gold = exp.get("gold_answer") or exp.get("gold_decomposition") or ""
        if gold:
            out[c.id] = gold.strip()
    return out


def _render_ask_full_section(
    recs: list[dict[str, Any]],
    *,
    case_id: str = "",
    gold_by_case: dict[str, str] | None = None,
) -> str:
    ask_recs = [r for r in recs if r.get("job") == "ask" and r.get("ok")]
    if not ask_recs:
        return ""

    gold = (gold_by_case or {}).get(case_id, "").strip()
    metrics_rows: list[list[str]] = []
    for rec in sorted(ask_recs, key=lambda r: -(r.get("score") or {}).get("overall", 0.0)):
        o = _flatten_ask_observables(rec)
        rm = (rec.get("response") or {}).get("metrics") or {}
        sdk_ms = rm.get("duration_ms") or o.get("metric_extra_adapter_duration_ms") or ""
        metrics_rows.append([
            rec.get("adapter", ""),
            str(rec.get("duration_ms", 0)),
            str(o.get("answer_chars") if o["answer_chars"] is not None else "—"),
            str(o["citation_count"] if o["citation_count"] is not None else "—"),
            str(o["tokens_in"] if o["tokens_in"] is not None else "—"),
            str(o["tokens_out"] if o["tokens_out"] is not None else "—"),
            str(o["tool_calls"] if o["tool_calls"] is not None else "—"),
            f"${o['cost_usd']:.4f}" if o.get("cost_usd") is not None else "—",
            str(o["model"]) if o.get("model") else "—",
            str(rec.get("run_id") or "—"),
            str(sdk_ms) if sdk_ms != "" else "—",
        ])

    hdr = ["Adapter", "wall ms", "chars", "cites", "tok in", "tok out", "tools", "$", "model", "run_id", "sdk ms"]
    blk = [_markdown_table(hdr, metrics_rows)]

    if gold:
        blk.append("")
        blk.append("#### Golden reference answer")
        blk.append("")
        blk.append("_Compare each adapter answer below to this reference._")
        blk.append("")
        blk.append(_fence_block(gold))
        blk.append("")

    blk.append("")
    blk.append("_Wall ms_: eval harness latency. **sdk ms** is ``metrics.duration_ms`` from the adapter when set._")
    blk.append("")
    blk.append("#### Full answer bodies")
    blk.append("")
    blk.append("_Compare verbatim outputs below — use your editor Search or Beyond Compare across saved run dirs._")

    for rec in sorted(ask_recs, key=lambda r: -(r.get("score") or {}).get("overall", 0.0)):
        adapter = rec.get("adapter", "?")
        r = rec.get("response") or {}
        answer = r.get("answer") or "(empty)"
        o = _flatten_ask_observables(rec)

        hdr_line = (
            f"`{adapter}` — chars={o.get('answer_chars')} words={o.get('answer_words')} · "
            f"citations={o.get('citation_count')} · "
            f"runner_wall_ms={rec.get('duration_ms')} · "
            f"score={(rec.get('score') or {}).get('overall', 0):.2f}"
        )
        blk.append(f"<details>")
        blk.append("")
        blk.append(f"<summary>{hdr_line}</summary>")
        blk.append("")
        blk.append("")
        blk.append("_Structured citations:_")
        blk.append("")
        blk.append(_citations_lines(r))
        blk.append("")
        blk.append("")
        blk.append(_fence_block(answer))
        blk.append("")
        blk.append("")
        blk.append("</details>")
        blk.append("")

    return "\n".join(blk)


@dataclass
class _Totals:
    runs: int = 0
    successes: int = 0
    score_sum: float = 0.0
    cost_sum: float = 0.0
    duration_ms_sum: int = 0
    # ask-only aggregates (successful ask rows only)
    ask_succ: int = 0
    ask_chars_sum: int = 0
    cit_sum: int = 0
    tok_in_sum: int = 0
    tok_in_n: int = 0
    tok_out_sum: int = 0
    tok_out_n: int = 0
    tool_calls_sum: int = 0
    tool_calls_n: int = 0


def _leaderboard(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by: dict[str, _Totals] = defaultdict(_Totals)
    for row in summary_rows:
        name = row["adapter"]
        t = by[name]
        t.runs += 1
        t.duration_ms_sum += int(row.get("duration_ms") or 0)
        cost = row.get("cost_usd")
        if cost is not None:
            t.cost_sum += float(cost)

        if row.get("ok"):
            t.successes += 1
            sc = row.get("score")
            if sc is None:
                sc = 0.0
            t.score_sum += float(sc)

        if row.get("ok") and row.get("job") == "ask":
            obs = row.get("ask_observed") or {}
            if obs.get("answer_chars") is not None:
                t.ask_succ += 1
                t.ask_chars_sum += int(obs["answer_chars"])
                if obs.get("citation_count") is not None:
                    t.cit_sum += int(obs["citation_count"])
                if obs.get("tokens_in") is not None:
                    t.tok_in_sum += int(obs["tokens_in"])
                    t.tok_in_n += 1
                if obs.get("tokens_out") is not None:
                    t.tok_out_sum += int(obs["tokens_out"])
                    t.tok_out_n += 1
                if obs.get("tool_calls") is not None:
                    t.tool_calls_sum += int(obs["tool_calls"])
                    t.tool_calls_n += 1

    out: list[dict[str, Any]] = []
    for name, t in by.items():
        succ_rate = (t.successes / t.runs) if t.runs else 0.0
        avg_score = (t.score_sum / t.successes) if t.successes else 0.0
        avg_ms = (t.duration_ms_sum / t.runs) if t.runs else 0.0
        n_a = t.ask_succ if t.ask_succ else 0

        row = {
            "adapter": name,
            "runs": t.runs,
            "successes": t.successes,
            "success_rate": round(succ_rate, 3),
            "avg_score": round(avg_score, 3),
            "avg_duration_ms": int(avg_ms),
            "total_cost_usd": round(t.cost_sum, 4),
            "avg_answer_chars": round(t.ask_chars_sum / n_a, 1) if n_a else None,
            "avg_citations": round(t.cit_sum / n_a, 2) if n_a else None,
            "avg_tokens_in": round(t.tok_in_sum / t.tok_in_n, 1) if t.tok_in_n else None,
            "avg_tokens_out": round(t.tok_out_sum / t.tok_out_n, 1) if t.tok_out_n else None,
            "avg_tool_calls": round(t.tool_calls_sum / t.tool_calls_n, 2) if t.tool_calls_n else None,
        }
        out.append(row)

    out.sort(key=lambda r: (-(r["avg_score"] or 0), -(r["avg_answer_chars"] or 0), r["avg_duration_ms"]))
    return out


def render_report(run_dir: Path, *, eval_dir: Path | None = None) -> tuple[Path, Path]:
    """Write ``report.md`` and ``summary.json`` next to the runs/ folder.

    Returns ``(report_path, summary_path)``.
    """
    if eval_dir is None:
        eval_dir = run_dir.parent.parent if run_dir.parent.name == "outputs" else None
    gold_by_case = _gold_lookup(eval_dir) if eval_dir else {}

    manifest = json.loads((run_dir / "manifest.json").read_text())
    runs_dir = run_dir / "runs"
    if not runs_dir.is_dir():
        raise FileNotFoundError(f"no runs/ under {run_dir}")

    summary: list[dict[str, Any]] = []
    rows_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)

    cases_meta = {c["id"]: c for c in manifest["cases"]}

    for case_dir in sorted(runs_dir.iterdir()):
        if not case_dir.is_dir():
            continue
        case_id = case_dir.name
        for run_file in sorted(case_dir.glob("*.json")):
            rec = json.loads(run_file.read_text())
            adapter = rec.get("adapter") or run_file.stem
            cost = None
            r = rec.get("response") or {}
            m = r.get("metrics") or {}
            cost = _safe_float(m.get("cost_usd"))
            ask_observed = _flatten_ask_observables(rec)

            row = {
                "case_id": case_id,
                "adapter": adapter,
                "job": rec.get("job"),
                "ok": rec.get("ok"),
                "status": rec.get("status"),
                "score": (rec.get("score") or {}).get("overall", 0.0),
                "duration_ms": rec.get("duration_ms", 0),
                "cost_usd": cost,
                "tokens_in": m.get("tokens_in"),
                "tokens_out": m.get("tokens_out"),
                "tool_calls": m.get("tool_calls"),
                "adapter_model": m.get("model"),
                "run_id": rec.get("run_id"),
                "error": rec.get("error"),
                "ask_observed": {k: v for k, v in ask_observed.items() if v is not None},
            }
            summary.append(row)
            rows_by_case[case_id].append(rec)

    leaderboard = _leaderboard(summary)

    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps({
        "manifest": manifest,
        "leaderboard": leaderboard,
        "rows": summary,
    }, indent=2))

    report_path = run_dir / "report.md"
    report_path.write_text(
        _markdown(manifest, leaderboard, rows_by_case, cases_meta, gold_by_case=gold_by_case)
    )
    return report_path, summary_path


def _markdown(manifest: dict[str, Any], leaderboard: list[dict[str, Any]],
              rows_by_case: dict[str, list[dict[str, Any]]], cases_meta: dict[str, dict],
              *, gold_by_case: dict[str, str] | None = None) -> str:
    lines: list[str] = []
    lines.append("# Adapter bake-off report")
    lines.append("")
    lines.append(f"- Started: `{manifest.get('started_at', '?')}`")
    lines.append(f"- Finished: `{manifest.get('finished_at', '(in progress)')}`")
    rm = manifest.get("run_meta") or {}
    if rm:
        lines.append(
            f"- Client: **`{rm.get('client_backend', '?')}`**"
            + (f" ({rm.get('client_base_url')})" if rm.get("client_base_url") else "")
        )
    lines.append(f"- Adapters: {', '.join(f'`{a}`' for a in manifest.get('adapters', []))}")
    lines.append(f"- Cases: {len(manifest.get('cases', []))}")
    lines.append("")

    lines.append("## Leaderboard")
    lines.append("")
    lines.append("| Adapter | runs | succ | avg score | avg wall ms | total $ | avg chars¹ | avg cites¹ | avg tok in² | avg tok out² | avg tools² |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for row in leaderboard:
        chars = row.get("avg_answer_chars")
        cites = row.get("avg_citations")
        tin = row.get("avg_tokens_in")
        tout = row.get("avg_tokens_out")
        tools = row.get("avg_tool_calls")
        lines.append(
            f"| `{row['adapter']}` | {row['runs']} | "
            f"{row['successes']}/{row['runs']} ({int(row['success_rate']*100)}%) | "
            f"{row['avg_score']:.2f} | {row['avg_duration_ms']} | "
            f"${row['total_cost_usd']:.4f} | "
            f"{chars if chars is not None else '—'} | "
            f"{cites if cites is not None else '—'} | "
            f"{tin if tin is not None else '—'} | "
            f"{tout if tout is not None else '—'} | "
            f"{tools if tools is not None else '—'} |"
        )
    lines.append("")
    lines.append(
        "¹ **`ask`** successes only · ² averages include only runs where "
        "**that** metric key was populated by the adapter."
    )
    lines.append("")

    lines.append("## Per-case overview")
    lines.append("")
    for case_id in sorted(rows_by_case.keys()):
        meta = cases_meta.get(case_id, {})
        recs = rows_by_case[case_id]
        lines.append(f"### `{case_id}`")
        lines.append("")
        if meta.get("tags"):
            lines.append(f"_tags:_ {', '.join('`' + t + '`' for t in meta['tags'])}")
            lines.append("")
        lines.append("| Adapter | ok | score | wall ms | response preview |")
        lines.append("|---|---|---|---|---|")
        for rec in sorted(recs, key=lambda r: -(r.get("score") or {}).get("overall", 0.0)):
            adapter = rec.get("adapter")
            ok = "✅" if rec.get("ok") else f"❌ ({rec.get('status')})"
            score = (rec.get("score") or {}).get("overall", 0.0)
            ms = rec.get("duration_ms", 0)
            preview = _response_preview(rec)
            lines.append(f"| `{adapter}` | {ok} | {score:.2f} | {ms} | {preview} |")
        lines.append("")

        # Ask: metrics matrix + verbatim answers live here.
        lines.append(_render_ask_full_section(recs, case_id=case_id, gold_by_case=gold_by_case))
        gold = (gold_by_case or {}).get(case_id, "").strip()
        decompose_recs = [r for r in recs if r.get("job") == "decompose" and r.get("ok")]
        if gold and decompose_recs:
            lines.append("#### Golden reference decomposition")
            lines.append("")
            lines.append(_fence_block(gold))
            lines.append("")

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
            lines.append(f"<details><summary>`{rec['adapter']}` rubric ({score.get('overall', 0):.2f})</summary>")
            lines.append("")
            for c in checks:
                mark = "✅" if c.get("ok") else "❌"
                w = c.get("weight", 1)
                lines.append(f"- {mark} `{c.get('name')}` (w={w}) — {c.get('detail', '')}")
            lines.append("")
            lines.append("</details>")
            lines.append("")

    return "\n".join(lines) + "\n"
