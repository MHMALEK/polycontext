"""Side-by-side comparison harness for the local agent vs. Sourcebot.

Runs each question in `eval/questions.toml` through both engines, captures the
metrics the existing pipeline already tracks (tokens, wall time, cost,
citations, tool calls for the local agent), and writes:

- `outputs/compare/{run_id}/{qid}-{engine}.md`  -- the raw answer
- `outputs/compare/{run_id}/{qid}.md`           -- side-by-side
- `outputs/compare/{run_id}/summary.md`         -- table across all questions
- `outputs/compare/{run_id}/summary.jsonl`      -- one row per (qid, engine)

Sequential by design: avoid rate-limit collisions and keep logs readable.
"""
from __future__ import annotations

import asyncio
import json
import tomllib
import traceback
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console

from .clients.sourcebot import AskResult, SourcebotAskError, ask_sourcebot
from .config import Settings
from .engines._local_agent_impl import Answer, local_ask
from .core.models import estimate_cost_usd
from .core.usage import usage_from_result

console = Console()


@dataclass
class CitationRow:
    repo: str
    path: str
    line_start: int | None = None
    line_end: int | None = None


@dataclass
class EngineRun:
    engine: str
    question_id: str
    ok: bool
    answer: str
    citations: list[CitationRow] = field(default_factory=list)
    wall_seconds: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    model: str | None = None
    cost_usd: float | None = None
    transport: str | None = None
    tool_calls: dict[str, int] = field(default_factory=dict)
    confidence: str | None = None
    open_questions: list[str] = field(default_factory=list)
    error: str | None = None

    def to_jsonl_row(self) -> dict[str, Any]:
        d = asdict(self)
        return d


def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _load_questions(path: Path) -> list[dict[str, Any]]:
    with path.open("rb") as f:
        data = tomllib.load(f)
    questions = data.get("questions") or []
    if not isinstance(questions, list):
        raise ValueError(f"{path}: expected [[questions]] array")
    for q in questions:
        if "id" not in q or "text" not in q:
            raise ValueError(f"{path}: each question must have id and text; got {q!r}")
        q.setdefault("tags", [])
    return questions


async def _run_sourcebot(qid: str, text: str, settings: Settings) -> EngineRun:
    try:
        result: AskResult = await ask_sourcebot(text, settings=settings)
    except SourcebotAskError as e:
        return EngineRun(
            engine="sourcebot", question_id=qid, ok=False, answer="",
            error=f"SourcebotAskError: {e}",
        )
    except Exception as e:  # defensive
        return EngineRun(
            engine="sourcebot", question_id=qid, ok=False, answer="",
            error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
        )

    meta = result.metadata
    in_tok = meta.total_input_tokens if meta else None
    out_tok = meta.total_output_tokens if meta else None
    tot_tok = meta.total_tokens if meta else None
    model = meta.model_name if meta else None
    cost = (
        estimate_cost_usd(model, in_tok or 0, out_tok or 0)
        if (model and (in_tok or out_tok))
        else None
    )
    return EngineRun(
        engine="sourcebot",
        question_id=qid,
        ok=bool(result.answer.strip()),
        answer=result.answer,
        citations=[
            CitationRow(
                repo=c.repo or "",
                path=c.path or "",
                line_start=c.start_line,
                line_end=c.end_line,
            )
            for c in result.citations
        ],
        wall_seconds=result.wall_seconds,
        input_tokens=in_tok,
        output_tokens=out_tok,
        total_tokens=tot_tok,
        model=model,
        cost_usd=cost,
        transport=(meta.transport if meta else None),
    )


async def _run_local(qid: str, text: str, settings: Settings) -> EngineRun:
    import time
    t0 = time.monotonic()
    try:
        result, deps = await local_ask(text, settings)
    except Exception as e:
        return EngineRun(
            engine="local", question_id=qid, ok=False, answer="",
            wall_seconds=round(time.monotonic() - t0, 2),
            error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
        )
    wall = round(time.monotonic() - t0, 2)

    answer: Answer = result.output
    in_tok, out_tok = usage_from_result(result)
    tot_tok = ((in_tok or 0) + (out_tok or 0)) or None
    cost = estimate_cost_usd(settings.decompose_model, in_tok or 0, out_tok or 0)
    return EngineRun(
        engine="local",
        question_id=qid,
        ok=bool(answer.answer.strip()),
        answer=answer.answer,
        citations=[
            CitationRow(
                repo=c.repo, path=c.path,
                line_start=c.line_start, line_end=c.line_end,
            )
            for c in answer.citations
        ],
        wall_seconds=wall,
        input_tokens=in_tok,
        output_tokens=out_tok,
        total_tokens=tot_tok,
        model=settings.decompose_model,
        cost_usd=cost,
        transport="local",
        tool_calls=dict(deps.tool_calls),
        confidence=answer.confidence,
        open_questions=list(answer.open_questions),
    )


def _render_engine_md(question: str, run: EngineRun) -> str:
    head_label = {
        "sourcebot": "Sourcebot",
        "local": "Local agent",
    }.get(run.engine, run.engine)
    out: list[str] = [f"# {head_label} — {run.question_id}", ""]
    out.append(f"_question_: {question.strip().splitlines()[0][:200]}")
    out.append("")
    meta_bits: list[str] = []
    if run.transport:
        meta_bits.append(f"transport: `{run.transport}`")
    if run.model:
        meta_bits.append(f"model: `{run.model}`")
    if run.wall_seconds is not None:
        meta_bits.append(f"wall: {run.wall_seconds}s")
    if run.total_tokens is not None:
        meta_bits.append(f"tokens: {run.total_tokens}")
    elif run.input_tokens or run.output_tokens:
        meta_bits.append(f"tokens: {run.input_tokens or 0}/{run.output_tokens or 0}")
    if run.cost_usd is not None:
        meta_bits.append(f"cost: ${run.cost_usd:.4f}")
    if run.confidence:
        meta_bits.append(f"confidence: {run.confidence}")
    if run.tool_calls:
        meta_bits.append(
            "tools: "
            + ", ".join(f"{k}={v}" for k, v in sorted(run.tool_calls.items()))
        )
    if meta_bits:
        out.append("_" + " · ".join(meta_bits) + "_")
        out.append("")
    if run.error:
        out.append("## Error")
        out.append("```")
        out.append(run.error.strip())
        out.append("```")
        out.append("")
    out.append("## Answer")
    out.append((run.answer or "_(empty)_").strip())
    out.append("")
    if run.citations:
        out.append("## Citations")
        for c in run.citations:
            suf = ""
            if c.line_start and c.line_end and c.line_end != c.line_start:
                suf = f":L{c.line_start}-L{c.line_end}"
            elif c.line_start:
                suf = f":L{c.line_start}"
            out.append(f"- `{c.repo}/{c.path}{suf}`")
        out.append("")
    if run.open_questions:
        out.append("## Open questions")
        for q in run.open_questions:
            out.append(f"- {q}")
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def _render_side_by_side(question: dict[str, Any], sb: EngineRun, loc: EngineRun) -> str:
    qid = question["id"]
    text = question["text"]
    tags = question.get("tags") or []
    out: list[str] = [f"# {qid}", ""]
    if tags:
        out.append("_tags_: " + ", ".join(f"`{t}`" for t in tags))
        out.append("")
    out.append("## Question")
    out.append("```")
    out.append(text.strip())
    out.append("```")
    out.append("")

    def _meta_table(r: EngineRun) -> list[str]:
        rows = [
            f"| ok | {r.ok} |",
            f"| transport | `{r.transport or '—'}` |",
            f"| model | `{r.model or '—'}` |",
            f"| wall_s | {r.wall_seconds if r.wall_seconds is not None else '—'} |",
            f"| tokens (in/out/total) | {r.input_tokens or '—'} / {r.output_tokens or '—'} / {r.total_tokens or '—'} |",
            f"| cost_usd | {f'${r.cost_usd:.4f}' if r.cost_usd is not None else '—'} |",
            f"| citations | {len(r.citations)} |",
        ]
        if r.confidence:
            rows.append(f"| confidence | {r.confidence} |")
        if r.tool_calls:
            rows.append(
                "| tools | "
                + ", ".join(f"{k}={v}" for k, v in sorted(r.tool_calls.items()))
                + " |"
            )
        if r.error:
            rows.append(f"| error | `{r.error.splitlines()[0][:120]}` |")
        return ["| | |", "|---|---|", *rows]

    out.append("## Metrics — Sourcebot")
    out.extend(_meta_table(sb))
    out.append("")
    out.append("## Metrics — Local agent")
    out.extend(_meta_table(loc))
    out.append("")
    out.append("## Answer — Sourcebot")
    out.append((sb.answer or "_(empty)_").strip())
    out.append("")
    out.append("## Answer — Local agent")
    out.append((loc.answer or "_(empty)_").strip())
    out.append("")
    if sb.citations:
        out.append("## Citations — Sourcebot")
        for c in sb.citations:
            suf = (
                f":L{c.line_start}-L{c.line_end}"
                if c.line_start and c.line_end and c.line_end != c.line_start
                else (f":L{c.line_start}" if c.line_start else "")
            )
            out.append(f"- `{c.repo}/{c.path}{suf}`")
        out.append("")
    if loc.citations:
        out.append("## Citations — Local agent")
        for c in loc.citations:
            suf = (
                f":L{c.line_start}-L{c.line_end}"
                if c.line_start and c.line_end and c.line_end != c.line_start
                else (f":L{c.line_start}" if c.line_start else "")
            )
            out.append(f"- `{c.repo}/{c.path}{suf}`")
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def _render_summary(rows: list[tuple[dict[str, Any], EngineRun, EngineRun]]) -> str:
    out: list[str] = ["# Compare summary", ""]
    out.append(
        "| qid | tags | sb wall | sb tok | sb $ | sb cites | loc wall | loc tok | loc $ | loc cites | loc conf |"
    )
    out.append(
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"
    )
    sb_costs: list[float] = []
    loc_costs: list[float] = []
    sb_walls: list[float] = []
    loc_walls: list[float] = []
    for q, sb, loc in rows:
        tags = ",".join(q.get("tags") or [])
        out.append(
            "| "
            + " | ".join(
                [
                    q["id"],
                    f"`{tags}`",
                    f"{sb.wall_seconds or '—'}",
                    f"{sb.total_tokens or '—'}",
                    f"${sb.cost_usd:.4f}" if sb.cost_usd is not None else "—",
                    str(len(sb.citations)),
                    f"{loc.wall_seconds or '—'}",
                    f"{loc.total_tokens or '—'}",
                    f"${loc.cost_usd:.4f}" if loc.cost_usd is not None else "—",
                    str(len(loc.citations)),
                    loc.confidence or "—",
                ]
            )
            + " |"
        )
        if sb.cost_usd is not None:
            sb_costs.append(sb.cost_usd)
        if loc.cost_usd is not None:
            loc_costs.append(loc.cost_usd)
        if sb.wall_seconds is not None:
            sb_walls.append(sb.wall_seconds)
        if loc.wall_seconds is not None:
            loc_walls.append(loc.wall_seconds)
    out.append("")
    out.append("## Aggregates")

    def _agg(name: str, vals: list[float]) -> str:
        if not vals:
            return f"- **{name}**: no data"
        vals_sorted = sorted(vals)
        n = len(vals_sorted)
        median = vals_sorted[n // 2]
        return (
            f"- **{name}** (n={n}): sum={sum(vals):.4f}, median={median:.4f}, "
            f"max={max(vals):.4f}"
        )

    out.append(_agg("Sourcebot wall_s", sb_walls))
    out.append(_agg("Local wall_s", loc_walls))
    out.append(_agg("Sourcebot cost_usd", sb_costs))
    out.append(_agg("Local cost_usd", loc_costs))
    return "\n".join(out) + "\n"


async def run_compare(
    questions_path: Path,
    settings: Settings,
    *,
    only_ids: list[str] | None = None,
    skip_engine: str | None = None,
) -> Path:
    questions = _load_questions(questions_path)
    if only_ids:
        wanted = set(only_ids)
        questions = [q for q in questions if q["id"] in wanted]
        if not questions:
            raise ValueError(f"no questions matched ids={only_ids}")

    run_id = _now_stamp()
    out_root = settings.output_dir / "compare" / run_id
    out_root.mkdir(parents=True, exist_ok=True)
    console.print(f"[bold]compare run[/]: {run_id}  → {out_root}")
    console.print(f"  questions: {len(questions)}  · repos_root: {settings.repos_root}")

    summary_jsonl = out_root / "summary.jsonl"
    summary_fh = summary_jsonl.open("w")
    pairs: list[tuple[dict[str, Any], EngineRun, EngineRun]] = []

    try:
        for i, q in enumerate(questions, 1):
            qid = q["id"]
            text = q["text"]
            console.print(f"\n[bold cyan][{i}/{len(questions)}][/] {qid}")

            # Sourcebot first (network call; quicker on retry if it errors).
            if skip_engine == "sourcebot":
                sb = EngineRun(engine="sourcebot", question_id=qid, ok=False, answer="", error="skipped")
            else:
                console.print("  → sourcebot ...", end="")
                sb = await _run_sourcebot(qid, text, settings)
                status = "[green]ok[/]" if sb.ok else f"[red]fail[/] ({sb.error.splitlines()[0][:80] if sb.error else 'empty'})"
                console.print(f" {status} ({sb.wall_seconds or '?'}s)")

            if skip_engine == "local":
                loc = EngineRun(engine="local", question_id=qid, ok=False, answer="", error="skipped")
            else:
                console.print("  → local ...", end="")
                loc = await _run_local(qid, text, settings)
                status = "[green]ok[/]" if loc.ok else f"[red]fail[/] ({loc.error.splitlines()[0][:80] if loc.error else 'empty'})"
                console.print(f" {status} ({loc.wall_seconds or '?'}s)")

            (out_root / f"{qid}-sourcebot.md").write_text(_render_engine_md(text, sb))
            (out_root / f"{qid}-local.md").write_text(_render_engine_md(text, loc))
            (out_root / f"{qid}.md").write_text(_render_side_by_side(q, sb, loc))
            summary_fh.write(json.dumps(sb.to_jsonl_row(), default=str) + "\n")
            summary_fh.write(json.dumps(loc.to_jsonl_row(), default=str) + "\n")
            summary_fh.flush()

            pairs.append((q, sb, loc))
    finally:
        summary_fh.close()

    summary_md = out_root / "summary.md"
    summary_md.write_text(_render_summary(pairs))
    console.print(f"\n[green]✓[/] summary: [bold]{summary_md}[/]")
    return out_root
