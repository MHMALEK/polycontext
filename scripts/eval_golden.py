#!/usr/bin/env python3
"""Run adapter bake-off cases against golden reference answers.

Golden answers live in ``eval/questions.toml`` (``gold_answer`` for ask) and
decompose YAML (``gold_decomposition``). Scoring uses the existing bake-off
runner plus ``--use-judge`` (Gemini Flash compares candidate vs reference).

This is for *real* end-to-end runs — not mocked unit tests. Prerequisites:

  - Repos indexed in Sourcebot (for ``sourcebot`` / ``sourcebot_ollama``)
  - ``make adapters`` or agent-node on ``AGENT_NODE_URL``
  - Ollama running when testing ``cline_sdk`` (``CLINE_SDK_USE_OLLAMA``) or
    ``sourcebot_ollama``
  - ``GEMINI_API_KEY`` for LLM-as-judge (rule-based rubric still runs without it)

Examples::

    uv run python scripts/eval_golden.py --quick
    uv run python scripts/eval_golden.py --job ask --adapters sourcebot,cline_sdk,sourcebot_ollama
    uv run python scripts/eval_golden.py --job decompose --ids d2-farm-name-validation
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import textwrap
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Reuse bake-off dotenv shim before Settings import.
from eval.bakeoff.cli import _load_dotenv_into_environ  # noqa: E402

_load_dotenv_into_environ()

from eval.bakeoff.cases import Case, discover_cases, filter_cases  # noqa: E402
from eval.bakeoff.client import build_client  # noqa: E402
from eval.bakeoff.report import render_report  # noqa: E402
from eval.bakeoff.runner import new_output_dir, run_bakeoff  # noqa: E402
from tech_decomposition.config import get_settings  # noqa: E402

EVAL_DIR = PROJECT_ROOT / "eval"
OUTPUTS_DIR = EVAL_DIR / "outputs"

# Full comparison default (heavy — runs multiple adapters; prefer --minimal locally).
DEFAULT_ADAPTERS = "sourcebot,cline_sdk,sourcebot_ollama"
# One question, one adapter — low RAM; good for iterating on Ollama quality.
MINIMAL_ASK_ID = "q-user-roles"
MINIMAL_DECOMPOSE_ID = "d2-farm-name-validation"
MINIMAL_DEFAULT_ADAPTER = "sourcebot_ollama"
# Compare: ADAPTER=sourcebot_ollama_max for experimental max-context RAG
MINIMAL_DEFAULT_TIMEOUT = 420.0
QUICK_CASE_IDS = ["q-user-roles", "q-farm-name-unicode"]
QUICK_DECOMPOSE_IDS = ["d2-farm-name-validation"]


def _cases_with_gold(cases: list[Case]) -> list[Case]:
    out: list[Case] = []
    for c in cases:
        exp = c.expected or {}
        if c.job == "ask" and exp.get("gold_answer"):
            out.append(c)
        elif c.job == "decompose" and (exp.get("gold_decomposition") or exp.get("gold_answer")):
            out.append(c)
    return out


def _gold_text(case: Case) -> str:
    exp = case.expected or {}
    if case.job == "decompose":
        return (exp.get("gold_decomposition") or exp.get("gold_answer") or "").strip()
    return (exp.get("gold_answer") or "").strip()


def _judge_scores(score: dict) -> tuple[float | None, float | None, str]:
    cov = acc = None
    notes = score.get("notes") or ""
    for chk in score.get("checks") or []:
        name = chk.get("name") or ""
        detail = chk.get("detail") or ""
        try:
            val = float(detail)
        except (TypeError, ValueError):
            continue
        if name == "gold_coverage":
            cov = val
        elif name == "gold_accuracy":
            acc = val
    return cov, acc, notes


def _print_grounding_debug(rec: dict) -> None:
    """Retrieval telemetry from ``sourcebot_ollama`` (any question domain)."""
    m = (rec.get("response") or {}).get("metrics") or {}
    extra = m.get("extra") or {}
    terms = extra.get("grounding_extracted_terms")
    search_q = extra.get("grounding_search_query")
    paths = extra.get("grounding_snippet_paths")
    if not (terms or search_q or paths):
        return
    print("    [retrieval]")
    if terms:
        print(f"      terms: {terms}")
    if search_q:
        print(f"      query: {search_q}")
    if paths:
        print(f"      snippets ({len(paths)}): " + ", ".join(paths[:8]))
        if len(paths) > 8:
            print(f"        … +{len(paths) - 8} more")


def _print_golden_summary(cases: list[Case], run_dir: Path, *, debug_grounding: bool = False) -> None:
    """Human-readable comparison: reference vs each adapter answer."""
    print("\n" + "=" * 72)
    print("GOLDEN COMPARISON (reference vs adapter outputs)")
    print("=" * 72)

    for case in cases:
        gold = _gold_text(case)
        print(f"\n## {case.id} [{case.job}]")
        print("-" * 72)
        print("REFERENCE (gold):")
        print(textwrap.indent(gold[:1200] + ("…" if len(gold) > 1200 else ""), "  "))
        print()

        case_dir = run_dir / "runs" / case.id
        if not case_dir.is_dir():
            print("  (no runs recorded)")
            continue

        rows: list[tuple[float, str, dict]] = []
        for run_file in sorted(case_dir.glob("*.json")):
            rec = json.loads(run_file.read_text())
            adapter = rec.get("adapter") or run_file.stem
            if not rec.get("ok"):
                rows.append((-1.0, adapter, rec))
                continue
            overall = (rec.get("score") or {}).get("overall", 0.0)
            rows.append((overall, adapter, rec))

        rows.sort(key=lambda t: -t[0])
        for _score, adapter, rec in rows:
            if not rec.get("ok"):
                print(f"  {adapter:18} FAILED  {rec.get('error', '')[:200]}")
                continue
            sc = rec.get("score") or {}
            cov, acc, notes = _judge_scores(sc)
            judge_bit = ""
            if cov is not None:
                judge_bit = f"  coverage={cov:.2f}  accuracy={acc:.2f}"
            if notes:
                judge_bit += f"  ({notes})"
            print(f"  {adapter:18} score={sc.get('overall', 0):.2f}{judge_bit}")
            if debug_grounding:
                _print_grounding_debug(rec)

            resp = rec.get("response") or {}
            if case.job == "ask":
                body = (resp.get("answer") or "").strip()
            else:
                decomp = resp.get("decomposition") or {}
                from eval.bakeoff.scorer import _decomposition_to_text

                body = _decomposition_to_text(decomp)
            preview = body[:500] + ("…" if len(body) > 500 else "")
            print(textwrap.indent(preview, "    "))
            print()


async def _run(args: argparse.Namespace) -> int:
    all_cases = discover_cases(EVAL_DIR)
    if args.minimal:
        case_id = MINIMAL_ASK_ID if args.job == "ask" else MINIMAL_DECOMPOSE_ID
        cases = filter_cases(all_cases, job=args.job, ids=[case_id])
    elif args.quick:
        ids = QUICK_CASE_IDS if (args.job or "ask") == "ask" else QUICK_DECOMPOSE_IDS
        cases = filter_cases(all_cases, job=args.job, ids=ids)
    else:
        cases = filter_cases(all_cases, job=args.job, ids=args.ids, tags=args.tags)
        if args.gold_only:
            cases = _cases_with_gold(cases)

    if not cases:
        print("no cases match filters", file=sys.stderr)
        return 1

    adapters = [a.strip() for a in args.adapters.split(",") if a.strip()]
    settings = get_settings()
    if not settings.gemini_api_key:
        print(
            "warning: GEMINI_API_KEY unset — judge scoring disabled; "
            "only rule-based rubric will run",
            file=sys.stderr,
        )

    client = build_client(args.base_url)
    out_dir = new_output_dir(OUTPUTS_DIR)
    print(f"Running {len(cases)} case(s) × {len(adapters)} adapter(s)", file=sys.stderr)
    print(f"output: {out_dir}", file=sys.stderr)

    def _progress(done, total, case_id, adapter):
        print(f"[{done:>3}/{total}] {adapter:16} {case_id}", file=sys.stderr)

    try:
        await run_bakeoff(
            client=client,
            cases=cases,
            adapters=adapters,
            output_dir=out_dir,
            timeout_seconds=args.timeout,
            on_progress=_progress,
            grounded=args.grounded,
            use_judge=args.use_judge and bool(settings.gemini_api_key),
            settings=settings,
            run_meta={
                "harness": "eval_golden",
                "gold_only": args.gold_only,
                "minimal": args.minimal,
                "quick": args.quick,
                "use_judge": args.use_judge,
            },
        )
    finally:
        await client.aclose()

    report_path, summary_path = render_report(out_dir)
    _print_golden_summary(cases, out_dir, debug_grounding=args.debug_grounding)

    summary = json.loads(summary_path.read_text())
    print("\nLEADERBOARD (avg_score vs golden reference):")
    for row in summary.get("leaderboard", []):
        print(
            f"  {row['adapter']:18}  score={row['avg_score']:.2f}  "
            f"success={row['successes']}/{row['runs']}  "
            f"avg_ms={row['avg_duration_ms']}"
        )

    print(f"\nFull report: {report_path}")
    print(f"Summary JSON: {summary_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adapters", default=DEFAULT_ADAPTERS,
                   help=f"comma-separated (default: {DEFAULT_ADAPTERS})")
    p.add_argument("--job", choices=["ask", "decompose"], default="ask")
    p.add_argument("--ids", nargs="+", default=None, help="case id allowlist")
    p.add_argument("--tags", nargs="+", default=None)
    p.add_argument("--minimal", action="store_true",
                   help=f"one case + one adapter (ask={MINIMAL_ASK_ID}, "
                   f"decompose={MINIMAL_DECOMPOSE_ID}; default adapter {MINIMAL_DEFAULT_ADAPTER})")
    p.add_argument("--quick", action="store_true",
                   help=f"two cases (heavier): ask={QUICK_CASE_IDS}, decompose={QUICK_DECOMPOSE_IDS}")
    p.add_argument("--gold-only", action="store_true", default=True,
                   help="only cases with gold_answer / gold_decomposition (default: true)")
    p.add_argument("--no-gold-only", action="store_false", dest="gold_only",
                   help="include cases without a golden reference")
    p.add_argument("--use-judge", action="store_true", default=True,
                   help="LLM judge vs golden reference (default: on when GEMINI_API_KEY set)")
    p.add_argument("--no-judge", action="store_false", dest="use_judge")
    p.add_argument("--grounded", action="store_true",
                   help="force grounded=true on ask cases")
    p.add_argument("--debug-grounding", action="store_true", default=True,
                   help="print search terms + snippet paths after each run (default: on)")
    p.add_argument("--no-debug-grounding", action="store_false", dest="debug_grounding")
    p.add_argument("--base-url", default=None, help="live API URL; omit for in-process ASGI")
    p.add_argument("--timeout", type=float, default=None,
                   help="per-case timeout seconds (default: 420 for --minimal, else 900)")
    args = p.parse_args(argv)
    if args.minimal:
        if args.adapters == DEFAULT_ADAPTERS:
            args.adapters = MINIMAL_DEFAULT_ADAPTER
        if args.timeout is None:
            args.timeout = MINIMAL_DEFAULT_TIMEOUT
    elif args.timeout is None:
        args.timeout = 900.0
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
