"""Scoring rubrics.

Three families, one per job. All scorers are **rule-based** for now — no
LLM-as-judge in this cut — so the harness has zero external dependencies and
is deterministic.

Each scorer returns a ``Score`` with:
  * a normalized 0..1 ``overall`` for sorting in the report
  * a list of named ``checks`` (each pass/fail with a message) so the
    report can show *why* one adapter scored higher than another

Add an LLM-judge layer later by writing a second scorer that consumes the
same response objects and merges into ``checks``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .cases import Case


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""
    weight: float = 1.0


@dataclass
class Score:
    overall: float = 0.0
    checks: list[Check] = field(default_factory=list)
    notes: str = ""


def score_response(*, case: Case, response: dict[str, Any]) -> Score:
    """Dispatch on job type. ``response`` is the adapter's ``result`` payload."""
    if case.job == "ask":
        return _score_ask(case, response)
    if case.job == "decompose":
        return _score_decompose(case, response)
    return Score(notes=f"no scorer for job={case.job}")


# ---------------------------------------------------------------------------
# ask
# ---------------------------------------------------------------------------


def _score_ask(case: Case, resp: dict[str, Any]) -> Score:
    """Rubric for Q&A answers.

    Expected fields in ``case.expected``:
      * ``must_mention``      — list[str] substrings that should appear
      * ``should_mention``    — list[str] substrings that probably appear
      * ``min_chars``         — int, default 80 (catches truncated answers)
      * ``min_citations``     — int, default 0
    """
    expected = case.expected or {}
    answer = (resp.get("answer") or "").lower()
    citations = resp.get("citations") or []
    checks: list[Check] = []

    checks.append(Check(
        name="non_empty",
        ok=bool(answer.strip()),
        detail=f"{len(answer)} chars",
        weight=1.0,
    ))
    checks.append(Check(
        name="min_chars",
        ok=len(answer) >= int(expected.get("min_chars", 80)),
        detail=f"{len(answer)} >= {expected.get('min_chars', 80)}",
        weight=0.5,
    ))
    min_cit = int(expected.get("min_citations", 0))
    if min_cit > 0:
        checks.append(Check(
            name="has_citations",
            ok=len(citations) >= min_cit,
            detail=f"{len(citations)} citations",
            weight=1.0,
        ))

    for needle in expected.get("must_mention", []) or []:
        checks.append(Check(
            name=f"mentions[{needle}]",
            ok=needle.lower() in answer,
            detail=needle,
            weight=2.0,
        ))
    for needle in expected.get("should_mention", []) or []:
        checks.append(Check(
            name=f"prefers[{needle}]",
            ok=needle.lower() in answer,
            detail=needle,
            weight=0.5,
        ))

    return _aggregate(checks)


# ---------------------------------------------------------------------------
# decompose
# ---------------------------------------------------------------------------


def _score_decompose(case: Case, resp: dict[str, Any]) -> Score:
    """Rubric for tech decompositions.

    Expected fields in ``case.expected``:
      * ``min_subtasks``         — int, default 2
      * ``max_subtasks``         — int, default 12 (longer = noise)
      * ``must_touch_repos``     — list[str], affected_repos should include all
      * ``must_mention_files``   — list[str], any subtask's files should match
    """
    expected = case.expected or {}
    decomp = resp.get("decomposition") or {}
    subtasks = decomp.get("subtasks") or []
    affected = set(decomp.get("affected_repos") or [])
    files = {f for st in subtasks for f in (st.get("files") or [])}
    checks: list[Check] = []

    checks.append(Check(
        name="has_decomposition",
        ok=bool(decomp),
        detail="Decomposition object present",
        weight=2.0,
    ))
    checks.append(Check(
        name="has_overview",
        ok=bool((decomp.get("overview") or "").strip()),
        weight=1.0,
    ))
    min_st = int(expected.get("min_subtasks", 2))
    max_st = int(expected.get("max_subtasks", 12))
    checks.append(Check(
        name="subtask_count_in_range",
        ok=min_st <= len(subtasks) <= max_st,
        detail=f"{len(subtasks)} (expected {min_st}..{max_st})",
        weight=1.5,
    ))
    for st in subtasks:
        if not (st.get("title") and st.get("description")):
            checks.append(Check(
                name="subtask_has_title_and_description",
                ok=False,
                detail=f"missing on subtask: {st.get('title') or '<no title>'}",
                weight=1.0,
            ))
            break
    else:
        if subtasks:
            checks.append(Check(name="subtask_has_title_and_description", ok=True, weight=1.0))

    for repo in expected.get("must_touch_repos", []) or []:
        checks.append(Check(
            name=f"affects_repo[{repo}]",
            ok=repo in affected,
            detail=repo,
            weight=1.5,
        ))
    for substr in expected.get("must_mention_files", []) or []:
        checks.append(Check(
            name=f"touches_file[{substr}]",
            ok=any(substr in f for f in files),
            detail=substr,
            weight=1.5,
        ))

    return _aggregate(checks)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _aggregate(checks: list[Check]) -> Score:
    """Weighted pass-rate, clamped to [0, 1]."""
    if not checks:
        return Score(overall=0.0, checks=checks, notes="no checks")
    total_w = sum(c.weight for c in checks)
    passed_w = sum(c.weight for c in checks if c.ok)
    overall = passed_w / total_w if total_w > 0 else 0.0
    return Score(overall=round(overall, 3), checks=checks)
