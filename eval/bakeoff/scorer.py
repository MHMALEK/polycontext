"""Scoring rubrics.

Two layers:

1. **Rule-based** (``score_response`` / ``_score_ask`` / ``_score_decompose``).
   Cheap, deterministic substring + structure checks. Used by default and
   always when no ``gold_answer`` is available.

2. **LLM-as-judge** (``score_response_async`` with ``use_judge=True``).
   When a case carries a ``gold_answer`` in ``expected``, a single Gemini
   Flash call grades the adapter's answer along two dimensions —
   coverage (does it carry the facts that the gold has?) and accuracy
   (does it contradict the gold?) — and merges those as extra ``Check``
   entries alongside the rule-based ones. Substring matching can't tell
   "good answer, slightly different wording" from "wrong answer"; the
   judge can.

Each scorer returns a ``Score`` with:
  * a normalized 0..1 ``overall`` for sorting in the report
  * a list of named ``checks`` (each pass/fail with a message) so the
    report can show *why* one adapter scored higher than another
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

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
    """Dispatch on job type — rule-based only (sync, no LLM)."""
    if case.job == "ask":
        return _score_ask(case, response)
    if case.job == "decompose":
        return _score_decompose(case, response)
    return Score(notes=f"no scorer for job={case.job}")


async def score_response_async(
    *,
    case: Case,
    response: dict[str, Any],
    settings=None,
    use_judge: bool = False,
) -> Score:
    """Dispatch + optional LLM-judge layer.

    When ``use_judge=True`` AND ``settings`` is provided AND the case has a
    ``gold_answer`` in ``expected``, an LLM judge call grades the adapter
    answer against the gold and merges two extra checks (gold_coverage,
    gold_accuracy) into the rule-based score before final aggregation.

    Falls back to plain rule-based scoring on any judge failure.

    Decompose cases may use ``gold_decomposition`` (reference breakdown text).
    """
    base = score_response(case=case, response=response)
    if not use_judge or settings is None:
        return base

    expected = case.expected or {}
    if case.job == "ask":
        gold = expected.get("gold_answer")
        if not gold or not isinstance(gold, str):
            return base
        candidate = (response.get("answer") or "").strip()
        if not candidate:
            return base
        question = case.input.get("query", "")
    elif case.job == "decompose":
        gold = expected.get("gold_decomposition") or expected.get("gold_answer")
        if not gold or not isinstance(gold, str):
            return base
        candidate = _decomposition_to_text(response.get("decomposition") or {})
        if not candidate.strip():
            return base
        question = case.input.get("query") or case.input.get("ticket_text") or ""
    else:
        return base

    try:
        verdict = await _judge_against_gold(
            question=question,
            gold=gold.strip(),
            candidate=candidate,
            settings=settings,
        )
    except Exception:
        return base
    if verdict is None:
        return base
    # When the judge runs, its two dimensions should DOMINATE the total —
    # rule-based mentions/min_chars are brittle (exact substring match) and
    # often penalize semantically correct answers that paraphrase. Heavy
    # weight on the judge means: if coverage/accuracy are high, the score
    # is high regardless of whether the answer used the exact tokens the
    # ``must_mention`` list expected. Rule-based checks remain in the list
    # for diagnostic value (visible in the report) but don't dominate.
    checks = list(base.checks)
    checks.append(Check(
        name="gold_coverage",
        ok=verdict.coverage >= 0.7,
        detail=f"{verdict.coverage:.2f}",
        weight=10.0,
    ))
    checks.append(Check(
        name="gold_accuracy",
        ok=verdict.accuracy >= 0.8,
        detail=f"{verdict.accuracy:.2f}",
        weight=10.0,
    ))
    out = _aggregate(checks)
    out.notes = f"judge: {verdict.notes}"
    return out


# ---------------------------------------------------------------------------
# LLM-as-judge
# ---------------------------------------------------------------------------


class _JudgeVerdict(BaseModel):
    """Output of the LLM judge — two scalar dimensions + a short note."""

    coverage: float = Field(
        ge=0.0, le=1.0,
        description=(
            "Fraction of distinct facts in the reference answer that appear "
            "in the candidate answer (in any phrasing). 1.0 = candidate "
            "covers everything the reference covers. Extra accurate facts "
            "in the candidate do NOT lower this score."
        ),
    )
    accuracy: float = Field(
        ge=0.0, le=1.0,
        description=(
            "1.0 if the candidate does not contradict any fact in the "
            "reference. Reduce by ~0.2 per contradiction. Plausible extra "
            "info that is not in the reference is NOT a contradiction."
        ),
    )
    notes: str = Field(
        default="",
        description="One short sentence (under 30 words) explaining the verdict.",
    )


_JUDGE_SYSTEM = """\
You are evaluating code-Q&A answers. You will see:
  - QUESTION: the user's question
  - REFERENCE: a known-correct condensed answer
  - CANDIDATE: an answer to grade

Score two dimensions on 0.0-1.0:

coverage — Fraction of distinct FACTS in the REFERENCE that appear in
the CANDIDATE (any phrasing, any structure). If REFERENCE covers 5 facts
and CANDIDATE covers 4 of them, coverage = 0.8. Extra accurate facts in
the candidate do NOT lower coverage.

accuracy — 1.0 if CANDIDATE contradicts no fact in REFERENCE. Reduce by
~0.2 per contradiction. Plausible additional info not in REFERENCE is
NOT a contradiction — only mark down for statements that conflict with
the reference.

Return ONE short sentence in notes explaining your scores.
"""


_JUDGE_AGENT_CACHE: dict[str, Any] = {}


def _build_judge_agent(settings) -> Any | None:
    if not getattr(settings, "gemini_api_key", ""):
        return None
    try:
        from pydantic_ai import Agent
        from pydantic_ai.models.gemini import GeminiModel
        from pydantic_ai.settings import ModelSettings
    except ImportError:
        return None
    os.environ.setdefault("GEMINI_API_KEY", settings.gemini_api_key)
    model_name = (getattr(settings, "enrich_model", None) or "gemini-2.5-flash").strip()
    cached = _JUDGE_AGENT_CACHE.get(model_name)
    if cached is not None:
        return cached
    try:
        agent = Agent(
            model=GeminiModel(model_name),
            output_type=_JudgeVerdict,
            system_prompt=_JUDGE_SYSTEM,
            model_settings=ModelSettings(temperature=0.0),
        )
        _JUDGE_AGENT_CACHE[model_name] = agent
        return agent
    except Exception:
        return None


def _decomposition_to_text(decomp: dict[str, Any]) -> str:
    """Flatten a decomposition object for LLM-judge comparison."""
    parts: list[str] = []
    if decomp.get("overview"):
        parts.append(f"Overview:\n{decomp['overview'].strip()}")
    repos = decomp.get("affected_repos") or []
    if repos:
        parts.append("Affected repos: " + ", ".join(str(r) for r in repos))
    for risk in decomp.get("risks") or []:
        parts.append(f"Risk: {risk}")
    for q in decomp.get("open_questions") or []:
        parts.append(f"Open question: {q}")
    for st in decomp.get("subtasks") or []:
        title = (st.get("title") or "").strip()
        desc = (st.get("description") or "").strip()
        repo = st.get("repo") or ""
        files = ", ".join(st.get("files") or [])
        line = f"- {title}"
        if repo:
            line += f" [{repo}]"
        if files:
            line += f" (files: {files})"
        if desc:
            line += f": {desc}"
        parts.append(line)
    return "\n".join(parts)


async def _judge_against_gold(
    *, question: str, gold: str, candidate: str, settings,
) -> _JudgeVerdict | None:
    agent = _build_judge_agent(settings)
    if agent is None:
        return None
    prompt = (
        f"QUESTION:\n{question.strip()}\n\n"
        f"REFERENCE:\n{gold.strip()}\n\n"
        f"CANDIDATE:\n{candidate.strip()}\n"
    )
    try:
        result = await agent.run(prompt)
        return result.output
    except Exception:
        return None


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
