"""Tests for tiered pipeline routing and coverage."""
from __future__ import annotations

from tech_decomposition.core.coverage import assess_coverage
from tech_decomposition.core.grounding import GroundedContext, GroundingMetrics, GroundingSnippet
from tech_decomposition.core.router import (
    TaskTier,
    answer_signals_insufficient,
    classify_task,
    should_agent_fallback,
    should_escalate_synthesis,
)


def test_classify_enumeration():
    route = classify_task(
        query="what are the current user roles we have?",
        job="ask",
        tags=["enumeration"],
    )
    assert route.tier == TaskTier.ENUMERATION
    assert route.prefer_single_shot is True


def test_classify_trace():
    route = classify_task(
        query="what happens when a user uploads master data?",
        job="ask",
        tags=["behavior"],
    )
    assert route.tier == TaskTier.TRACE
    assert route.prefer_single_shot is False


def test_coverage_sufficient_for_enumeration():
    ctx = GroundedContext(
        snippets=[
            GroundingSnippet(repo="traceability", path="src/constants.py", content="DATA_VIEWER\n" * 50),
            GroundingSnippet(repo="traceability", path="src/user_model.py", content="UserRoles\n" * 50),
        ],
        grounding_block="## snippets",
        metrics=GroundingMetrics(snippet_count=2, total_chars=1000),
    )
    route = classify_task(query="list all roles", job="ask", tags=["enumeration"])
    cov = assess_coverage(ctx, route, min_snippets=3, min_chars=800)
    assert cov.sufficient is True


def test_validation_and_behavior_prefetch_top_k():
    v = classify_task(
        query="what regex validates farm name?",
        job="ask",
        tags=["validation"],
    )
    b = classify_task(
        query="what happens when user imports geolocation only?",
        job="ask",
        tags=["behavior"],
    )
    assert v.prefetch_top_k == 18
    assert b.prefetch_top_k == 18


def test_should_escalate_synthesis():
    assert should_escalate_synthesis(tags=["validation"], coverage_sufficient=True)
    assert should_escalate_synthesis(tags=["master-data"], coverage_sufficient=False)
    assert should_escalate_synthesis(tags=["cross-repo"], coverage_sufficient=False)
    assert not should_escalate_synthesis(tags=["cross-repo"], coverage_sufficient=True)


def test_should_agent_fallback_requires_insufficient_synthesis():
    route = classify_task(
        query="what happens when user uploads?",
        job="ask",
        tags=["behavior"],
    )
    assert not should_agent_fallback(
        route=route,
        tags=["behavior"],
        coverage_sufficient=False,
        synthesis_text="The upload flow calls MasterDataService then persists rows.",
    )
    assert should_agent_fallback(
        route=route,
        tags=["behavior"],
        coverage_sufficient=False,
        synthesis_text="The snippets do not contain enough information to answer.",
    )
    assert not should_agent_fallback(
        route=route,
        tags=["validation"],
        coverage_sufficient=False,
        synthesis_text="not in snippets",
    )


def test_answer_signals_insufficient():
    assert answer_signals_insufficient("")
    assert answer_signals_insufficient("Cannot determine from the provided snippets.")
    assert not answer_signals_insufficient("Farm names use NODE_NAME_PATTERN.")
