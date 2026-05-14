"""Case discovery + filtering."""
from __future__ import annotations

from pathlib import Path

from eval.bakeoff.cases import (
    discover_cases,
    filter_cases,
    load_ask_cases,
    load_yaml_cases,
)


REPO = Path(__file__).resolve().parents[2]


def test_discover_finds_seed_cases():
    cases = discover_cases(REPO / "eval")
    ids = {c.id for c in cases}
    # Ask cases come from questions.toml
    assert "q1-data-sharing-geolocation" in ids
    # Decompose seed cases
    assert "d1-supplier-prefill" in ids
    assert "d2-farm-name-validation" in ids
    # Implement seed cases
    assert "i1-trivial-docstring" in ids


def test_ask_cases_have_query_input():
    cases = load_ask_cases(REPO / "eval" / "questions.toml")
    assert cases, "questions.toml has at least one case"
    for c in cases:
        assert c.job == "ask"
        assert c.input.get("query"), f"case {c.id} has no query"


def test_yaml_cases_carry_expected():
    cases = load_yaml_cases(REPO / "eval" / "cases" / "implement", "implement")
    assert cases
    i1 = next(c for c in cases if c.id == "i1-trivial-docstring")
    assert i1.expected.get("must_touch_paths") == [".py"]


def test_filter_by_job():
    cases = discover_cases(REPO / "eval")
    asks = filter_cases(cases, job="ask")
    assert all(c.job == "ask" for c in asks)
    assert len(asks) >= 6


def test_filter_by_id_intersection():
    cases = discover_cases(REPO / "eval")
    filtered = filter_cases(cases, ids=["d1-supplier-prefill", "nonexistent"])
    assert [c.id for c in filtered] == ["d1-supplier-prefill"]


def test_filter_by_tag_intersection():
    cases = discover_cases(REPO / "eval")
    filtered = filter_cases(cases, tags=["validation"])
    assert filtered
    assert all("validation" in c.tags for c in filtered)
