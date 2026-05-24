"""Report helpers: quality value extraction + retrieval-vs-synthesis diagnosis."""
from __future__ import annotations

from eval.bakeoff.report import _diagnose, _quality_values_from_checks


def test_extract_values_parses_leading_float_from_detail():
    checks = [
        {"name": "non_empty", "ok": True, "detail": "120 chars"},
        {"name": "context_recall", "ok": True, "detail": "0.83 (5/6 expected paths in retrieval)"},
        {"name": "gold_coverage", "ok": True, "detail": "0.91"},
        {"name": "gold_accuracy", "ok": False, "detail": "0.60"},
    ]
    out = _quality_values_from_checks(checks)
    assert out == {"context_recall": 0.83, "gold_coverage": 0.91, "gold_accuracy": 0.60}


def test_extract_skips_unparseable_details():
    checks = [
        {"name": "context_recall", "ok": False, "detail": "adapter surfaced no grounding_paths"},
        {"name": "gold_coverage", "ok": True, "detail": "0.75"},
    ]
    out = _quality_values_from_checks(checks)
    assert "context_recall" not in out
    assert out["gold_coverage"] == 0.75


def test_diagnose_good():
    assert _diagnose({"context_recall": 0.9, "gold_coverage": 0.85, "gold_accuracy": 1.0}) == "✓"


def test_diagnose_retrieval_miss():
    assert _diagnose({"context_recall": 0.3, "gold_coverage": 0.4}) == "RS"
    assert _diagnose({"context_recall": 0.3, "gold_coverage": 0.85}) == "R"


def test_diagnose_synthesis_miss():
    assert _diagnose({"context_recall": 0.9, "gold_coverage": 0.4}) == "S"


def test_diagnose_partial_data():
    # If only recall is labeled, treat the unobserved dimension as ok (don't
    # invent a synthesis failure). Same the other way.
    assert _diagnose({"context_recall": 0.9}) == "✓"
    assert _diagnose({"context_recall": 0.3}) == "R"
    assert _diagnose({"gold_coverage": 0.4}) == "S"


def test_diagnose_no_data():
    assert _diagnose({}) == "?"
