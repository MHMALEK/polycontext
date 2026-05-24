"""Context recall scoring — does the retriever surface labeled answer files?"""
from __future__ import annotations

from eval.bakeoff.cases import Case
from eval.bakeoff.scorer import _context_recall_check, score_response


def _ask_case(expected: dict) -> Case:
    return Case(
        id="t",
        job="ask",
        tags=[],
        input={"query": "irrelevant for this test"},
        expected=expected,
    )


def _response_with_paths(paths: list[str], answer: str = "an answer") -> dict:
    return {
        "answer": answer,
        "citations": [],
        "metrics": {"extra": {"grounding_paths": paths}},
    }


def test_context_recall_full_match_passes():
    case = _ask_case({"expected_files": ["src/users/roles.py", "src/auth/jwt.py"]})
    resp = _response_with_paths([
        "backend/src/users/roles.py",
        "backend/src/auth/jwt.py",
        "backend/src/unrelated.py",
    ])
    check = _context_recall_check(case, resp)
    assert check is not None
    assert check.ok is True
    assert "2/2" in check.detail


def test_context_recall_partial_match_records_missing():
    case = _ask_case({"expected_files": ["src/users/roles.py", "src/billing/invoices.py"]})
    resp = _response_with_paths(["backend/src/users/roles.py"])
    check = _context_recall_check(case, resp)
    assert check is not None
    assert check.ok is False
    assert "1/2" in check.detail
    assert "src/billing/invoices.py" in check.detail


def test_context_recall_returns_none_when_case_has_no_labels():
    case = _ask_case({"min_chars": 80})
    resp = _response_with_paths(["whatever.py"])
    assert _context_recall_check(case, resp) is None


def test_context_recall_flags_missing_grounding_paths_when_labels_exist():
    # If a case is labeled but the adapter didn't surface paths, that's a
    # *failure* — we want a visible signal that the metric couldn't run, not
    # silent omission.
    case = _ask_case({"expected_files": ["src/x.py"]})
    resp = {"answer": "...", "metrics": {"extra": {}}}
    check = _context_recall_check(case, resp)
    assert check is not None
    assert check.ok is False
    assert "no grounding_paths" in check.detail


def test_context_recall_normalizes_backslashes_and_leading_slash():
    case = _ask_case({"expected_files": ["/src/users/roles.py"]})
    resp = _response_with_paths(["backend\\src\\users\\roles.py"])
    check = _context_recall_check(case, resp)
    assert check is not None
    assert check.ok is True


def test_context_recall_threshold_override():
    case = _ask_case({
        "expected_files": ["a.py", "b.py", "c.py", "d.py"],
        "min_context_recall": 0.5,
    })
    resp = _response_with_paths(["a.py", "b.py"])
    check = _context_recall_check(case, resp)
    assert check is not None
    # 0.5 recall meets the lowered bar.
    assert check.ok is True


def test_score_ask_aggregates_context_recall_into_overall():
    case = _ask_case({
        "expected_files": ["src/users/roles.py"],
        "min_chars": 5,
    })
    resp = _response_with_paths(["backend/src/users/roles.py"], answer="ok answer")
    score = score_response(case=case, response=resp)
    names = {c.name for c in score.checks}
    assert "context_recall" in names
    assert score.overall > 0
