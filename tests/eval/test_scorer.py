"""Scorer rubrics — verify each job's checks produce sensible scores."""
from __future__ import annotations

from eval.bakeoff.cases import Case
from eval.bakeoff.scorer import score_response


def _ask_case(**expected):
    return Case(id="t", job="ask", input={"query": "x"}, expected=expected)


def _decompose_case(**expected):
    return Case(id="t", job="decompose", input={"ticket_text": "x"}, expected=expected)


def _implement_case(**expected):
    return Case(id="t", job="implement", input={"repo": "x"}, expected=expected)


# ----- ask -----------------------------------------------------------------


def test_ask_empty_answer_scores_low():
    s = score_response(case=_ask_case(), response={"answer": ""})
    assert s.overall < 0.5
    assert any(c.name == "non_empty" and not c.ok for c in s.checks)


def test_ask_required_substring_rewards_match():
    resp = {"answer": "The regex /[A-Z]+/ rejects this; see validator.py"}
    s_match = score_response(case=_ask_case(must_mention=["regex"]), response=resp)
    s_miss = score_response(case=_ask_case(must_mention=["banana"]), response=resp)
    assert s_match.overall > s_miss.overall


def test_ask_citation_threshold():
    resp = {"answer": "long enough answer ............................................." * 4,
            "citations": [{"repo": "r", "path": "f.py", "line_start": 1, "line_end": 1, "content": ""}]}
    case = _ask_case(min_citations=1)
    s = score_response(case=case, response=resp)
    assert any(c.name == "has_citations" and c.ok for c in s.checks)


# ----- decompose -----------------------------------------------------------


def test_decompose_empty_scores_zero():
    s = score_response(case=_decompose_case(), response={})
    assert s.overall == 0.0 or any(not c.ok for c in s.checks)


def test_decompose_full_response_passes_baseline_checks():
    resp = {
        "decomposition": {
            "ticket_title": "x",
            "overview": "do the thing",
            "affected_repos": ["frontend"],
            "subtasks": [
                {"title": "a", "description": "d", "repo": "frontend",
                 "files": ["src/components/Farm.tsx"]},
                {"title": "b", "description": "d", "repo": "frontend",
                 "files": ["src/utils/validate.ts"]},
            ],
        },
    }
    s = score_response(
        case=_decompose_case(min_subtasks=2, must_touch_repos=["frontend"],
                              must_mention_files=["validate"]),
        response=resp,
    )
    assert s.overall >= 0.9, f"expected high score, got {s.overall}: {s.checks}"


def test_decompose_subtask_count_out_of_range():
    resp = {"decomposition": {
        "ticket_title": "x", "overview": "o", "affected_repos": [],
        "subtasks": [],
    }}
    s = score_response(case=_decompose_case(min_subtasks=2), response=resp)
    assert any(c.name == "subtask_count_in_range" and not c.ok for c in s.checks)


# ----- implement -----------------------------------------------------------


def test_implement_no_files_changed_fails():
    resp = {"branch": "tech-decomp/abc", "files_changed": [], "mr_url": None}
    s = score_response(case=_implement_case(), response=resp)
    assert any(c.name == "changed_files_min" and not c.ok for c in s.checks)


def test_implement_must_touch_path():
    resp = {"branch": "tech-decomp/abc",
            "files_changed": ["src/utils/validate.py", "README.md"],
            "mr_url": "https://gitlab.com/foo/-/merge_requests/1"}
    s_pass = score_response(
        case=_implement_case(must_touch_paths=["validate"]), response=resp,
    )
    s_fail = score_response(
        case=_implement_case(must_touch_paths=["nonexistent"]), response=resp,
    )
    assert s_pass.overall > s_fail.overall


def test_implement_must_not_touch_path():
    resp = {"branch": "b", "files_changed": ["src/x.py", "node_modules/foo.js"],
            "mr_url": None}
    s = score_response(case=_implement_case(must_not_touch_paths=["node_modules"]),
                       response=resp)
    assert any(c.name.startswith("avoids[") and not c.ok for c in s.checks)


def test_implement_require_mr_when_set():
    resp = {"branch": "b", "files_changed": ["x.py"], "mr_url": None}
    s = score_response(case=_implement_case(require_mr=True), response=resp)
    assert any(c.name == "opened_mr" and not c.ok for c in s.checks)
