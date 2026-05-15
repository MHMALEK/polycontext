"""Tests for the shared CLI-agent base class.

Covers:
  * ``parse_jsonl_events`` — tolerant JSONL parser handles cline/opencode-
    shaped streams, mixed text + JSON, and pure-text fallback.
  * ``ticket_blob`` / ``subtask_prompt`` — the helpers moved off the
    three adapter files. Sanity checks on the assembly.
  * Preamble templates render with a ``model_tag``.

The base ``CliAgentAdapter`` itself is exercised end-to-end through the
real adapters (``_cline.py``, ``_opencode.py``, ``_cursor.py``); we don't
mock subprocess.run here because ``test_subprocess.py`` already covers
the runner contract.
"""
from __future__ import annotations

import json

import pytest

from tech_decomposition.adapters._cli_agent_base import (
    DECOMPOSE_PREAMBLE,
    ParsedAgentRun,
    parse_jsonl_events,
    subtask_prompt,
    ticket_blob,
)
from tech_decomposition.adapters.base import (
    AdapterDecomposeInput,
    AdapterImplementInput,
)
from tech_decomposition.models import Subtask


# ---------------------------------------------------------------------------
# parse_jsonl_events
# ---------------------------------------------------------------------------


def test_parse_jsonl_events_returns_none_for_plain_text() -> None:
    assert parse_jsonl_events("hello world, no JSON here") is None
    assert parse_jsonl_events("") is None


def test_parse_jsonl_events_picks_up_text_and_tokens() -> None:
    stream = "\n".join(
        json.dumps(o)
        for o in [
            {"type": "say", "text": "thinking out loud"},
            {"type": "tool_use", "tool": "Read"},
            {"type": "final", "text": "final answer", "input_tokens": 42, "output_tokens": 7},
        ]
    )
    parsed = parse_jsonl_events(stream)
    assert parsed is not None
    assert parsed.answer == "final answer"
    assert parsed.tokens_in == 42
    assert parsed.tokens_out == 7
    assert parsed.tool_calls == 1


def test_parse_jsonl_events_handles_nested_usage_block() -> None:
    # Mirrors the Anthropic-style nested usage object that opencode may emit.
    stream = json.dumps({
        "type": "result",
        "content": "done",
        "usage": {"input_tokens": 100, "output_tokens": 25, "total_cost": 0.0042},
        "model": "claude-opus-4-7",
    })
    parsed = parse_jsonl_events(stream)
    assert parsed is not None
    assert parsed.answer == "done"
    assert parsed.tokens_in == 100
    assert parsed.tokens_out == 25
    assert parsed.cost_usd == pytest.approx(0.0042)
    assert parsed.model == "claude-opus-4-7"


def test_parse_jsonl_events_falls_back_to_joined_text() -> None:
    # No event flagged as final → join all text chunks in order.
    stream = "\n".join(
        json.dumps(o)
        for o in [
            {"type": "say", "text": "part one"},
            {"type": "say", "text": "part two"},
        ]
    )
    parsed = parse_jsonl_events(stream)
    assert parsed is not None
    assert "part one" in parsed.answer
    assert "part two" in parsed.answer


def test_parse_jsonl_events_keeps_largest_token_value() -> None:
    # Several events restate cumulative usage — keep the max so we don't
    # under-report.
    stream = "\n".join(
        json.dumps(o)
        for o in [
            {"type": "say", "text": "x", "input_tokens": 10},
            {"type": "say", "text": "y", "input_tokens": 30},
            {"type": "final", "text": "answer", "input_tokens": 25},
        ]
    )
    parsed = parse_jsonl_events(stream)
    assert parsed is not None
    assert parsed.tokens_in == 30


def test_parse_jsonl_events_ignores_garbage_lines() -> None:
    stream = (
        "starting up...\n"
        + json.dumps({"type": "final", "text": "answer", "input_tokens": 1})
        + "\nshutdown OK"
    )
    parsed = parse_jsonl_events(stream)
    assert parsed is not None
    assert parsed.answer == "answer"


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------


def test_ticket_blob_assembles_all_fields() -> None:
    inp = AdapterDecomposeInput(
        ticket_key="ABC-1",
        ticket_url="https://x/ABC-1",
        ticket_text="Add a thing",
        repos=["repo-a", "repo-b"],
    )
    blob = ticket_blob(inp)
    assert "Ticket key: ABC-1" in blob
    assert "Ticket URL: https://x/ABC-1" in blob
    assert "Add a thing" in blob
    assert "Repos to consider: repo-a, repo-b" in blob


def test_ticket_blob_skips_missing_fields() -> None:
    inp = AdapterDecomposeInput(ticket_text="only text")
    blob = ticket_blob(inp)
    assert blob.strip() == "only text"


def test_subtask_prompt_with_subtask() -> None:
    st = Subtask(
        title="Build widget",
        description="Build the widget per spec.",
        repo="repo-a",
        files=["a.py"],
        file_links=[],
        acceptance_criteria=["thing works", "tests pass"],
        estimated_complexity="small",
    )
    inp = AdapterImplementInput(repo="repo-a", subtask=st)
    out = subtask_prompt(inp)
    assert "# Build widget" in out
    assert "Build the widget per spec." in out
    assert "- thing works" in out
    assert "- tests pass" in out


def test_subtask_prompt_falls_back_to_free_text() -> None:
    inp = AdapterImplementInput(repo="repo-a", free_text="do the thing")
    assert subtask_prompt(inp) == "do the thing"


def test_decompose_preamble_includes_model_tag() -> None:
    rendered = DECOMPOSE_PREAMBLE.format(model_tag="cline")
    assert '"decomposition_model": "cline"' in rendered


# ---------------------------------------------------------------------------
# ParsedAgentRun ergonomics
# ---------------------------------------------------------------------------


def test_parsed_agent_run_defaults() -> None:
    p = ParsedAgentRun(answer="x")
    assert p.answer == "x"
    assert p.model is None
    assert p.tokens_in is None
    assert p.tokens_out is None
    assert p.cost_usd is None
    assert p.tool_calls == 0
    assert p.extra == {}
