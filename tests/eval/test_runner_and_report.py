"""End-to-end harness test using a fake AdapterClient.

The runner doesn't care whether responses come from a real adapter or a
mock — it only needs the documented envelope shape. We exploit that to
exercise the full case → run → report flow without needing any of the
real adapters installed.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.bakeoff.cases import Case
from eval.bakeoff.report import render_report
from eval.bakeoff.runner import run_bakeoff


class FakeClient:
    """Deterministic stand-in for ``AdapterClient``.

    Each (adapter, job) returns a canned response. Lets us assert that the
    runner correctly threads inputs through, writes the right files, and
    that the reporter produces a sensible leaderboard.
    """

    def __init__(self):
        self.calls: list[tuple[str, str, dict]] = []

    async def list_adapters(self):
        return []

    async def call(self, adapter: str, job: str, body: dict) -> dict:
        self.calls.append((adapter, job, body))
        if adapter == "boom":
            return {"ok": False, "status": 502, "error": "intentional"}
        if job == "ask":
            return {
                "ok": True, "status": 200, "run_id": "r1",
                "result": {
                    "adapter": adapter,
                    "answer": "the regex /[A-Z]+/ rejects this — see validator.py for details and more bytes for length",
                    "citations": [{"repo": "x", "path": "validator.py",
                                    "line_start": 1, "line_end": 5, "content": ""}],
                    "metrics": {"duration_ms": 100, "cost_usd": 0.001, "model": adapter + "-mock"},
                },
            }
        if job == "decompose":
            return {
                "ok": True, "status": 200, "run_id": "r2",
                "result": {
                    "adapter": adapter,
                    "decomposition": {
                        "ticket_title": "t", "overview": "o", "affected_repos": ["frontend"],
                        "subtasks": [
                            {"title": "a", "description": "d", "repo": "frontend",
                             "files": ["src/validate.ts"]},
                            {"title": "b", "description": "d", "repo": "frontend",
                             "files": ["src/Form.tsx"]},
                        ],
                    },
                    "markdown": "# decomp\n",
                    "metrics": {"duration_ms": 200, "cost_usd": 0.01, "model": adapter + "-mock"},
                },
            }
        return {"ok": True, "status": 200, "result": {"adapter": adapter}}

    async def aclose(self) -> None:
        pass


@pytest.fixture
def fake_cases():
    return [
        Case(id="ask-1", job="ask", input={"query": "anything"},
             expected={"must_mention": ["regex"]}),
        Case(id="decompose-1", job="decompose", input={"ticket_text": "x"},
             expected={"min_subtasks": 2, "must_touch_repos": ["frontend"]}),
    ]


async def test_runner_writes_one_file_per_pair(fake_cases, tmp_path: Path):
    client = FakeClient()
    out = await run_bakeoff(
        client=client,
        cases=fake_cases,
        adapters=["a", "b"],
        output_dir=tmp_path / "run",
    )
    runs = out / "runs"
    files = sorted(p.relative_to(out).as_posix() for p in runs.rglob("*.json"))
    assert files == [
        "runs/ask-1/a.json", "runs/ask-1/b.json",
        "runs/decompose-1/a.json", "runs/decompose-1/b.json",
    ]
    rec = json.loads((runs / "ask-1" / "a.json").read_text())
    assert rec["ok"] is True
    assert rec["score"]["overall"] > 0


async def test_runner_captures_adapter_failure(fake_cases, tmp_path: Path):
    client = FakeClient()
    out = await run_bakeoff(
        client=client,
        cases=fake_cases,
        adapters=["boom"],
        output_dir=tmp_path / "run",
    )
    rec = json.loads((out / "runs" / "ask-1" / "boom.json").read_text())
    assert rec["ok"] is False
    assert rec["status"] == 502
    assert "intentional" in (rec["error"] or "")


async def test_report_renders_markdown_and_summary(fake_cases, tmp_path: Path):
    client = FakeClient()
    out = await run_bakeoff(
        client=client,
        cases=fake_cases,
        adapters=["a", "b", "boom"],
        output_dir=tmp_path / "run",
    )
    report_path, summary_path = render_report(out)

    md = report_path.read_text()
    assert "# Adapter bake-off report" in md
    assert "| `a` |" in md and "| `b` |" in md
    assert "❌" in md  # boom failure is rendered

    summary = json.loads(summary_path.read_text())
    leaderboard = {row["adapter"]: row for row in summary["leaderboard"]}
    assert leaderboard["a"]["successes"] == 2
    assert leaderboard["boom"]["successes"] == 0
    # Sort: successful adapters come before failed ones
    assert summary["leaderboard"][0]["adapter"] in {"a", "b"}
