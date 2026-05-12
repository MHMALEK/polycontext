from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .config import Settings
from .models import RetrievedContext


# Per-million-token prices in USD (≤200k context tier for Gemini 2.5 Pro).
# Update if Gemini pricing changes. Keys are model name fragments — matched by
# `in` so "gemini-2.5-flash" matches both "gemini-2.5-flash" and "gemini/gemini-2.5-flash".
PRICING_PER_M_USD: dict[str, dict[str, float]] = {
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
}


def _price_for(model: str) -> dict[str, float] | None:
    for key, prices in PRICING_PER_M_USD.items():
        if key in model:
            return prices
    return None


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    p = _price_for(model)
    if not p:
        return None
    return round(
        (input_tokens / 1_000_000) * p["input"]
        + (output_tokens / 1_000_000) * p["output"],
        6,
    )


class StageMetrics(BaseModel):
    seconds: float
    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


class RetrievalMetrics(BaseModel):
    seconds: float
    total_snippets: int
    total_chars: int
    by_source: dict[str, int] = Field(default_factory=dict)
    by_repo: dict[str, int] = Field(default_factory=dict)
    by_repo_source: dict[str, dict[str, int]] = Field(default_factory=dict)


class RunMetrics(BaseModel):
    run_id: str
    started_at: datetime
    ticket_key: str | None
    ticket_title: str
    mode: str = "cheap"
    escalated_to_deep: bool = False

    total_seconds: float
    enrich: StageMetrics
    retrieval: RetrievalMetrics
    decompose: StageMetrics

    deep_iterations: int | None = None
    deep_tool_calls: dict[str, int] = Field(default_factory=dict)

    total_cost_usd: float | None
    posted_to_jira: bool = False
    markdown_path: str | None = None
    subtask_count: int = 0
    affected_repos: list[str] = Field(default_factory=list)


def retrieval_metrics(ctx: RetrievedContext, seconds: float) -> RetrievalMetrics:
    by_source: Counter = Counter()
    by_repo: Counter = Counter()
    by_repo_source: dict[str, Counter] = {}
    for rc in ctx.repos:
        by_repo[rc.repo] = len(rc.snippets)
        by_repo_source.setdefault(rc.repo, Counter())
        for s in rc.snippets:
            by_source[s.source] += 1
            by_repo_source[rc.repo][s.source] += 1
    return RetrievalMetrics(
        seconds=round(seconds, 3),
        total_snippets=ctx.total_snippets,
        total_chars=ctx.total_chars,
        by_source=dict(by_source),
        by_repo=dict(by_repo),
        by_repo_source={r: dict(c) for r, c in by_repo_source.items()},
    )


def usage_from_result(result: Any) -> tuple[int | None, int | None]:
    """Pull (input_tokens, output_tokens) out of a pydantic_ai run result.

    Pydantic AI's result.usage() returns a Usage with request_tokens /
    response_tokens. Handle both new and old field names defensively.
    """
    try:
        u = result.usage()
    except Exception:
        return (None, None)
    if u is None:
        return (None, None)
    in_tok = getattr(u, "request_tokens", None) or getattr(u, "input_tokens", None)
    out_tok = getattr(u, "response_tokens", None) or getattr(u, "output_tokens", None)
    return (in_tok, out_tok)


def write_metrics(row: RunMetrics, settings: Settings) -> Path:
    out_dir = settings.output_dir / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "runs.jsonl"
    with path.open("a") as f:
        f.write(row.model_dump_json() + "\n")
    return path


def now_utc() -> datetime:
    return datetime.now(timezone.utc)
