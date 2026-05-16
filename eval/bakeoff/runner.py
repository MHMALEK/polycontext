"""Bake-off runner.

Fans cases × adapters and writes raw + scored results to disk. Sequential per
adapter pair on purpose — agent CLIs can hammer the same Anthropic /
Gemini key, so we don't want N parallel runs blowing through rate limits.

Outputs land under ``eval/outputs/eval-<timestamp>/`` so old runs are
preserved for comparison. The directory layout is committed in stone for
``report.py`` to consume:

    eval-<ts>/
    ├── manifest.json
    ├── runs/<case-id>/<adapter>.json
    └── (after `report` step) report.md, summary.json
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .cases import Case, Job
from .client import AdapterClient
from .scorer import Score, score_response

log = logging.getLogger(__name__)


@dataclass
class RunRecord:
    """One adapter's response to one case, plus scoring + timing."""

    case_id: str
    job: Job
    adapter: str
    ok: bool
    status: int
    error: str | None
    duration_ms: int
    run_id: str | None = None
    response: dict[str, Any] = field(default_factory=dict)
    score: dict[str, Any] = field(default_factory=dict)


def _case_input_for_adapter(case: Case) -> dict[str, Any]:
    """Per-job mapping from the case's ``input`` shape to the adapter's
    request body shape.

    The case files use the *minimal* fields a human would write; the
    adapters' Pydantic models accept a few extras. This is where we fill
    in the rest of the request payload.
    """
    if case.job == "ask":
        return {
            "query": case.input["query"],
            "repos": case.input.get("repos"),
            "top_k": case.input.get("top_k", 8),
            "branch": case.input.get("branch"),
            "starting_ref": case.input.get("starting_ref"),
        }
    if case.job == "decompose":
        return {
            "ticket_key": case.input.get("ticket_key"),
            "ticket_url": case.input.get("ticket_url"),
            "ticket_text": case.input.get("ticket_text"),
            "repos": case.input.get("repos"),
            "mode": case.input.get("mode", "auto"),
        }
    if case.job == "implement":
        return {
            "repo": case.input["repo"],
            "free_text": case.input.get("free_text"),
            "subtask": case.input.get("subtask"),
            "base_branch": case.input.get("base_branch", "main"),
            "ticket_key": case.input.get("ticket_key"),
            "draft": case.input.get("draft", True),
        }
    raise ValueError(f"unsupported job: {case.job!r}")


async def run_case(
    client: AdapterClient,
    case: Case,
    adapter: str,
    *,
    timeout_seconds: float = 900.0,
) -> RunRecord:
    """One adapter, one case. Failures are captured, not raised."""
    body = _case_input_for_adapter(case)
    t = time.monotonic()
    try:
        envelope = await asyncio.wait_for(
            client.call(adapter, case.job, body),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError:
        return RunRecord(
            case_id=case.id, job=case.job, adapter=adapter,
            ok=False, status=504, error=f"runner timeout after {timeout_seconds}s",
            duration_ms=int((time.monotonic() - t) * 1000),
            run_id=None,
        )
    except Exception as e:  # noqa: BLE001 — capture everything for the report
        return RunRecord(
            case_id=case.id, job=case.job, adapter=adapter,
            ok=False, status=500, error=f"{type(e).__name__}: {e}",
            duration_ms=int((time.monotonic() - t) * 1000),
            run_id=None,
        )

    duration_ms = int((time.monotonic() - t) * 1000)

    if not envelope.get("ok"):
        return RunRecord(
            case_id=case.id, job=case.job, adapter=adapter,
            ok=False, status=envelope.get("status", 500),
            error=envelope.get("error", "unknown"),
            duration_ms=duration_ms,
            run_id=envelope.get("run_id"),
        )

    result = envelope.get("result") or {}
    score: Score = score_response(case=case, response=result)
    return RunRecord(
        case_id=case.id, job=case.job, adapter=adapter,
        ok=True, status=envelope.get("status", 200), error=None,
        duration_ms=duration_ms,
        run_id=envelope.get("run_id"),
        response=result,
        score=asdict(score),
    )


async def run_bakeoff(
    *,
    client: AdapterClient,
    cases: list[Case],
    adapters: list[str],
    output_dir: Path,
    timeout_seconds: float = 900.0,
    on_progress=None,
    run_meta: dict[str, Any] | None = None,
) -> Path:
    """Execute the full grid and write everything to disk.

    Returns the output directory path so callers can chain into ``report``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "adapters": adapters,
        "cases": [{"id": c.id, "job": c.job, "tags": c.tags} for c in cases],
        "timeout_seconds": timeout_seconds,
        "run_meta": run_meta or {},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    runs_dir = output_dir / "runs"
    runs_dir.mkdir(exist_ok=True)

    total = len(cases) * len(adapters)
    done = 0
    for case in cases:
        case_dir = runs_dir / case.id
        case_dir.mkdir(exist_ok=True)
        for adapter in adapters:
            done += 1
            if on_progress:
                on_progress(done, total, case.id, adapter)
            log.info("[%d/%d] case=%s adapter=%s", done, total, case.id, adapter)
            rec = await run_case(client, case, adapter, timeout_seconds=timeout_seconds)
            (case_dir / f"{adapter}.json").write_text(json.dumps(asdict(rec), indent=2, default=str))

    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return output_dir


def new_output_dir(base: Path) -> Path:
    """``eval/outputs/eval-<utc-timestamp>/`` — never collides between runs."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    p = base / f"eval-{ts}"
    p.mkdir(parents=True, exist_ok=True)
    return p
