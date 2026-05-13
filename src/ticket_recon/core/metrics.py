"""JSONL-backed implementation of MetricsObserver.

Writes one row per stage to ``{output_dir}/metrics/runs.jsonl`` plus a
summary row per run. The existing ``analyze.py`` reads the same path.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import Settings


class JsonlMetricsObserver:
    """Append per-stage rows and a per-run summary row to runs.jsonl."""

    def __init__(self, settings: Settings, *, mode: str = "ask"):
        self.settings = settings
        self.mode = mode
        self._run_costs: dict[str, list[float]] = {}
        self._stages: dict[str, list[dict[str, Any]]] = {}
        self._path = settings.output_dir / "metrics" / "runs.jsonl"
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def _write(self, row: dict[str, Any]) -> None:
        with self._path.open("a") as f:
            f.write(json.dumps(row, default=str) + "\n")

    def on_stage_complete(
        self,
        *,
        run_id: str,
        stage: str,
        strategy: str,
        seconds: float,
        model: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        cost_usd: float | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        row = {
            "type": "stage",
            "ts": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id,
            "mode": self.mode,
            "stage": stage,
            "strategy": strategy,
            "seconds": seconds,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost_usd,
        }
        if extra:
            row["extra"] = extra
        self._write(row)
        self._stages.setdefault(run_id, []).append(row)
        if cost_usd is not None:
            self._run_costs.setdefault(run_id, []).append(cost_usd)

    def on_run_complete(
        self,
        *,
        run_id: str,
        total_seconds: float,
        total_cost_usd: float | None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        if total_cost_usd is None and run_id in self._run_costs:
            total_cost_usd = round(sum(self._run_costs[run_id]), 6)
        row = {
            "type": "run",
            "ts": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id,
            "mode": self.mode,
            "total_seconds": total_seconds,
            "total_cost_usd": total_cost_usd,
            "stages": [s["stage"] for s in self._stages.get(run_id, [])],
        }
        if extra:
            row["extra"] = extra
        self._write(row)
