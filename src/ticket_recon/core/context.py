"""RunContext: carries config, run id, metrics observer through pipeline stages."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..config import Settings
    from .protocols import MetricsObserver


@dataclass
class StageResult:
    """Lightweight record of a single stage execution; used by Pipeline hooks."""

    stage: str
    strategy: str
    seconds: float
    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunContext:
    """Per-invocation state. Threaded through every stage.

    Carries settings (read-only config), a unique run_id (for correlation in
    metrics / logs), the active mode label (e.g. "ask", "decompose", "compare"),
    an optional MetricsObserver, and a free-form scratchpad for stages that
    need to stash data (rarely — most stage I/O should go through the typed
    payloads).
    """

    settings: "Settings"
    mode: str = "ask"
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metrics: "MetricsObserver | None" = None
    scratch: dict[str, Any] = field(default_factory=dict)

    def record_stage(self, result: StageResult) -> None:
        """Forward a StageResult to the metrics observer, if one is attached."""
        if self.metrics is None:
            return
        self.metrics.on_stage_complete(
            run_id=self.run_id,
            stage=result.stage,
            strategy=result.strategy,
            seconds=result.seconds,
            model=result.model,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cost_usd=result.cost_usd,
            extra=result.extra or None,
        )
