"""RunContext: per-invocation state threaded through the API handlers."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..config import Settings


@dataclass
class RunContext:
    """Per-invocation state.

    Carries settings (read-only config), a unique run_id (for correlation in
    metrics / logs), the active mode label (e.g. "ask", "decompose"), and a
    free-form scratchpad for handlers that need to stash data.

    ``input_preview`` is set by the API handler before persisting the run row
    so the analytics view can show *what* each run asked.
    """

    settings: "Settings"
    mode: str = "ask"
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    input_preview: str | None = None
    scratch: dict[str, Any] = field(default_factory=dict)
