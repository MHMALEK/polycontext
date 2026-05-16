"""Adapter contract: a uniform surface over heterogeneous code agents.

Every registered adapter (e.g. Cursor SDK, Cline SDK, Sourcebot) implements some subset
of two methods:

    ask        — code Q&A
    decompose  — ticket → structured subtasks

Capabilities are declared at the class level so the registry and router can
report what's supported without instantiating every backend. Methods that an
adapter does not implement raise ``NotSupported`` (the API surface translates
this to a 501).

Adapters are async so HTTP-backed and subprocess-backed implementations don't
block the event loop. In-process Python backends still implement the async
interface — most will just ``return`` synchronously from an ``async def``.
"""
from __future__ import annotations

from abc import ABC
from typing import ClassVar, Literal

from pydantic import BaseModel, Field

from ..models import Decomposition, Snippet


Capability = Literal["ask", "decompose"]


class NotSupported(Exception):
    """Raised when an adapter is asked to do something it doesn't implement."""

    def __init__(self, adapter: str, capability: Capability):
        self.adapter = adapter
        self.capability = capability
        super().__init__(f"adapter {adapter!r} does not support {capability!r}")


# ---------------------------------------------------------------------------
# Shared metrics
# ---------------------------------------------------------------------------


class AdapterMetrics(BaseModel):
    """Uniform per-call telemetry. Every adapter populates what it can.

    Fields are intentionally optional — not every adapter tracks token counts
    or cost (e.g. local models, subprocess CLIs without structured output).
    The UI's bake-off view degrades gracefully on missing values.
    """

    duration_ms: int = 0
    tokens_in: int | None = None
    tokens_out: int | None = None
    cost_usd: float | None = None
    model: str | None = None
    tool_calls: int = 0
    extra: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# ask
# ---------------------------------------------------------------------------


class AdapterAskInput(BaseModel):
    query: str = Field(min_length=1)
    thread_id: str | None = Field(
        default=None,
        description="First message's run id — continue this chat. Omit to start a new thread.",
    )
    repos: list[str] | None = None
    top_k: int = Field(default=8, ge=1, le=50)
    branch: str | None = None
    starting_ref: str | None = None


class AdapterAskResult(BaseModel):
    adapter: str
    answer: str
    citations: list[Snippet] = Field(default_factory=list)
    metrics: AdapterMetrics = Field(default_factory=AdapterMetrics)


# ---------------------------------------------------------------------------
# decompose
# ---------------------------------------------------------------------------


class AdapterDecomposeInput(BaseModel):
    query: str | None = None
    repos: list[str] | None = None
    # Ignored by the in-repo pipeline (Sourcebot + structure). Remote adapters may interpret it.
    mode: Literal["cheap", "deep", "auto"] = "auto"


class AdapterDecomposeResult(BaseModel):
    adapter: str
    decomposition: Decomposition
    markdown: str = ""
    metrics: AdapterMetrics = Field(default_factory=AdapterMetrics)


# ---------------------------------------------------------------------------
# Adapter base class
# ---------------------------------------------------------------------------


class Adapter(ABC):
    """Base class for all adapters.

    Subclasses set the ``name`` and ``capabilities`` ClassVars and override
    only the methods they implement. Methods left at the base raise
    ``NotSupported`` so the router can return a clean 501.

    Example::

        class MyAdapter(Adapter):
            name = "mine"
            capabilities = {"ask"}

            async def ask(self, inp):
                ...

    The constructor takes the ``Settings`` object so adapters can read API
    keys, model defaults, repo paths, etc. without globals.
    """

    name: ClassVar[str] = "base"
    capabilities: ClassVar[set[Capability]] = set()
    description: ClassVar[str] = ""

    def __init__(self, settings):
        self.settings = settings

    # ----- the two operations ----------------------------------------------

    async def ask(self, inp: AdapterAskInput) -> AdapterAskResult:
        raise NotSupported(self.name, "ask")

    async def decompose(self, inp: AdapterDecomposeInput) -> AdapterDecomposeResult:
        raise NotSupported(self.name, "decompose")

    # ----- introspection ----------------------------------------------------

    def health(self) -> dict:
        """Best-effort readiness check. Override for backends that need to
        verify a binary, env var, or service.

        Returns a dict with at least ``{"ok": bool}`` and an optional
        ``"reason"`` when not ok.
        """
        return {"ok": True}

    def supports(self, cap: Capability) -> bool:
        return cap in self.capabilities
