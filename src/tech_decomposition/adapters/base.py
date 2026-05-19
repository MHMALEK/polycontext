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
    grounded: bool = Field(
        default=False,
        description=(
            "Opt-in: prepend a grounding block (Sourcebot search snippets for ``query``) "
            "to the user prompt before the adapter sees it. The API endpoint handles this "
            "wrapping; adapters themselves are oblivious."
        ),
    )


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


class RawDecomposeText(BaseModel):
    """What an adapter's ``_decompose_raw_text`` returns.

    The adapter is only responsible for running its LLM and giving back the
    text it produced. Schema validation lives outside the adapter (in
    ``core/decomposition_structurer.py``), so individual adapters don't need
    to repeat the extract_json + model_validate dance and can't 502 from a
    malformed model response.
    """

    text: str
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

    async def _decompose_raw_text(self, inp: AdapterDecomposeInput) -> RawDecomposeText:
        """Subclasses implement this. Drive the underlying LLM and return its
        raw text response plus telemetry. No JSON parsing, no schema validation,
        no Decomposition construction — those happen in ``decompose`` below."""
        raise NotSupported(self.name, "decompose")

    async def decompose(self, inp: AdapterDecomposeInput) -> AdapterDecomposeResult:
        """Final orchestration: drive the adapter's LLM, then route the raw
        text through the shared structurer to get a validated ``Decomposition``.

        Adapters override ``_decompose_raw_text``, not this method. That keeps
        schema enforcement in one place (``core/decomposition_structurer``)
        and out of every individual adapter.
        """
        # Lazy import to avoid a circular import (core/ imports adapters/base
        # indirectly through extract_json).
        from ..core.decomposition_structurer import structure_decomposition

        raw = await self._decompose_raw_text(inp)
        structured = await structure_decomposition(
            text=raw.text,
            query=(inp.query or ""),
            settings=self.settings,
            model_tag=self.name,
        )
        # Attach structurer telemetry + the raw upstream text (truncated) to
        # the adapter's metrics so callers can see when LLM-repair fired and
        # what the upstream actually emitted. The Decomposition itself is
        # rendered as the ``markdown`` field below — the upstream raw is for
        # debugging, not for display.
        extra = dict(raw.metrics.extra or {})
        extra["structurer_used_llm_repair"] = structured.used_llm_repair
        if structured.repair_model:
            extra["structurer_model"] = structured.repair_model
        if structured.notes:
            extra["structurer_notes"] = structured.notes
        if raw.text:
            # Keep it bounded so the runstore row doesn't bloat on a chatty
            # adapter — 4 kB is plenty for triage.
            extra["upstream_raw_text"] = raw.text[:4000]
            extra["upstream_raw_text_len"] = len(raw.text)
        return AdapterDecomposeResult(
            adapter=self.name,
            decomposition=structured.decomposition,
            markdown=structured.decomposition.to_markdown(),
            metrics=AdapterMetrics(
                duration_ms=raw.metrics.duration_ms,
                tokens_in=raw.metrics.tokens_in,
                tokens_out=raw.metrics.tokens_out,
                cost_usd=raw.metrics.cost_usd,
                model=raw.metrics.model,
                tool_calls=raw.metrics.tool_calls,
                extra=extra,
            ),
        )

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
