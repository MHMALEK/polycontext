"""Adapter contract: a uniform surface over heterogeneous code agents.

Every registered adapter (e.g. Cursor SDK, Cline SDK, Sourcebot) implements some subset
of three methods:

    ask        — code Q&A
    decompose  — ticket → structured subtasks
    implement  — task → GitLab MR

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
from pathlib import Path
from typing import ClassVar, Literal

from pydantic import BaseModel, Field

from ..models import Decomposition, Snippet, Subtask


Capability = Literal["ask", "decompose", "implement"]


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
    ticket_key: str | None = None
    ticket_url: str | None = None
    ticket_text: str | None = None
    repos: list[str] | None = None
    # Each adapter interprets this as it wishes; adapters that don't
    # distinguish modes simply ignore the field.
    mode: Literal["cheap", "deep", "auto"] = "auto"


class AdapterDecomposeResult(BaseModel):
    adapter: str
    decomposition: Decomposition
    markdown: str = ""
    metrics: AdapterMetrics = Field(default_factory=AdapterMetrics)


# ---------------------------------------------------------------------------
# implement
# ---------------------------------------------------------------------------


class AdapterImplementInput(BaseModel):
    """A single piece of work the adapter should turn into a branch + MR.

    Exactly one of (subtask, free_text) must be set. ``repo`` names a key in
    ``settings.gitlab_projects``; the wrapper resolves it to a worktree path
    before calling the adapter.
    """

    repo: str = Field(min_length=1)
    subtask: Subtask | None = None
    free_text: str | None = Field(
        default=None,
        description="Plain-English task description, used when no Subtask is available.",
    )
    base_branch: str = "main"
    ticket_key: str | None = None
    draft: bool = True


class AdapterImplementResult(BaseModel):
    adapter: str
    mr_url: str | None = None
    branch: str
    commits: list[str] = Field(default_factory=list)
    diff_summary: str = ""
    files_changed: list[str] = Field(default_factory=list)
    metrics: AdapterMetrics = Field(default_factory=AdapterMetrics)


class ImplementContext(BaseModel):
    """Resolved state passed from the wrapper to the adapter.

    The wrapper provisions a worktree at ``worktree_path`` (a fresh branch off
    ``base_branch``) before invoking the adapter. The adapter's job is purely
    to edit files inside that directory; commit, push, and MR creation are
    handled by the shared wrapper (``core.implement.run_implement``).
    """

    worktree_path: Path
    branch: str
    repo: str
    base_branch: str

    model_config = {"arbitrary_types_allowed": True}


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

    # ----- the three operations --------------------------------------------

    async def ask(self, inp: AdapterAskInput) -> AdapterAskResult:
        raise NotSupported(self.name, "ask")

    async def decompose(self, inp: AdapterDecomposeInput) -> AdapterDecomposeResult:
        raise NotSupported(self.name, "decompose")

    async def implement(
        self,
        inp: AdapterImplementInput,
        ctx: ImplementContext,
    ) -> AdapterImplementResult:
        """Edit files inside ``ctx.worktree_path`` to satisfy ``inp``.

        The adapter MUST NOT commit, push, or open an MR — the wrapper does
        all that. The adapter SHOULD only touch files under
        ``ctx.worktree_path``. The returned ``files_changed`` list is
        advisory; the wrapper checks git status to compute the real diff.
        """
        raise NotSupported(self.name, "implement")

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
