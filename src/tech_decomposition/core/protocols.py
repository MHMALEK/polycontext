"""Stage protocols for the tech-decomposition pipeline.

Five stages are defined here. Each is a runtime-checkable Protocol so any class
matching the shape can be plugged in without inheritance. Concrete data classes
(``LoadedInput``, ``EnrichedQuestion``, ``EngineResult``, ``Rendered``,
``SinkResult``) carry the payload between stages — they're plain Pydantic
models so they serialize cleanly across CLI / API / JSONL boundaries.
"""
from __future__ import annotations

from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from .context import RunContext


# ---------------------------------------------------------------------------
# Data classes flowing between stages
# ---------------------------------------------------------------------------


class LoadedInput(BaseModel):
    """Output of InputSource.load(): a normalized question with metadata.

    `kind` identifies the input type so downstream stages can branch
    (e.g. JiraTicketSource → kind="jira_ticket", supplies title/body/labels).
    """

    kind: Literal["question", "jira_ticket", "text_file", "raw"]
    title: str = ""
    body: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class EnrichedQuestion(BaseModel):
    """Output of Enricher.enrich(): a normalized question primed for the engine."""

    question: str
    intent: str | None = None
    code_keywords: list[str] = Field(default_factory=list)
    search_queries: list[str] = Field(default_factory=list)
    suspected_repos: list[str] = Field(default_factory=list)
    confidence: Literal["low", "medium", "high"] | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class EngineResult(BaseModel):
    """Output of Engine.run(): the engine's answer with structured payload.

    `payload` carries the engine-specific structured output (Answer,
    Decomposition, etc.) so renderers can pick fields without re-parsing
    markdown. `answer_markdown` is the user-facing prose.
    """

    engine: str
    answer_markdown: str
    payload: dict[str, Any] = Field(default_factory=dict)
    citations: list[dict[str, Any]] = Field(default_factory=list)
    model: str | None = None
    transport: str | None = None
    wall_seconds: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class Rendered(BaseModel):
    """Output of Renderer.render(): the final shape to deliver."""

    format: Literal["markdown", "html", "text", "json", "jira_adf"]
    body: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class SinkResult(BaseModel):
    """Output of Sink.deliver(): where the rendered output went."""

    sink: str
    location: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Stage protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class InputSource(Protocol):
    name: str

    async def load(self, ref: str | dict[str, Any], ctx: RunContext) -> LoadedInput: ...


@runtime_checkable
class Enricher(Protocol):
    name: str

    async def enrich(self, loaded: LoadedInput, ctx: RunContext) -> EnrichedQuestion: ...


@runtime_checkable
class Engine(Protocol):
    name: str

    async def run(self, q: EnrichedQuestion, ctx: RunContext) -> EngineResult: ...


@runtime_checkable
class Renderer(Protocol):
    name: str

    def render(self, result: EngineResult, ctx: RunContext) -> Rendered: ...


@runtime_checkable
class Sink(Protocol):
    name: str

    async def deliver(self, rendered: Rendered, ctx: RunContext) -> SinkResult: ...


# ---------------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------------


@runtime_checkable
class MetricsObserver(Protocol):
    """Receives per-stage metrics from the Pipeline.

    Implementations can write JSONL, push to OpenTelemetry, log, etc. The
    pipeline calls ``on_stage_complete`` after every stage and ``on_run_complete``
    once the whole pipeline finishes.
    """

    def on_stage_start(self, *, run_id: str, stage: str, strategy: str) -> None: ...

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
    ) -> None: ...

    def on_run_complete(
        self,
        *,
        run_id: str,
        total_seconds: float,
        total_cost_usd: float | None,
        extra: dict[str, Any] | None = None,
    ) -> None: ...
