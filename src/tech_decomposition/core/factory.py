"""Factory helpers: build a Pipeline by name from settings.

Centralizes the "which strategies to wire for mode X" decision so the CLI
and API can request a pipeline by mode without knowing the internals.
"""
from __future__ import annotations

from typing import Literal

from ..config import Settings
from ..engines.decompose import DecomposeEngine
from ..engines.sourcebot import SourcebotEngine
from ..enrichers.cheap import CheapEnricher
from ..inputs.raw import RawQuestionSource
from ..inputs.text_file import TextFileSource
from ..renderers.markdown import MarkdownRenderer
from ..renderers.passthrough import PassthroughMarkdownRenderer
from ..sinks.cli import CLISink
from ..sinks.file import FileSink
from .metrics import JsonlMetricsObserver
from .pipeline import Pipeline
from .protocols import Engine, InputSource

AskEngineName = Literal["sourcebot"]
TicketSourceName = Literal["text_file", "raw"]


def build_ask_engine(name: AskEngineName, *, max_steps: int | None = None) -> Engine:
    if name == "sourcebot":
        return SourcebotEngine(max_steps=max_steps)
    raise ValueError(f"unknown ask engine: {name!r}")


def build_ticket_source(name: TicketSourceName) -> InputSource:
    if name == "text_file":
        return TextFileSource()
    if name == "raw":
        return RawQuestionSource()
    raise ValueError(f"unknown ticket source: {name!r}")


def build_ask_pipeline(
    settings: Settings,
    *,
    engine: AskEngineName = "sourcebot",
    max_steps: int | None = None,
    include_file_sink: bool = True,
    include_cli_sink: bool = True,
) -> Pipeline:
    """Q&A pipeline: raw question → engine → renderer → sinks."""
    active_engine = build_ask_engine(engine, max_steps=max_steps)

    sinks = []
    if include_cli_sink:
        sinks.append(CLISink())
    if include_file_sink:
        sinks.append(FileSink(subdir="answers"))
    return Pipeline(
        input_source=RawQuestionSource(),
        enricher=None,
        engine=active_engine,
        renderers=[MarkdownRenderer()],
        sinks=sinks,
    )


def build_decompose_pipeline(
    settings: Settings,
    *,
    source: TicketSourceName = "text_file",
    repos: list[str] | None = None,
    include_file_sink: bool = True,
    include_cli_sink: bool = True,
) -> Pipeline:
    """Decompose pipeline: ticket source → enrich → decompose engine → render → sinks.

    Decomposition uses Sourcebot (same path as ``ask``) for grounded exploration,
    then a local structured post-process into ``Decomposition``.
    """
    engine: Engine = DecomposeEngine(repos=repos)
    sinks = []
    if include_cli_sink:
        sinks.append(CLISink())
    if include_file_sink:
        sinks.append(FileSink(subdir="decompositions"))

    return Pipeline(
        input_source=build_ticket_source(source),
        enricher=CheapEnricher(),
        engine=engine,
        renderers=[PassthroughMarkdownRenderer()],
        sinks=sinks,
    )


def build_metrics_observer(settings: Settings, *, mode: str) -> JsonlMetricsObserver:
    return JsonlMetricsObserver(settings, mode=mode)
