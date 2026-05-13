"""Factory helpers: build a Pipeline by name from settings.

Centralizes the "which strategies to wire for mode X" decision so the CLI
and API can request a pipeline by mode without knowing the internals.
"""
from __future__ import annotations

from typing import Literal

from ..config import Settings
from ..engines._wrappers import StructuredEngine
from ..engines.decompose import DecomposeEngine
from ..engines.deep_decompose import DeepDecomposeEngine
from ..engines.local_agent import LocalAgentEngine
from ..engines.sourcebot import SourcebotEngine
from ..enrichers.cheap import CheapEnricher
from ..inputs.jira_ticket import JiraTicketSource
from ..inputs.raw import RawQuestionSource
from ..inputs.text_file import TextFileSource
from ..renderers.jira_adf import JiraADFRenderer
from ..renderers.markdown import MarkdownRenderer
from ..renderers.passthrough import PassthroughMarkdownRenderer
from ..sinks.cli import CLISink
from ..sinks.jira_comment import JiraCommentSink
from ..sinks.markdown_file import MarkdownFileSink
from .metrics import JsonlMetricsObserver
from .pipeline import Pipeline
from .protocols import Engine, InputSource
from .structurer import PydanticAIStructurer

AskEngineName = Literal["sourcebot", "local"]
DecomposeMode = Literal["cheap", "deep", "auto"]
TicketSourceName = Literal["jira", "text_file"]


def build_ask_engine(name: AskEngineName, *, max_steps: int | None = None) -> Engine:
    if name == "sourcebot":
        return SourcebotEngine(max_steps=max_steps)
    if name == "local":
        return LocalAgentEngine()
    raise ValueError(f"unknown ask engine: {name!r}")


def build_ticket_source(name: TicketSourceName) -> InputSource:
    if name == "jira":
        return JiraTicketSource()
    if name == "text_file":
        return TextFileSource()
    raise ValueError(f"unknown ticket source: {name!r}")


def build_ask_pipeline(
    settings: Settings,
    *,
    engine: AskEngineName = "sourcebot",
    max_steps: int | None = None,
    structure_responses: bool = True,
    include_file_sink: bool = True,
    include_cli_sink: bool = True,
) -> Pipeline:
    """Q&A pipeline: raw question → engine → (structurer) → markdown → sinks.

    ``structure_responses=True`` (default) wraps the engine in a
    ``StructuredEngine`` that runs a Flash-tier pydantic-ai pass to strip
    narration and lift citations into typed objects. The local agent already
    emits structured output, so the wrapper is skipped for it.
    """
    inner = build_ask_engine(engine, max_steps=max_steps)
    if structure_responses and engine == "sourcebot":
        structurer = PydanticAIStructurer(settings)
        active_engine: Engine = StructuredEngine(inner, structurer)
    else:
        active_engine = inner

    sinks = []
    if include_cli_sink:
        sinks.append(CLISink())
    if include_file_sink:
        sinks.append(MarkdownFileSink(subdir="answers"))
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
    source: TicketSourceName = "jira",
    mode: DecomposeMode = "cheap",
    repos: list[str] | None = None,
    post_to_jira: bool = False,
    include_file_sink: bool = True,
    include_cli_sink: bool = True,
) -> Pipeline:
    """Decompose pipeline: ticket source → enrich → decompose engine → render → sinks.

    `mode="auto"` is resolved here using the same heuristics the old pipeline
    used: low-confidence enrichment, migration-flavored language, or three+
    suspected repos escalates to deep. Decision is made at *engine* selection
    time, after enrichment has run — implemented via a small wrapper engine.
    """
    if mode == "deep":
        engine: Engine = DeepDecomposeEngine(repos=repos)
    elif mode == "auto":
        engine = AutoEscalatingDecomposeEngine(repos=repos)
    else:
        engine = DecomposeEngine(repos=repos)

    renderers = [PassthroughMarkdownRenderer()]
    sinks = []
    if include_cli_sink:
        sinks.append(CLISink())
    if include_file_sink:
        sinks.append(MarkdownFileSink(subdir="decompositions"))
    if post_to_jira:
        # JiraCommentSink consumes the jira_adf rendering; append the ADF
        # renderer at the end of the list so the sink can find it.
        renderers.append(JiraADFRenderer())
        sinks.append(JiraCommentSink())

    return Pipeline(
        input_source=build_ticket_source(source),
        enricher=CheapEnricher(),
        engine=engine,
        renderers=renderers,
        sinks=sinks,
    )


class AutoEscalatingDecomposeEngine:
    """Decides cheap vs deep using the enriched query's signal.

    Kept here (not in engines/) because the logic is tied to pipeline
    composition rather than to either engine on its own.
    """

    name = "decompose_auto"

    def __init__(self, *, repos: list[str] | None = None):
        self.repos = repos

    async def run(self, q, ctx):
        if _should_escalate(q):
            chosen = DeepDecomposeEngine(repos=self.repos)
        else:
            chosen = DecomposeEngine(repos=self.repos)
        result = await chosen.run(q, ctx)
        # Bubble up which engine actually ran so observability is honest.
        result.extra["auto_escalated"] = (chosen.name == "decompose_deep")
        result.engine = f"auto({chosen.name})"
        return result


_MIGRATION_KEYWORDS = ("migrate", "migration", "port to", "move to ", "refactor across")


def _should_escalate(q) -> bool:
    if q.confidence == "low":
        return True
    text = (q.extra.get("summary") or q.question or "").lower()
    if any(kw in text for kw in _MIGRATION_KEYWORDS):
        return True
    if len(q.suspected_repos) >= 3:
        return True
    return False


def build_metrics_observer(settings: Settings, *, mode: str) -> JsonlMetricsObserver:
    return JsonlMetricsObserver(settings, mode=mode)
