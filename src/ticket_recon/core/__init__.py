"""ticket-recon core: stage Protocols, RunContext, Pipeline, model registry.

The pipeline runs as five composable stages:

    InputSource -> Enricher -> Engine -> Renderer -> Sink

Each stage is a Protocol; the strategy you swap in determines behavior
(local-agent vs Sourcebot engine, CLI vs Jira sink, etc.) without touching
the orchestrator. A ``RunContext`` carries config and metrics through stages.
"""
from .context import RunContext, StageResult
from .pipeline import Pipeline
from .protocols import (
    Engine,
    EngineResult,
    Enricher,
    EnrichedQuestion,
    InputSource,
    LoadedInput,
    MetricsObserver,
    Renderer,
    Rendered,
    Sink,
    SinkResult,
)

__all__ = [
    "Engine",
    "EngineResult",
    "Enricher",
    "EnrichedQuestion",
    "InputSource",
    "LoadedInput",
    "MetricsObserver",
    "Pipeline",
    "Rendered",
    "Renderer",
    "RunContext",
    "Sink",
    "SinkResult",
    "StageResult",
]
