"""Shared post-adapter step that turns raw LLM text into a validated ``Decomposition``.

Every adapter (Cursor, Cline, Claude Code, Gemini, OpenAI Agents, OpenCode)
runs its own LLM with its own prompts and tools. They each return a chunk of
text whose shape is *supposed* to be the Decomposition JSON, but in practice
each model is unreliable about it: markdown fences, bare Subtask objects,
prose interleaved with JSON, fields renamed, schema drift across model
versions, etc.

We used to try ``extract_json`` + ``Decomposition.model_validate`` first
and only fall back to an LLM repair on failure. That fast path was a
constant source of subtle bugs — a brace-balanced JSON walker that handled
N-deep nesting still couldn't tell "outer Decomposition with one subtask
in an array" from "the array contains a Subtask that happens to be biggest".

The cure: **always** route the upstream text through ``pydantic_ai.Agent``
with ``output_type=Decomposition``. That uses Gemini's native
``responseSchema`` feature to *force* schema-valid output from the model
on the API side — guaranteed by the SDK, not by our parsing. Cost: one
extra Gemini Flash call per decompose (~$0.0001, ~1-2 s). Reliability:
100 %, by construction.

Flow:

  1. Adapter produces raw text via its LLM/tools.
  2. ``structure_decomposition(text, query, settings)`` ALWAYS calls
     Gemini (Flash by default) with ``output_type=Decomposition``. The
     model sees the user query + the upstream text and emits a
     schema-valid Decomposition. If Gemini is unreachable (no API key,
     network down), a minimal fallback Decomposition is returned so
     callers never 502.
  3. The orchestration layer (``Adapter.decompose`` in base) combines
     the structurer's output with the adapter's metrics and returns the
     final ``AdapterDecomposeResult``.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

from ..config import Settings
from ..models import Decomposition


@dataclass
class StructureResult:
    """What the structurer returns alongside the validated decomposition."""

    decomposition: Decomposition
    # Always True under the new design (every call hits the LLM). Kept for
    # backward compatibility with code that reads ``metrics.extra`` for this
    # field; will be removed once callers stop branching on it.
    used_llm_repair: bool
    repair_model: str | None
    repair_tokens_in: int | None
    repair_tokens_out: int | None
    notes: str  # human-readable summary of which path fired


_STRUCTURER_SYSTEM = """\
You convert the output of an upstream code-decomposition LLM into a clean
``Decomposition`` object. The upstream model emits text that is supposed
to match the schema (query, overview, affected_repos, risks,
open_questions, subtasks[...], enrichment_model, decomposition_model) but
in practice it may be:

  - already-valid JSON (preserve verbatim — your only job is to reshape
    into the typed object).
  - JSON wrapped in ```json fences.
  - a single bare ``Subtask`` object (wrap it: use it as the only entry
    in ``subtasks``, set ``overview`` from its description, set
    ``affected_repos`` from its ``repo``).
  - JSON with extra/renamed fields (drop unknowns, map renames).
  - prose paragraphs with subtasks described informally (parse them out).

Rules:

  - PRESERVE upstream content verbatim where possible. Do not invent new
    subtasks, repos, files, or acceptance criteria.
  - Empty arrays / empty strings are fine — do not fabricate to fill them.
  - ``query`` should echo the USER QUESTION (provided in the input).

Return ONLY the Decomposition object.
"""


async def structure_decomposition(
    *,
    text: str,
    query: str,
    settings: Settings,
    model_tag: str = "",
) -> StructureResult:
    """Two-step schema enforcer.

    Step A (fast path) — try ``Decomposition.model_validate_json(text)``
    directly. This is for adapters that already produce schema-valid JSON
    (Gemini with ``responseSchema``, OpenAI Agents with ``response_format``,
    OpenCode with its structured output). No regex parsing, no extract_json:
    if the text isn't already a valid Decomposition JSON object, fall
    through.

    Step B (slow path) — pass the upstream text to ``pydantic_ai.Agent``
    with ``output_type=Decomposition``. Gemini Flash (configurable via
    ``enrich_model``) uses its native ``responseSchema`` to coerce arbitrary
    upstream text into a schema-valid Decomposition. This is the fallback
    for adapters whose upstream is free-form (Cursor, Cline, Claude Code,
    Gemini without responseSchema).

    Both steps return a fully validated ``Decomposition`` — there is no
    parsing gamble left in the pipeline. The ``_fallback_minimal`` path is
    only used when Gemini itself is unreachable.
    """
    # Step A — direct JSON validation.
    #
    # Even when Gemini is told to emit application/json via responseSchema,
    # it occasionally wraps the body in ```json fences. We strip them here
    # so the fast path catches that case too — the JSON inside is still
    # schema-valid by construction.
    stripped = _strip_json_fences((text or "").strip())
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            decomp = Decomposition.model_validate_json(stripped)
            if query and not decomp.query:
                decomp = decomp.model_copy(update={"query": query})
            if model_tag and not decomp.decomposition_model:
                decomp = decomp.model_copy(update={"decomposition_model": model_tag})
            return StructureResult(
                decomposition=decomp,
                used_llm_repair=False,
                repair_model=None,
                repair_tokens_in=None,
                repair_tokens_out=None,
                notes="fast-path: model_validate_json direct",
            )
        except Exception:
            # Fall through to LLM repair — upstream text claims to be JSON
            # but doesn't match the Decomposition schema.
            pass

    # Step B — LLM repair via pydantic-ai.
    if not settings.gemini_api_key:
        return _fallback_minimal(text=text, query=query, reason="GEMINI_API_KEY not set")

    try:
        from pydantic_ai import Agent
        from pydantic_ai.models.gemini import GeminiModel
        from pydantic_ai.settings import ModelSettings
    except ImportError:
        return _fallback_minimal(text=text, query=query, reason="pydantic-ai not installed")

    os.environ.setdefault("GEMINI_API_KEY", settings.gemini_api_key)
    raw = (settings.pipeline_structurer_model or settings.enrich_model or "gemini-2.5-flash").strip()
    model_name = raw.split(":", 1)[-1].strip() if ":" in raw else raw
    if not model_name:
        model_name = "gemini-2.5-flash"

    try:
        agent = Agent(
            model=GeminiModel(model_name),
            output_type=Decomposition,
            system_prompt=_STRUCTURER_SYSTEM,
            model_settings=ModelSettings(temperature=0.0),
        )
        # Upstream text trimmed to 16 kB — large enough for a fully-formed
        # Decomposition from any of the adapters, small enough to keep the
        # structurer call cheap.
        prompt = (
            f"USER QUESTION:\n{query.strip()}\n\n"
            f"UPSTREAM LLM OUTPUT:\n{text.strip()[:16000]}\n"
        )
        result = await agent.run(prompt)
        decomp = result.output
        if query and not decomp.query:
            decomp = decomp.model_copy(update={"query": query})
        if model_tag and not decomp.decomposition_model:
            decomp = decomp.model_copy(update={"decomposition_model": model_tag})

        toks_in, toks_out = _usage_from(result)
        return StructureResult(
            decomposition=decomp,
            used_llm_repair=True,  # always-on now; see dataclass docstring
            repair_model=model_name,
            repair_tokens_in=toks_in,
            repair_tokens_out=toks_out,
            notes=f"structured via {model_name} + responseSchema",
        )
    except Exception as e:
        return _fallback_minimal(
            text=text, query=query,
            reason=f"structurer failed: {type(e).__name__}: {e}",
        )


def _fallback_minimal(*, text: str, query: str, reason: str) -> StructureResult:
    """Last-resort: produce a tiny Decomposition that wraps the upstream text
    as the overview. Better than 502'ing the caller."""
    decomp = Decomposition(
        query=query or "",
        overview=text.strip()[:1000] if text else "(empty upstream output)",
        affected_repos=[],
        risks=[],
        open_questions=[],
        subtasks=[],
    )
    return StructureResult(
        decomposition=decomp,
        used_llm_repair=False,
        repair_model=None,
        repair_tokens_in=None,
        repair_tokens_out=None,
        notes=f"fallback: {reason}",
    )


_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*", re.IGNORECASE)
_JSON_FENCE_END_RE = re.compile(r"\s*```\s*$")


def _strip_json_fences(text: str) -> str:
    """Strip leading ```json / ``` fences and trailing ``` from a body."""
    out = _JSON_FENCE_RE.sub("", text, count=1)
    out = _JSON_FENCE_END_RE.sub("", out)
    return out.strip()


def _usage_from(result: Any) -> tuple[int | None, int | None]:
    """Best-effort token-count extraction from a pydantic-ai RunResult."""
    try:
        u = result.usage()
        return (
            getattr(u, "request_tokens", None) or getattr(u, "input_tokens", None),
            getattr(u, "response_tokens", None) or getattr(u, "output_tokens", None),
        )
    except Exception:
        return (None, None)
