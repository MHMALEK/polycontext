"""Shared post-adapter step that turns raw LLM text into a validated ``Decomposition``.

Every adapter (Cursor, Cline, Claude Code, Gemini, OpenAI Agents, OpenCode)
runs its own LLM with its own prompts and tools. They each return a chunk of
text whose shape is *supposed* to be the Decomposition JSON, but in practice:

  - Cursor's Composer-2 sometimes wraps the JSON in markdown fences.
  - Gemini occasionally returns a bare ``Subtask`` object instead of the
    ``Decomposition`` wrapper.
  - Cline/OpenCode/Claude Code can interleave reasoning prose around the
    JSON block.

Before this module, every adapter ran its own ``extract_json`` + ``model_validate``
post-process. When the model produced something the regex couldn't capture or
the schema couldn't bind, the whole call 502'd — every adapter had to handle
the same problem its own way.

This module centralizes the step. The flow becomes:

  1. Adapter produces raw text (its model's best attempt at the schema).
  2. ``structure_decomposition(text, query, settings)`` returns a validated
     ``Decomposition`` — always. Fast path tries pure parsing; on failure it
     escalates to a Gemini Flash call with
     ``pydantic_ai.Agent(output_type=Decomposition)`` which uses Gemini's
     ``responseSchema`` to force schema-valid output.
  3. The orchestration layer (``Adapter.decompose`` or ``api.py``) combines
     the structurer's output with the adapter's metrics and returns the
     final ``AdapterDecomposeResult``.

This keeps every adapter focused on *driving its LLM*. Schema enforcement
lives in exactly one place.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from ..adapters._extract_json import extract_json
from ..config import Settings
from ..models import Decomposition


@dataclass
class StructureResult:
    """What the structurer returns alongside the validated decomposition."""

    decomposition: Decomposition
    used_llm_repair: bool
    repair_model: str | None
    repair_tokens_in: int | None
    repair_tokens_out: int | None
    notes: str  # human-readable summary of which path fired


_REPAIR_SYSTEM = """\
You are repairing the output of an upstream code-decomposition LLM. The
upstream model was supposed to emit a JSON object matching the
``Decomposition`` schema (query, overview, affected_repos, risks,
open_questions, subtasks[...], enrichment_model, decomposition_model) but
its output is unstructured, partial, or wrapped incorrectly.

Your job: return a ``Decomposition`` object that faithfully captures the
intent of the upstream text. Rules:

  - PRESERVE the upstream content verbatim where possible. Do not invent
    new subtasks, repos, files, or acceptance criteria.
  - If the upstream output is a single bare ``Subtask`` object (no
    ``Decomposition`` wrapper), wrap it: use it as the only entry in
    ``subtasks``, set ``query`` to the user's original question,
    ``overview`` to the upstream description (or a 2-sentence summary if
    description was a list), and infer ``affected_repos`` from
    ``subtask.repo``.
  - If the upstream output has a partial Decomposition (some top-level
    fields missing), fill the missing fields from context. Empty arrays /
    empty strings are fine — do not fabricate.
  - If the upstream output is free prose (no JSON), parse it into the
    smallest reasonable Decomposition.

Return ONLY the Decomposition object. No prose around it.
"""


async def structure_decomposition(
    *,
    text: str,
    query: str,
    settings: Settings,
    model_tag: str = "",
) -> StructureResult:
    """Validate or repair the upstream LLM's decompose output.

    Fast path: parse JSON from the text and validate against the schema.
    Slow path: if fast path fails, call Gemini Flash via pydantic-ai with
    ``output_type=Decomposition`` to restructure the text. The Flash call
    sees the original user query + the failing upstream output and is
    forced (by Gemini's responseSchema) to emit a schema-valid object.
    """
    # Fast path — extract + validate. No LLM cost when the upstream output
    # is already well-shaped (which is the common case).
    try:
        obj = extract_json(text)
    except ValueError:
        obj = None

    if isinstance(obj, dict):
        try:
            decomp = Decomposition.model_validate(obj)
            # Backfill query if upstream skipped it (some adapters echo a
            # different framing or leave it empty).
            if query and not decomp.query:
                decomp = decomp.model_copy(update={"query": query})
            return StructureResult(
                decomposition=decomp,
                used_llm_repair=False,
                repair_model=None,
                repair_tokens_in=None,
                repair_tokens_out=None,
                notes="fast-path: extract_json + model_validate ok",
            )
        except Exception:
            pass  # fall through to LLM repair

    # Slow path — LLM repair.
    result = await _repair_with_llm(
        text=text, query=query, settings=settings, model_tag=model_tag,
    )
    return result


async def _repair_with_llm(
    *,
    text: str,
    query: str,
    settings: Settings,
    model_tag: str,
) -> StructureResult:
    """Use pydantic-ai with output_type=Decomposition to coerce the upstream
    text into a valid object. Gemini Flash is the default — fast and cheap
    (~$0.0001/call). Configurable via ``enrich_model`` setting.
    """
    if not settings.gemini_api_key:
        # No repair model available — last resort: synthesize a minimal
        # decomposition from the query alone so callers don't 502.
        return _fallback_minimal(text=text, query=query, reason="GEMINI_API_KEY not set")

    try:
        from pydantic_ai import Agent
        from pydantic_ai.models.gemini import GeminiModel
        from pydantic_ai.settings import ModelSettings
    except ImportError:
        return _fallback_minimal(text=text, query=query, reason="pydantic-ai not installed")

    os.environ.setdefault("GEMINI_API_KEY", settings.gemini_api_key)
    model_name = (settings.enrich_model or "gemini-2.5-flash").strip() or "gemini-2.5-flash"

    try:
        agent = Agent(
            model=GeminiModel(model_name),
            output_type=Decomposition,
            system_prompt=_REPAIR_SYSTEM,
            model_settings=ModelSettings(temperature=0.0),
        )
        prompt = (
            f"USER QUESTION:\n{query.strip()}\n\n"
            f"UPSTREAM LLM OUTPUT (may be malformed):\n{text.strip()[:8000]}\n"
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
            used_llm_repair=True,
            repair_model=model_name,
            repair_tokens_in=toks_in,
            repair_tokens_out=toks_out,
            notes=f"llm-repair via {model_name}",
        )
    except Exception as e:
        return _fallback_minimal(
            text=text, query=query, reason=f"llm-repair failed: {type(e).__name__}: {e}",
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
