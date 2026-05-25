"""Single-shot LLM synthesis over prefetched context — no tools."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from ..config import Settings
from .grounding import build_grounded_prompt


@dataclass
class SynthesisResult:
    text: str
    model: str
    tokens_in: int | None = None
    tokens_out: int | None = None
    duration_ms: int = 0


_ASK_SYNTH_SYSTEM = """\
You are a code Q&A assistant. You receive pre-fetched code snippets from the
codebase search index. Use them as primary evidence.

Rules:
- Answer ONLY from the snippets and the question. Do not invent file paths.
- Cite exact paths and line numbers when visible in snippets.
- For list/enumeration questions: list EVERY distinct item found across ALL
  snippets. Include UPPER_SNAKE constant names (e.g. DATA_VIEWER) verbatim.
- If snippets are insufficient, say what is missing — do not guess.
- Be concise but complete. Use markdown lists when listing items.
"""

# Used by the native streaming path when the request is Direct mode — no
# grounding snippets, no tools. The previous prompt told the model to
# answer "only from snippets", which makes it refuse when none were
# provided. This variant is suited for a single-shot LLM-only answer.
_ASK_DIRECT_SYSTEM = """\
You are a senior engineer answering a question directly from your own
knowledge. There are no code snippets attached — answer from what you
know about general engineering practice, common framework patterns, and
the question itself.

Rules:
- Be concise. Give the engineer a useful answer, not a textbook chapter.
- If the question is asking about something repo-specific (e.g. "Where is
  X in OUR codebase"), be honest that you don't have access to the code
  in this mode and suggest using Grounded or Agent mode instead.
- Use markdown lists for enumerations. Code fences for code examples.
- Don't refuse if the question is general — answer it.
"""

_DECOMPOSE_DRAFT_SYSTEM = """\
You decompose engineering tickets into a tech work breakdown for developers.
You receive pre-fetched code snippets — use them as primary evidence.

Write a structured markdown draft with these sections:

## Overview
2-4 sentences framing the work.

## Affected repos
Bullet list of repo names.

## Subtasks
For each subtask use ### heading with:
- Description
- Repo
- Files (only paths you saw in snippets or that clearly follow repo layout)
- Acceptance criteria (bullets)

## Risks
## Open questions

Rules:
- Do NOT output JSON.
- Do NOT invent file paths — if unsure, put the gap in Open questions.
- Prefer concrete symbols and paths from the snippets.
"""


def _normalize_model_id(raw: str, *, default: str) -> str:
    name = (raw or default).strip()
    if ":" in name:
        _, name = name.split(":", 1)
        return name.strip()
    return name


def synthesis_model_name(settings: Settings, *, model_id: str | None = None) -> str:
    if model_id:
        return model_id.strip()
    return _normalize_model_id(
        settings.pipeline_synthesis_model or "gemini-2.5-flash",
        default="gemini-2.5-flash",
    )


def escalation_model_name(settings: Settings) -> str:
    return _normalize_model_id(
        settings.pipeline_escalation_model or "gemini-2.5-pro",
        default="gemini-2.5-pro",
    )


def _build_agent(settings: Settings, *, system: str, model_id: str | None = None):
    """Route through llm_registry so the synthesis model can be ANY provider
    (Gemini, OpenAI, Anthropic, OpenRouter, custom OpenAI-compatible endpoint).

    The ``pipeline_synthesis_model`` setting accepts the same ``provider:name``
    form as the rest of the codebase, e.g.::

        PIPELINE_SYNTHESIS_MODEL=gemini:gemini-2.5-flash
        PIPELINE_SYNTHESIS_MODEL=openrouter:deepseek/deepseek-v3.2
        PIPELINE_SYNTHESIS_MODEL=openrouter:qwen/qwen3-235b-a22b-2507
    """
    from pydantic_ai import Agent
    from pydantic_ai.settings import ModelSettings

    from .llm_registry import get_model

    # Resolve to a full provider:name spec; bare names assume Gemini for compat.
    raw = (model_id or settings.pipeline_synthesis_model or "gemini-2.5-flash").strip()
    full_spec = raw if ":" in raw else f"gemini:{raw}"
    model = get_model(full_spec, settings)
    # Display name in telemetry — strip provider for brevity.
    display = full_spec.split(":", 1)[1] if ":" in full_spec else full_spec
    return Agent(
        model=model,
        system_prompt=system,
        model_settings=ModelSettings(temperature=0.1),
    ), display


async def synthesize_ask(
    *,
    query: str,
    grounding_block: str,
    settings: Settings,
    model_id: str | None = None,
) -> SynthesisResult:
    """One LLM call over prefetched snippets — no tools."""
    import time

    agent, model_name = _build_agent(
        settings, system=_ASK_SYNTH_SYSTEM, model_id=model_id,
    )
    prompt = build_grounded_prompt(grounding_block=grounding_block, query=query)
    t0 = time.monotonic()
    result = await agent.run(prompt)
    text = result.output if isinstance(result.output, str) else str(result.output)
    toks_in, toks_out = _usage_from(result)
    return SynthesisResult(
        text=text.strip(),
        model=model_name,
        tokens_in=toks_in,
        tokens_out=toks_out,
        duration_ms=int((time.monotonic() - t0) * 1000),
    )


async def synthesize_ask_stream(
    *,
    query: str,
    grounding_block: str,
    settings: Settings,
    model_id: str | None = None,
):
    """Stream token deltas from a single-shot LLM call (no tools, no agent loop).

    Used by the API's ``/v1/ask/stream`` for the no-tools modes (Direct,
    Grounded-only) across ANY adapter — bypasses each adapter's SDK and
    streams natively via pydantic-AI. Works for Gemini, Anthropic, OpenAI,
    OpenRouter (any model the ``llm_registry`` can build).

    Yields a sequence of events::

        {"kind": "text.delta",   "text": "..."}   # zero or more
        {"kind": "result",       "text": "<full>", "model": ..., "tokens_in": ..., "tokens_out": ..., "duration_ms": ...}

    The terminal ``result`` event always fires (even on degenerate runs);
    callers can break the loop after seeing it. Raises on hard errors so
    the bridge can emit a structured error event.
    """
    import time

    # Pick a system prompt suited to the mode: grounded uses the strict
    # "answer only from snippets" prompt; direct uses the LLM-only prompt.
    # Without this split, direct-mode queries get refused ("no snippets
    # were provided") because the grounded prompt insists on evidence.
    system_prompt = _ASK_SYNTH_SYSTEM if grounding_block else _ASK_DIRECT_SYSTEM
    agent, model_name = _build_agent(
        settings, system=system_prompt, model_id=model_id,
    )
    prompt = build_grounded_prompt(grounding_block=grounding_block, query=query) if grounding_block else query
    t0 = time.monotonic()
    final_text = ""
    toks_in: int | None = None
    toks_out: int | None = None
    # ``run_stream`` returns an async context manager that yields a
    # StreamedRunResult. ``stream_text(delta=True)`` then yields just the
    # incremental text — exactly what the UI's <StreamingAnswerBubble> wants.
    async with agent.run_stream(prompt) as response:
        async for delta in response.stream_text(delta=True):
            if not delta:
                continue
            final_text += delta
            yield {"kind": "text.delta", "text": delta}
        # Usage is only populated after the stream completes.
        try:
            u = response.usage()
            toks_in = getattr(u, "request_tokens", None) or getattr(u, "input_tokens", None)
            toks_out = getattr(u, "response_tokens", None) or getattr(u, "output_tokens", None)
        except Exception:  # noqa: BLE001
            pass
    yield {
        "kind": "result",
        "text": final_text.strip(),
        "model": model_name,
        "tokens_in": toks_in,
        "tokens_out": toks_out,
        "duration_ms": int((time.monotonic() - t0) * 1000),
    }


async def synthesize_decompose_draft(
    *,
    query: str,
    grounding_block: str,
    settings: Settings,
    model_id: str | None = None,
) -> SynthesisResult:
    """Prose/markdown draft for the structurer to coerce into Decomposition."""
    import time

    agent, model_name = _build_agent(
        settings, system=_DECOMPOSE_DRAFT_SYSTEM, model_id=model_id,
    )
    prompt = (
        f"{grounding_block.rstrip()}\n\n---\n\n"
        f"Ticket / task:\n{query.strip()}\n"
    )
    t0 = time.monotonic()
    result = await agent.run(prompt)
    text = result.output if isinstance(result.output, str) else str(result.output)
    toks_in, toks_out = _usage_from(result)
    return SynthesisResult(
        text=text.strip(),
        model=model_name,
        tokens_in=toks_in,
        tokens_out=toks_out,
        duration_ms=int((time.monotonic() - t0) * 1000),
    )


def _usage_from(result: Any) -> tuple[int | None, int | None]:
    try:
        u = result.usage()
        return (
            getattr(u, "request_tokens", None) or getattr(u, "input_tokens", None),
            getattr(u, "response_tokens", None) or getattr(u, "output_tokens", None),
        )
    except Exception:
        return (None, None)
