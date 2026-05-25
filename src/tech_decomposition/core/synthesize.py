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
    workspace_root: str | None = None,
    tools_enabled: bool = False,
):
    """Stream events from a pydantic-AI agent run — the **single
    streaming pipeline** for the chat UI.

    Replaces the six per-SDK streamers (opencode, gemini, claude_code,
    openai_agents, cursor, cline) with one universal path. When
    ``tools_enabled=True`` and ``workspace_root`` is set, the agent gets
    read_file/grep_search/glob/list_directory tools wired up and the
    pydantic-AI agent loop drives them.

    Yields::

        {"kind": "text.delta",  "text": "..."}                       # per-token
        {"kind": "tool.update", "callID": ..., "tool": ..., "status": "running" | "completed" | "error", ...}
        {"kind": "result",      "text": "<full>", "model": ..., "duration_ms": ..., "tool_calls": int}

    Works for Gemini, Anthropic, OpenAI, OpenRouter — any model
    ``llm_registry`` can build.
    """
    import time

    from pydantic_ai.messages import (
        FunctionToolCallEvent,
        FunctionToolResultEvent,
        PartDeltaEvent,
        PartStartEvent,
        TextPartDelta,
        ToolCallPart,
    )

    # Pick the system prompt by mode. Three flavors:
    #   - Agent (tools on + workspace)   → _ASK_AGENT_SYSTEM
    #   - Grounded (snippets, no tools)  → _ASK_SYNTH_SYSTEM
    #   - Direct  (no snippets, no tools)→ _ASK_DIRECT_SYSTEM
    use_tools = bool(tools_enabled and workspace_root)
    if use_tools:
        system_prompt = _ASK_AGENT_SYSTEM
    elif grounding_block:
        system_prompt = _ASK_SYNTH_SYSTEM
    else:
        system_prompt = _ASK_DIRECT_SYSTEM

    agent, model_name = _build_agent(
        settings, system=system_prompt, model_id=model_id,
    )

    if use_tools:
        from pathlib import Path as _Path

        from .agent_tools import Workspace, attach_workspace_tools

        attach_workspace_tools(agent, Workspace(root=_Path(workspace_root)))

    prompt = (
        build_grounded_prompt(grounding_block=grounding_block, query=query)
        if grounding_block
        else query
    )
    t0 = time.monotonic()
    final_text = ""
    tool_calls = 0
    # Per-call bookkeeping so the running → completed pairs match up by id.
    tool_starts: dict[str, dict[str, Any]] = {}

    async for event in agent.run_stream_events(prompt):
        if isinstance(event, PartStartEvent):
            # New content part. Tool-call parts get a "running" event so
            # the UI's chip appears immediately, before the tool runs.
            # Text parts get their deltas via PartDeltaEvent (no emit here).
            if isinstance(event.part, ToolCallPart):
                call_id = event.part.tool_call_id or f"call_{tool_calls}"
                tool = event.part.tool_name or "tool"
                file_path = _extract_file_path(event.part.args)
                tool_starts[call_id] = {
                    "name": tool,
                    "startMs": int((time.monotonic() - t0) * 1000),
                    "file_path": file_path,
                }
                tool_calls += 1
                yield {
                    "kind": "tool.update",
                    "callID": call_id,
                    "tool": tool,
                    "status": "running",
                    **({"filePath": file_path} if file_path else {}),
                }
        elif isinstance(event, PartDeltaEvent):
            if isinstance(event.delta, TextPartDelta) and event.delta.content_delta:
                final_text += event.delta.content_delta
                yield {"kind": "text.delta", "text": event.delta.content_delta}
        elif isinstance(event, FunctionToolCallEvent):
            # Backup path: some models emit FunctionToolCallEvent without
            # a preceding PartStartEvent. Catch those here so the chip
            # still renders.
            call_id = event.part.tool_call_id or f"call_{tool_calls}"
            if call_id not in tool_starts:
                tool = event.part.tool_name or "tool"
                file_path = _extract_file_path(event.part.args)
                tool_starts[call_id] = {
                    "name": tool,
                    "startMs": int((time.monotonic() - t0) * 1000),
                    "file_path": file_path,
                }
                tool_calls += 1
                yield {
                    "kind": "tool.update",
                    "callID": call_id,
                    "tool": tool,
                    "status": "running",
                    **({"filePath": file_path} if file_path else {}),
                }
        elif isinstance(event, FunctionToolResultEvent):
            call_id = event.tool_call_id or ""
            meta = tool_starts.get(call_id, {})
            tool = meta.get("name") or "tool"
            duration_ms = int((time.monotonic() - t0) * 1000) - meta.get("startMs", 0)
            is_error = getattr(event.result, "is_error", False) is True
            err_str: str | None = None
            if is_error:
                content = getattr(event.result, "content", "")
                err_str = (content if isinstance(content, str) else str(content))[:200]
            yield {
                "kind": "tool.update",
                "callID": call_id or f"call_{len(tool_starts)}",
                "tool": tool,
                "status": "error" if is_error else "completed",
                "durationMs": max(0, duration_ms),
                **({"filePath": meta["file_path"]} if meta.get("file_path") else {}),
                **({"error": err_str} if err_str else {}),
            }

    yield {
        "kind": "result",
        "text": final_text.strip(),
        "model": model_name,
        "tokens_in": None,  # Best-effort: pydantic-AI v1 doesn't expose
        "tokens_out": None,  # cumulative usage on run_stream_events.
        "duration_ms": int((time.monotonic() - t0) * 1000),
        "tool_calls": tool_calls,
    }


def _extract_file_path(args: Any) -> str | None:
    """Best-effort extraction of a 'path' arg from a tool call. Used to
    populate the UI chip's path label. None when no obvious path arg."""
    if isinstance(args, str):
        import json as _json
        try:
            args = _json.loads(args)
        except Exception:  # noqa: BLE001
            return None
    if not isinstance(args, dict):
        return None
    for key in ("path", "file_path", "filePath", "filename"):
        v = args.get(key)
        if isinstance(v, str) and v:
            return v
    return None


# Agent-mode system prompt: tools available, must explore before answering.
_ASK_AGENT_SYSTEM = """\
You are a senior code Q&A assistant with read-only tools over the
workspace: read_file, list_directory, glob, grep_search. Ground every
claim in real code from these tools — don't rely on training knowledge.

Mandatory workflow:
1. **Explore before answering.** Call ``glob`` for path patterns or
   ``grep_search`` for content. AT LEAST one tool call is required —
   no answers from memory.
2. **Read 2-3 files.** After locating candidates, ``read_file`` the
   most-relevant ones to confirm what's actually there.
3. **Cite paths + line numbers** in your final answer
   (``path/to/file.py:120-145``).
4. **Be concise.** Engineer-readable bullets, real evidence, brief whys.
5. **Never punt.** If the question is short or vague (e.g. "what is
   roles", "how does upload work"), interpret it as a request to find
   that concept. Do not ask the user to clarify — explore.
"""


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
