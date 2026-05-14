"""Adapter backed by Anthropic's Claude Agent SDK (Python).

The SDK ships with built-in Read/Grep/Glob/Edit/Write/Bash tools that are
already scoped to a working directory, so we get most of what we need for
free. Each method below sets ``cwd`` and an ``allowed_tools`` allowlist to
keep the agent inside its lane:

  * ``ask`` — read-only, scoped to ``settings.repos_root``
  * ``decompose`` — read-only, scoped to ``settings.repos_root``, asks for a
    JSON ``Decomposition`` payload at the end
  * ``implement`` — read+write, scoped to the worktree provisioned by the
    runner; ``Bash`` allowed for things like running formatters

Import is deferred to instance methods so missing the optional dep doesn't
break the registry — ``list_adapters`` reports ``installed=False`` instead.
"""
from __future__ import annotations

import json
import logging
import re
import time

from ..models import Decomposition, Snippet
from .base import (
    Adapter,
    AdapterAskInput,
    AdapterAskResult,
    AdapterDecomposeInput,
    AdapterDecomposeResult,
    AdapterImplementInput,
    AdapterImplementResult,
    AdapterMetrics,
    Capability,
    ImplementContext,
)

log = logging.getLogger(__name__)


def _import_sdk():
    """Late import so the package is only required when the adapter is used."""
    try:
        import claude_agent_sdk as sdk  # type: ignore
    except ImportError as e:
        raise ImportError(
            "claude-agent-sdk is not installed. Add it to your environment "
            "(``pip install claude-agent-sdk``) to enable the claude_sdk adapter."
        ) from e
    return sdk


_DECOMPOSE_SYSTEM = """You are a senior engineer breaking a Jira ticket into a concrete tech decomposition.

Use the Read/Grep/Glob tools to ground every claim in actual code under the
repositories root. Never invent file paths.

When you are done exploring, output exactly one JSON object matching this
schema (no prose, no fences):

{
  "ticket_key": str | null,
  "ticket_title": str,
  "ticket_url": str | null,
  "overview": str,
  "affected_repos": [str],
  "risks": [str],
  "open_questions": [str],
  "subtasks": [
    {
      "title": str,
      "description": str,
      "repo": str,
      "files": [str],
      "file_links": [str],
      "acceptance_criteria": [str],
      "estimated_complexity": "small" | "medium" | "large" | "unknown"
    }
  ],
  "enrichment_model": "",
  "decomposition_model": "claude-sdk"
}
"""

_ASK_SYSTEM = """You are a code Q&A assistant. Use the Read/Grep/Glob tools to find
evidence in the repositories root before answering. Cite each claim by file path and
line range. Keep the answer concise and engineer-oriented."""

_IMPLEMENT_SYSTEM = """You are implementing a single subtask inside a clean git
worktree at the working directory. Edit only files inside this directory. Do not
commit, push, or open MRs — the surrounding system does that after you finish.

When the subtask is satisfied, output a short JSON status:
{"status": "done", "summary": "<one-line description of what changed>"}
"""


class ClaudeSDKAdapter(Adapter):
    name = "claude_sdk"
    capabilities: set[Capability] = {"ask", "decompose", "implement"}
    description = "Anthropic Claude Agent SDK with Read/Grep/Edit/Write/Bash tools."

    def health(self) -> dict:
        try:
            _import_sdk()
        except ImportError as e:
            return {"ok": False, "reason": str(e)}
        if not self.settings.anthropic_api_key:
            return {"ok": False, "reason": "ANTHROPIC_API_KEY not set"}
        return {"ok": True}

    def _model(self) -> str:
        # Reuse the same Claude model that's already in the project's decompose
        # model spec when it's Anthropic; otherwise default to Sonnet 4.5.
        spec = self.settings.decompose_model
        if spec.startswith("anthropic:"):
            return spec.split(":", 1)[1]
        return "claude-sonnet-4-5"

    async def ask(self, inp: AdapterAskInput) -> AdapterAskResult:
        sdk = _import_sdk()
        t = time.monotonic()
        options = sdk.ClaudeAgentOptions(
            model=self._model(),
            system_prompt=_ASK_SYSTEM,
            cwd=str(self.settings.repos_root),
            allowed_tools=["Read", "Grep", "Glob"],
            permission_mode="bypassPermissions",
            max_turns=12,
        )
        answer, citations, metrics_kw = await _run_agent(
            sdk, options, inp.query, collect_citations=True,
        )
        return AdapterAskResult(
            adapter=self.name,
            answer=answer,
            citations=citations,
            metrics=AdapterMetrics(
                duration_ms=int((time.monotonic() - t) * 1000),
                model=options.model,
                **metrics_kw,
            ),
        )

    async def decompose(self, inp: AdapterDecomposeInput) -> AdapterDecomposeResult:
        sdk = _import_sdk()
        ticket_blob = _ticket_blob(inp)
        prompt = f"Decompose this ticket:\n\n{ticket_blob}"
        t = time.monotonic()
        options = sdk.ClaudeAgentOptions(
            model=self._model(),
            system_prompt=_DECOMPOSE_SYSTEM,
            cwd=str(self.settings.repos_root),
            allowed_tools=["Read", "Grep", "Glob"],
            permission_mode="bypassPermissions",
            max_turns=30,
        )
        answer, _, metrics_kw = await _run_agent(sdk, options, prompt, collect_citations=False)
        try:
            decomp_dict = _extract_json(answer)
            decomp = Decomposition.model_validate(decomp_dict)
        except Exception as e:
            raise RuntimeError(
                f"claude_sdk decompose did not return parseable JSON: {e}\n--- raw ---\n{answer[:2000]}"
            ) from e
        return AdapterDecomposeResult(
            adapter=self.name,
            decomposition=decomp,
            markdown=answer,
            metrics=AdapterMetrics(
                duration_ms=int((time.monotonic() - t) * 1000),
                model=options.model,
                **metrics_kw,
            ),
        )

    async def implement(
        self, inp: AdapterImplementInput, ctx: ImplementContext,
    ) -> AdapterImplementResult:
        sdk = _import_sdk()
        prompt = _implement_prompt(inp)
        t = time.monotonic()
        options = sdk.ClaudeAgentOptions(
            model=self._model(),
            system_prompt=_IMPLEMENT_SYSTEM,
            cwd=str(ctx.worktree_path),
            allowed_tools=["Read", "Grep", "Glob", "Edit", "Write", "Bash"],
            permission_mode="bypassPermissions",
            max_turns=60,
        )
        answer, _, metrics_kw = await _run_agent(sdk, options, prompt, collect_citations=False)
        # The runner does the actual commit/push/MR; we just return metrics
        # and a one-line summary so the orchestrator can record it.
        summary = _extract_summary(answer) or "implement run finished"
        return AdapterImplementResult(
            adapter=self.name,
            mr_url=None,
            branch=ctx.branch,
            commits=[],
            diff_summary=summary,
            files_changed=[],
            metrics=AdapterMetrics(
                duration_ms=int((time.monotonic() - t) * 1000),
                model=options.model,
                **metrics_kw,
            ),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _run_agent(sdk, options, prompt: str, *, collect_citations: bool):
    """Run a prompt through ``ClaudeSDKClient`` and collect the final text.

    Returns ``(answer_text, citations, metrics_kwargs)`` where ``metrics_kwargs``
    is ``{tokens_in, tokens_out, cost_usd, tool_calls}`` — anything the SDK
    surfaces in its result messages. Missing fields stay ``None`` / 0.
    """
    answer_parts: list[str] = []
    citations: list[Snippet] = []
    tokens_in = tokens_out = 0
    cost_usd: float | None = None
    tool_calls = 0

    async with sdk.ClaudeSDKClient(options=options) as client:
        await client.query(prompt)
        async for message in client.receive_response():
            # The SDK's message stream surfaces several message types
            # (AssistantMessage, ToolUseBlock, ResultMessage, ...). We
            # introspect attributes defensively so a version bump that
            # changes class names doesn't crash the adapter.
            text = _message_text(message)
            if text:
                answer_parts.append(text)
            if collect_citations:
                citations.extend(_message_citations(message))
            if _is_tool_use(message):
                tool_calls += 1
            usage = _message_usage(message)
            if usage:
                tokens_in += usage.get("input_tokens") or 0
                tokens_out += usage.get("output_tokens") or 0
                if usage.get("cost_usd") is not None:
                    cost_usd = (cost_usd or 0.0) + usage["cost_usd"]

    answer = "\n".join(p for p in answer_parts if p).strip()
    return answer, citations, {
        "tokens_in": tokens_in or None,
        "tokens_out": tokens_out or None,
        "cost_usd": cost_usd,
        "tool_calls": tool_calls,
    }


def _message_text(message) -> str:
    """Best-effort text extraction from any of the SDK's message classes."""
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            txt = getattr(block, "text", None)
            if txt:
                parts.append(txt)
        return "".join(parts)
    return ""


def _is_tool_use(message) -> bool:
    content = getattr(message, "content", None)
    if not isinstance(content, list):
        return False
    return any(getattr(b, "type", None) == "tool_use" for b in content)


def _message_usage(message) -> dict | None:
    usage = getattr(message, "usage", None)
    if usage is None:
        return None
    if isinstance(usage, dict):
        return usage
    return {
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
        "cost_usd": getattr(usage, "cost_usd", None),
    }


def _message_citations(message) -> list[Snippet]:
    """If the assistant used Read/Grep tools, build Snippet citations from
    the tool_use blocks. The SDK gives us file paths and line ranges; we
    synthesize the snippet content from the message's text block when the
    model quoted it.
    """
    content = getattr(message, "content", None)
    if not isinstance(content, list):
        return []
    out: list[Snippet] = []
    for block in content:
        if getattr(block, "type", None) != "tool_use":
            continue
        if getattr(block, "name", "") not in ("Read", "Grep"):
            continue
        params = getattr(block, "input", {}) or {}
        path = params.get("file_path") or params.get("path") or params.get("pattern")
        if not path:
            continue
        repo, rel = _split_repo_path(str(path))
        out.append(Snippet(
            repo=repo,
            path=rel,
            line_start=int(params.get("offset", 1) or 1),
            line_end=int(params.get("offset", 1) or 1) + int(params.get("limit", 0) or 0),
            content="",
            source="anchor",
        ))
    return out


def _split_repo_path(absolute: str) -> tuple[str, str]:
    parts = absolute.lstrip("/").split("/", 2)
    if len(parts) < 2:
        return ("unknown", absolute)
    # ``settings.repos_root`` already lives several dirs deep, so the first
    # path component after it is the repo name. Caller may post-process.
    return (parts[-2] if len(parts) >= 2 else parts[0], "/".join(parts[1:]))


def _ticket_blob(inp: AdapterDecomposeInput) -> str:
    parts = []
    if inp.ticket_key:
        parts.append(f"Ticket key: {inp.ticket_key}")
    if inp.ticket_url:
        parts.append(f"Ticket URL: {inp.ticket_url}")
    if inp.ticket_text:
        parts.append(inp.ticket_text)
    if inp.repos:
        parts.append(f"Repos to consider: {', '.join(inp.repos)}")
    return "\n\n".join(parts)


def _implement_prompt(inp: AdapterImplementInput) -> str:
    if inp.subtask:
        st = inp.subtask
        parts = [
            f"# {st.title}",
            "",
            st.description,
        ]
        if st.acceptance_criteria:
            parts.append("")
            parts.append("## Acceptance criteria")
            for ac in st.acceptance_criteria:
                parts.append(f"- {ac}")
        if st.files:
            parts.append("")
            parts.append("## Likely files to touch")
            for f in st.files:
                parts.append(f"- {f}")
        return "\n".join(parts)
    return inp.free_text or "Implement the requested task."


_JSON_OBJECT_RE = re.compile(r"\{(?:[^{}]|\{[^{}]*\})*\}", re.DOTALL)


def _extract_json(text: str) -> dict:
    """Pull the largest top-level JSON object out of a possibly-noisy answer."""
    candidates = _JSON_OBJECT_RE.findall(text)
    candidates.sort(key=len, reverse=True)
    for c in candidates:
        try:
            obj = json.loads(c)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise ValueError("no JSON object found in agent output")


def _extract_summary(text: str) -> str | None:
    try:
        obj = _extract_json(text)
    except ValueError:
        return None
    if isinstance(obj, dict):
        s = obj.get("summary")
        if isinstance(s, str):
            return s
    return None
