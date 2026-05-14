"""Adapter backed by Aider (https://aider.chat).

Aider is purpose-built for surgical, repo-aware edits. That's its sweet
spot, and ``implement`` here uses it the way users use it day-to-day.

For completeness we also implement ``ask`` and ``decompose`` via Aider's
``ask`` edit-format (read-only chat with full repo map context). Quality on
Q&A is generally lower than Sourcebot — Aider's strength is producing edits,
not citations — but exposing the capability lets the bake-off compare.

Aider's Python API is synchronous, so we offload calls to a thread with
``asyncio.to_thread`` to keep FastAPI's event loop free.

Auto-commit handling: Aider commits on every ``coder.run`` by default. We
disable that (``auto_commits=False``) so the implement runner can produce
a single squashed commit at the end with a uniform message format.
"""
from __future__ import annotations

import asyncio
import logging
import time

from ..models import Decomposition
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
from ._subprocess import extract_json

log = logging.getLogger(__name__)


def _import_aider():
    try:
        from aider.coders import Coder  # type: ignore
        from aider.io import InputOutput  # type: ignore
        from aider.models import Model  # type: ignore
    except ImportError as e:
        raise ImportError(
            "aider-chat is not installed. Add it to your environment "
            "(``pip install aider-chat``) to enable the aider adapter."
        ) from e
    return Coder, InputOutput, Model


class AiderAdapter(Adapter):
    name = "aider"
    capabilities: set[Capability] = {"ask", "decompose", "implement"}
    description = (
        "Aider (aider.chat) — repo-aware surgical editor. Strongest at implement; "
        "ask/decompose go through Aider's read-only chat mode."
    )

    def health(self) -> dict:
        try:
            _import_aider()
        except ImportError as e:
            return {"ok": False, "reason": str(e)}
        if not (
            self.settings.anthropic_api_key
            or self.settings.openai_api_key
            or self.settings.gemini_api_key
        ):
            return {"ok": False, "reason": "no LLM API key configured"}
        return {"ok": True}

    def _model_name(self) -> str:
        """Map the project's decompose_model spec to an Aider/LiteLLM name.

        Accepts both ``provider:model`` (e.g. ``anthropic:claude-sonnet-4-5``)
        and bare names (``gemini-2.5-pro``). Bare names are sniffed: a leading
        ``gemini-`` or ``gpt-`` is enough to infer the provider so users don't
        have to update existing ``.env`` files to use this adapter.
        """
        spec = (self.settings.decompose_model or "").strip()
        if ":" in spec:
            provider, model = spec.split(":", 1)
            if provider == "anthropic":
                return model
            if provider == "openai":
                return model
            if provider == "gemini":
                return f"gemini/{model}"
            if provider == "openrouter":
                return f"openrouter/{model}"
        if spec.startswith("gemini-") or spec.startswith("models/gemini-"):
            return f"gemini/{spec.removeprefix('models/')}"
        if spec.startswith("gpt-") or spec.startswith("o1-") or spec.startswith("o3"):
            return spec
        if spec.startswith("claude-"):
            return spec
        # Pick a provider based on which API key is available so we don't
        # silently call out to a provider the user hasn't paid for.
        if self.settings.anthropic_api_key:
            return "claude-sonnet-4-5"
        if self.settings.gemini_api_key:
            return "gemini/gemini-2.5-pro"
        if self.settings.openai_api_key:
            return "gpt-4o"
        return "claude-sonnet-4-5"

    async def ask(self, inp: AdapterAskInput) -> AdapterAskResult:
        """Q&A via Aider's ``ask`` edit format — read-only chat with repo map context.

        We point Aider at ``settings.repos_root`` so its repo map covers every
        configured repo. Quality varies vs. Sourcebot: Aider gives a chat-like
        answer rather than ranked citations, but for "how does X work" it can
        be useful, and the bake-off lets you compare directly.
        """
        Coder, InputOutput, Model = _import_aider()
        model_name = self._model_name()
        repos_root = self.settings.repos_root

        def _sync_ask() -> dict:
            t0 = time.monotonic()
            io = InputOutput(yes=True, pretty=False)
            model = Model(model_name)
            import os
            old = os.getcwd()
            os.chdir(str(repos_root))
            try:
                coder = Coder.create(
                    main_model=model,
                    io=io,
                    edit_format="ask",
                    auto_commits=False,
                    dirty_commits=False,
                )
                coder.run(inp.query)
            finally:
                os.chdir(old)
            elapsed = time.monotonic() - t0
            # The last assistant message lives on the coder; field name has
            # churned. Try the most common attributes in order.
            answer = (
                getattr(coder, "last_assistant_message", None)
                or _last_message(coder)
                or ""
            )
            return {
                "answer": answer,
                "duration_ms": int(elapsed * 1000),
                "tokens_in": getattr(coder, "message_tokens_sent", None),
                "tokens_out": getattr(coder, "message_tokens_received", None),
                "cost_usd": getattr(coder, "total_cost", None),
            }

        try:
            res = await asyncio.to_thread(_sync_ask)
        except Exception as e:
            raise RuntimeError(f"aider ask failed: {type(e).__name__}: {e}") from e

        return AdapterAskResult(
            adapter=self.name,
            answer=res["answer"],
            citations=[],  # Aider doesn't emit structured citations
            metrics=AdapterMetrics(
                duration_ms=res["duration_ms"],
                tokens_in=res["tokens_in"],
                tokens_out=res["tokens_out"],
                cost_usd=res["cost_usd"],
                model=model_name,
            ),
        )

    async def decompose(self, inp: AdapterDecomposeInput) -> AdapterDecomposeResult:
        """Decompose via Aider's ``ask`` mode with a JSON-output prompt.

        Aider has no native concept of "produce structured planning output"
        so we tack a JSON schema onto the prompt and parse what comes back.
        Output quality depends heavily on the underlying model — same
        contract the claude_sdk adapter uses.
        """
        Coder, InputOutput, Model = _import_aider()
        model_name = self._model_name()
        repos_root = self.settings.repos_root
        prompt = _DECOMPOSE_PROMPT.format(ticket=_ticket_blob(inp))

        def _sync_decompose() -> dict:
            t0 = time.monotonic()
            io = InputOutput(yes=True, pretty=False)
            model = Model(model_name)
            import os
            old = os.getcwd()
            os.chdir(str(repos_root))
            try:
                coder = Coder.create(
                    main_model=model,
                    io=io,
                    edit_format="ask",
                    auto_commits=False,
                    dirty_commits=False,
                )
                coder.run(prompt)
            finally:
                os.chdir(old)
            elapsed = time.monotonic() - t0
            answer = (
                getattr(coder, "last_assistant_message", None)
                or _last_message(coder)
                or ""
            )
            return {
                "answer": answer,
                "duration_ms": int(elapsed * 1000),
                "tokens_in": getattr(coder, "message_tokens_sent", None),
                "tokens_out": getattr(coder, "message_tokens_received", None),
                "cost_usd": getattr(coder, "total_cost", None),
            }

        try:
            res = await asyncio.to_thread(_sync_decompose)
        except Exception as e:
            raise RuntimeError(f"aider decompose failed: {type(e).__name__}: {e}") from e

        try:
            decomp_dict = extract_json(res["answer"])
            decomp = Decomposition.model_validate(decomp_dict)
        except Exception as e:
            raise RuntimeError(
                f"aider decompose did not return parseable Decomposition JSON: {e}\n"
                f"--- raw ---\n{res['answer'][:2000]}"
            ) from e

        return AdapterDecomposeResult(
            adapter=self.name,
            decomposition=decomp,
            markdown=res["answer"],
            metrics=AdapterMetrics(
                duration_ms=res["duration_ms"],
                tokens_in=res["tokens_in"],
                tokens_out=res["tokens_out"],
                cost_usd=res["cost_usd"],
                model=model_name,
            ),
        )

    async def implement(
        self, inp: AdapterImplementInput, ctx: ImplementContext,
    ) -> AdapterImplementResult:
        Coder, InputOutput, Model = _import_aider()
        prompt = _implement_prompt(inp)
        model_name = self._model_name()

        def _sync_run() -> dict:
            t0 = time.monotonic()
            io = InputOutput(yes=True, pretty=False)
            model = Model(model_name)
            fnames = [str(ctx.worktree_path / f) for f in (inp.subtask.files if inp.subtask else [])]
            coder = Coder.create(
                main_model=model,
                io=io,
                fnames=fnames or None,
                auto_commits=False,
                dirty_commits=False,
                dry_run=False,
            )
            # Aider walks the repo from cwd. Set it to the worktree so the repo
            # map is correct and edits land in the right place.
            import os
            old = os.getcwd()
            os.chdir(str(ctx.worktree_path))
            try:
                coder.run(prompt)
            finally:
                os.chdir(old)
            elapsed = time.monotonic() - t0
            # Aider exposes token usage on the coder after a run; field names
            # have churned across versions, so introspect defensively.
            tokens_in = getattr(coder, "message_tokens_sent", None) or getattr(coder, "total_tokens_sent", None)
            tokens_out = getattr(coder, "message_tokens_received", None) or getattr(coder, "total_tokens_received", None)
            cost = getattr(coder, "total_cost", None)
            return {
                "duration_ms": int(elapsed * 1000),
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "cost_usd": cost,
            }

        try:
            usage = await asyncio.to_thread(_sync_run)
        except Exception as e:
            raise RuntimeError(f"aider run failed: {type(e).__name__}: {e}") from e

        return AdapterImplementResult(
            adapter=self.name,
            mr_url=None,
            branch=ctx.branch,
            commits=[],
            diff_summary="",
            files_changed=[],
            metrics=AdapterMetrics(
                duration_ms=usage["duration_ms"],
                tokens_in=usage["tokens_in"],
                tokens_out=usage["tokens_out"],
                cost_usd=usage["cost_usd"],
                model=model_name,
            ),
        )


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
            parts.append("Acceptance criteria:")
            for ac in st.acceptance_criteria:
                parts.append(f"- {ac}")
        return "\n".join(parts)
    return inp.free_text or "Implement the requested task."


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


def _last_message(coder) -> str:
    """Best-effort extraction of the last assistant turn from an Aider Coder.

    Aider's internal field names have churned across versions; check each in
    the order most likely to be present.
    """
    for attr in ("partial_response_content", "last_assistant_message"):
        v = getattr(coder, attr, None)
        if v:
            return v
    msgs = getattr(coder, "done_messages", None) or getattr(coder, "cur_messages", None) or []
    for m in reversed(msgs):
        if isinstance(m, dict) and m.get("role") == "assistant":
            return m.get("content", "") or ""
    return ""


_DECOMPOSE_PROMPT = """Decompose this Jira ticket into a tech decomposition. Use the repository
context you can see to ground every file reference; never invent paths.

Output exactly one JSON object matching this schema and NOTHING ELSE (no prose,
no fences, no markdown):

{{
  "ticket_key": str | null,
  "ticket_title": str,
  "ticket_url": str | null,
  "overview": str,
  "affected_repos": [str],
  "risks": [str],
  "open_questions": [str],
  "subtasks": [
    {{
      "title": str,
      "description": str,
      "repo": str,
      "files": [str],
      "file_links": [str],
      "acceptance_criteria": [str],
      "estimated_complexity": "small" | "medium" | "large" | "unknown"
    }}
  ],
  "enrichment_model": "",
  "decomposition_model": "aider"
}}

Ticket:
{ticket}
"""
