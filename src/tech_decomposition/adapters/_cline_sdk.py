"""Adapter backed by Cline's Node SDK, via a Docker sidecar bridge.

Why this exists alongside ``_cline.py``: Cline's CLI gives us
``cline "<prompt>"`` and that's it — no real token/cost telemetry, no
way to inject custom system prompts cleanly, no streaming events. The
SDK (``@cline/sdk``) exposes all of that but it's Node-only. The
sidecar service in ``bridge/cline_sdk/`` runs the SDK and gives us an
HTTP surface this Python adapter can call.

When to pick which:
  * ``cline`` (CLI adapter) — simpler, fewer moving parts, but opaque metrics
  * ``cline_sdk`` (this one) — real token/cost, custom system prompts, will
    be the home for the planned "Sourcebot tool injection" experiment

Both adapters delegate to the same Cline runtime; the difference is purely
how we drive it.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import httpx

from ..models import Decomposition
from ._grounding import (
    DEFAULT_MAX_FILES,
    format_grounding_block,
    retrieve_context_paths,
)
from ._subprocess import extract_json
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


_ASK_SYSTEM = (
    "You are a code Q&A assistant. Ground every claim in real code from the "
    "working directory. Cite file paths with line ranges when possible. "
    "Keep the answer concise and engineer-oriented."
)

_DECOMPOSE_SYSTEM = (
    "You decompose Jira tickets into structured tech work. Use file-reading "
    "tools to verify every file reference; never invent paths. Output EXACTLY "
    "one JSON object matching the schema in the user message — no prose, no "
    "markdown fences."
)

_IMPLEMENT_SYSTEM = (
    "You are implementing a single subtask inside a clean git worktree at the "
    "working directory. Edit only files inside this directory. Do not commit, "
    "push, or open MRs — that's handled by the surrounding system after you "
    "finish."
)

_DECOMPOSE_USER_TEMPLATE = """Decompose this ticket. Output ONE JSON object
matching this schema (and nothing else):

{{
  "ticket_key": str | null, "ticket_title": str, "ticket_url": str | null,
  "overview": str, "affected_repos": [str], "risks": [str], "open_questions": [str],
  "subtasks": [
    {{"title": str, "description": str, "repo": str, "files": [str],
      "file_links": [str], "acceptance_criteria": [str],
      "estimated_complexity": "small"|"medium"|"large"|"unknown"}}
  ],
  "enrichment_model": "", "decomposition_model": "cline_sdk"
}}

Ticket:
{ticket}
"""


class ClineSDKAdapter(Adapter):
    name = "cline_sdk"
    capabilities: set[Capability] = {"ask", "decompose", "implement"}
    description = (
        "Cline via Node SDK (sidecar at bridge/cline_sdk/). Real token/cost "
        "telemetry and custom system prompts. Sourcebot is wired in as an "
        "agent tool the model can call on demand."
    )

    #: Whether to also inject a Sourcebot+ripgrep prelude with concrete
    #: file pointers into the prompt — on top of the tool-call grounding
    #: the bridge already provides. Set True via the ``Grounded`` subclass.
    _prelude_grounding: bool = False

    # ---- health ----------------------------------------------------------------

    def health(self) -> dict:
        if not self.settings.cline_sdk_bridge_url:
            return {"ok": False, "reason": "CLINE_SDK_BRIDGE_URL not set"}
        try:
            with httpx.Client(timeout=3.0) as c:
                r = c.get(self.settings.cline_sdk_bridge_url.rstrip("/") + "/health")
                if r.status_code >= 400:
                    return {"ok": False, "reason": f"bridge /health -> {r.status_code}"}
        except httpx.HTTPError as e:
            return {"ok": False, "reason": f"bridge unreachable: {e}"}
        # The bridge itself can be up while having no API key wired through.
        # Surface the same "no provider key" reason the other adapters use so
        # the bake-off UI shows a consistent picture.
        if not self._provider_key():
            return {"ok": False, "reason": "no LLM provider key in env (GEMINI/ANTHROPIC/OPENAI)"}
        return {"ok": True}

    # ---- ask / decompose / implement ------------------------------------------

    async def ask(self, inp: AdapterAskInput) -> AdapterAskResult:
        t = time.monotonic()
        prompt = await self._maybe_prepend_grounding(inp.query)
        out = await self._run(
            system=_ASK_SYSTEM,
            prompt=prompt,
            cwd=self.settings.repos_root,
            timeout_seconds=self.settings.cline_sdk_timeout_seconds,
            enable_find_code=True,  # Sourcebot as a tool the agent can call
        )
        return AdapterAskResult(
            adapter=self.name,
            answer=out["answer"],
            citations=[],
            metrics=_metrics_from(out, t, self._prelude_grounding),
        )

    async def decompose(self, inp: AdapterDecomposeInput) -> AdapterDecomposeResult:
        t = time.monotonic()
        ticket = _ticket_blob(inp)
        user_prompt = _DECOMPOSE_USER_TEMPLATE.format(ticket=ticket)
        user_prompt = await self._maybe_prepend_grounding(user_prompt, basis=ticket)
        out = await self._run(
            system=_DECOMPOSE_SYSTEM,
            prompt=user_prompt,
            cwd=self.settings.repos_root,
            timeout_seconds=self.settings.cline_sdk_timeout_seconds,
            enable_find_code=True,
        )
        try:
            decomp = Decomposition.model_validate(extract_json(out["answer"]))
        except Exception as e:
            raise RuntimeError(
                f"cline_sdk decompose did not return parseable JSON: {e}\n"
                f"--- raw ---\n{(out['answer'] or '')[:2000]}"
            ) from e
        return AdapterDecomposeResult(
            adapter=self.name,
            decomposition=decomp,
            markdown=out["answer"],
            metrics=_metrics_from(out, t, self._prelude_grounding),
        )

    async def implement(
        self, inp: AdapterImplementInput, ctx: ImplementContext,
    ) -> AdapterImplementResult:
        # No prelude on implement — subtask already names files.
        t = time.monotonic()
        out = await self._run(
            system=_IMPLEMENT_SYSTEM,
            prompt=_subtask_prompt(inp),
            cwd=ctx.worktree_path,
            timeout_seconds=self.settings.cline_sdk_timeout_seconds,
        )
        return AdapterImplementResult(
            adapter=self.name,
            mr_url=None,
            branch=ctx.branch,
            commits=[],
            diff_summary=(out["answer"] or "")[-1000:].strip(),
            files_changed=[],
            metrics=_metrics_from(out, t, self._prelude_grounding),
        )

    # ---- grounding prelude (opt-in via subclass) -------------------------------

    async def _maybe_prepend_grounding(self, prompt: str, *, basis: str | None = None) -> str:
        """Prepend a Sourcebot+ripgrep file-pointer block if grounding is on.

        ``basis`` is the text the retrieval keywords come from — usually
        the same as ``prompt`` for ``ask`` but different for ``decompose``
        (where the prompt embeds the JSON schema and we want to retrieve
        based on the ticket body, not the schema).
        """
        if not self._prelude_grounding:
            return prompt
        try:
            paths = await retrieve_context_paths(
                basis or prompt, self.settings, max_files=DEFAULT_MAX_FILES,
            )
            block = format_grounding_block(paths, self.settings.repos_root)
        except Exception as e:  # noqa: BLE001 — best-effort
            log.warning("%s: grounding prelude failed: %s", self.name, e)
            return prompt
        return block + prompt

    # ---- internals -------------------------------------------------------------

    def _provider_key(self) -> tuple[str | None, str | None, str | None]:
        """Pick a provider+model+key from settings, preferring Gemini.

        Returns ``(providerId, modelId, apiKey)`` matching the bridge's
        accepted shape. Provider IDs come from Cline's
        ``BUILT_IN_PROVIDER_IDS`` — ``gemini`` (not ``google``),
        ``openai-native`` (not ``openai``), ``anthropic``.
        """
        if self.settings.gemini_api_key:
            return ("gemini", "gemini-2.5-pro", self.settings.gemini_api_key)
        if self.settings.anthropic_api_key:
            return ("anthropic", "claude-sonnet-4-5", self.settings.anthropic_api_key)
        if self.settings.openai_api_key:
            return ("openai-native", "gpt-4o", self.settings.openai_api_key)
        return (None, None, None)

    async def _run(
        self, *, system: str, prompt: str, cwd: Path, timeout_seconds: float,
        enable_find_code: bool = False,
    ) -> dict:
        provider_id, model_id, api_key = self._provider_key()
        if not provider_id:
            raise RuntimeError("no LLM provider key configured; cline_sdk needs one")
        url = self.settings.cline_sdk_bridge_url.rstrip("/") + "/run"
        body = {
            "systemPrompt": system,
            "prompt": prompt,
            "cwd": str(cwd),
            "providerId": provider_id,
            "modelId": model_id,
            "apiKey": api_key,
            "timeoutSec": int(timeout_seconds),
            "enableFindCode": enable_find_code,
        }
        async with httpx.AsyncClient(timeout=timeout_seconds + 30) as c:
            r = await c.post(url, json=body)
            if r.status_code >= 400:
                raise RuntimeError(
                    f"cline_sdk bridge /run -> {r.status_code}: {r.text[:500]}"
                )
            return r.json()


def _metrics_from(out: dict, start: float, grounded: bool = False) -> AdapterMetrics:
    extra: dict = {"bridge_event_tail_size": len(out.get("events") or [])}
    if grounded:
        extra["grounded"] = True
    return AdapterMetrics(
        duration_ms=int((time.monotonic() - start) * 1000),
        tokens_in=out.get("tokensIn"),
        tokens_out=out.get("tokensOut"),
        cost_usd=out.get("costUsd"),
        # ``events`` from the bridge is a debugging tail, not user-facing —
        # park its length in ``extra`` so the report can mention it without
        # cluttering the main metrics block.
        extra=extra,
    )


class ClineSDKGroundedAdapter(ClineSDKAdapter):
    """Cline SDK with a Sourcebot+ripgrep file-pointer prelude on ask/decompose.

    The base adapter already exposes Sourcebot to Cline as an agent tool;
    this variant additionally pre-loads concrete file paths into the
    prompt so the model doesn't have to discover them itself. Useful for
    A/B-ing whether the prelude beats tool-only grounding.
    """

    name = "cline_sdk_grounded"
    description = (
        "Cline SDK with a Sourcebot+ripgrep retrieval prelude added on top of "
        "the bridge's Sourcebot-as-tool grounding."
    )
    _prelude_grounding = True


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


def _subtask_prompt(inp: AdapterImplementInput) -> str:
    if inp.subtask:
        st = inp.subtask
        out = [f"# {st.title}", "", st.description]
        if st.acceptance_criteria:
            out.append("")
            out.append("Acceptance criteria:")
            out.extend(f"- {ac}" for ac in st.acceptance_criteria)
        return "\n".join(out)
    return inp.free_text or "Implement the requested task."
