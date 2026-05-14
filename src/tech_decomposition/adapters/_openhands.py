"""Adapter backed by OpenHands (https://github.com/All-Hands-AI/OpenHands).

Routes verified by probing the live ``ghcr.io/all-hands-ai/openhands:0.50``
container's ``/openapi.json``. Note: OpenHands' ``main`` branch has migrated
to a different ``/api/v1/app-conversations`` surface — when you bump the
pinned image, re-run ``curl http://localhost:PORT/openapi.json`` and update
this file rather than trusting the GitHub source.

v0.50 conversation flow:
  * POST  /api/conversations           with ``InitSessionRequest``
        Body fields used here: ``initial_user_msg``, ``repository``,
        ``conversation_instructions``.
  * POST  /api/conversations/{id}/start  — kick off the agent
        Body: ``ProvidersSetModel`` (mostly optional).
  * GET   /api/conversations/{id}/events?exclude_hidden=true
        Returns the event log. We poll until the assistant goes idle
        (no new events in N consecutive polls).
  * POST  /api/conversations/{id}/stop  — clean up after we're done.

Note: this version's API expects WebSocket-style real-time messaging for
*follow-up* messages within a session. We only need single-shot Q&A /
decompose / implement here, so the ``initial_user_msg`` field carries the
prompt and we never need a follow-up POST.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import httpx

from ..models import Decomposition
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


_ASK_PREAMBLE = (
    "Answer this question about the codebase mounted in your workspace. "
    "Cite file paths with line ranges where relevant. Be concise.\n\n"
    "Question:\n"
)

_DECOMPOSE_PREAMBLE = """Decompose this Jira ticket into a tech decomposition using
the code in your workspace. Output ONE JSON object matching this schema and
nothing else (no prose, no fences):

{{
  "ticket_key": str | null, "ticket_title": str, "ticket_url": str | null,
  "overview": str, "affected_repos": [str], "risks": [str], "open_questions": [str],
  "subtasks": [
    {{"title": str, "description": str, "repo": str, "files": [str],
      "file_links": [str], "acceptance_criteria": [str],
      "estimated_complexity": "small"|"medium"|"large"|"unknown"}}
  ],
  "enrichment_model": "", "decomposition_model": "openhands"
}}

Ticket:
"""

_IMPLEMENT_PREAMBLE = (
    "You are implementing a subtask inside a clean git worktree at your workspace. "
    "Edit files in place. Do not commit, push, or open MRs — the surrounding "
    "system does that after you finish.\n\n"
    "Subtask:\n"
)


class OpenHandsAdapter(Adapter):
    name = "openhands"
    capabilities: set[Capability] = {"ask", "decompose", "implement"}
    description = "OpenHands (All-Hands-AI v0.50) — autonomous agent via REST API in its Docker container."

    def health(self) -> dict:
        if not self.settings.openhands_base_url:
            return {"ok": False, "reason": "OPENHANDS_BASE_URL not set"}
        try:
            with httpx.Client(timeout=3.0) as c:
                r = c.get(self.settings.openhands_base_url.rstrip("/") + "/health")
                if r.status_code >= 400:
                    return {"ok": False, "reason": f"openhands /health -> {r.status_code}"}
        except httpx.HTTPError as e:
            return {"ok": False, "reason": f"openhands unreachable: {e}"}
        return {"ok": True}

    # ----- jobs -------------------------------------------------------------

    async def ask(self, inp: AdapterAskInput) -> AdapterAskResult:
        t = time.monotonic()
        text = await self._run_to_completion(prompt=_ASK_PREAMBLE + inp.query)
        return AdapterAskResult(
            adapter=self.name,
            answer=text,
            citations=[],
            metrics=AdapterMetrics(duration_ms=int((time.monotonic() - t) * 1000)),
        )

    async def decompose(self, inp: AdapterDecomposeInput) -> AdapterDecomposeResult:
        t = time.monotonic()
        text = await self._run_to_completion(prompt=_DECOMPOSE_PREAMBLE + _ticket_blob(inp))
        try:
            decomp = Decomposition.model_validate(extract_json(text))
        except Exception as e:
            raise RuntimeError(
                f"openhands decompose did not return parseable JSON: {e}\n--- raw ---\n{text[:2000]}"
            ) from e
        return AdapterDecomposeResult(
            adapter=self.name,
            decomposition=decomp,
            markdown=text,
            metrics=AdapterMetrics(duration_ms=int((time.monotonic() - t) * 1000)),
        )

    async def implement(
        self, inp: AdapterImplementInput, ctx: ImplementContext,
    ) -> AdapterImplementResult:
        t = time.monotonic()
        text = await self._run_to_completion(
            prompt=_IMPLEMENT_PREAMBLE + _subtask_prompt(inp),
            repository=ctx.worktree_path.name,
        )
        return AdapterImplementResult(
            adapter=self.name,
            mr_url=None,
            branch=ctx.branch,
            commits=[],
            diff_summary=text[-1000:].strip(),
            files_changed=[],
            metrics=AdapterMetrics(duration_ms=int((time.monotonic() - t) * 1000)),
        )

    # ----- internals --------------------------------------------------------

    async def _run_to_completion(
        self,
        *,
        prompt: str,
        repository: str | None = None,
    ) -> str:
        """Init session → start agent → poll events until idle → stop.

        Returns the concatenated assistant message text. Raises on transport
        errors or when the deadline expires before idle.
        """
        deadline_seconds = self.settings.openhands_timeout_seconds
        base = self.settings.openhands_base_url.rstrip("/")
        headers = {"Content-Type": "application/json"}
        if self.settings.openhands_api_key:
            headers["Authorization"] = f"Bearer {self.settings.openhands_api_key}"

        async with httpx.AsyncClient(base_url=base, headers=headers, timeout=60.0) as c:
            conv_id = await self._init_session(c, prompt=prompt, repository=repository)
            try:
                await self._start(c, conv_id)
                return await self._poll_events(c, conv_id, deadline_seconds=deadline_seconds)
            finally:
                await self._stop_quietly(c, conv_id)

    async def _init_session(
        self, c: httpx.AsyncClient, *, prompt: str, repository: str | None,
    ) -> str:
        body: dict[str, Any] = {"initial_user_msg": prompt}
        if repository:
            body["repository"] = repository
        r = await c.post("/api/conversations", json=body)
        if r.status_code >= 400:
            raise RuntimeError(
                f"openhands POST /api/conversations -> {r.status_code}: {r.text[:300]}"
            )
        data = r.json()
        conv_id = data.get("conversation_id") or data.get("id") or data.get("status_uuid")
        if not conv_id:
            raise RuntimeError(f"openhands created session but no id field in response: {data}")
        return str(conv_id)

    async def _start(self, c: httpx.AsyncClient, conv_id: str) -> None:
        # ProvidersSetModel is mostly optional — empty body kicks the agent
        # off with whatever providers are already configured server-side.
        r = await c.post(f"/api/conversations/{conv_id}/start", json={})
        if r.status_code >= 400:
            raise RuntimeError(
                f"openhands POST /api/conversations/{conv_id}/start -> {r.status_code}: {r.text[:300]}"
            )

    async def _poll_events(
        self, c: httpx.AsyncClient, conv_id: str, *, deadline_seconds: float,
    ) -> str:
        """Long-poll the events stream; stop after 3 idle windows.

        ``exclude_hidden=true`` strips agent-internal observation events that
        aren't useful for the bake-off report. ``start_date`` advances each
        loop so we only see new events.
        """
        deadline = time.monotonic() + deadline_seconds
        from datetime import datetime, timezone
        start_date = datetime.now(timezone.utc).isoformat()
        seen_ids: set[str] = set()
        assistant_text: list[str] = []
        idle_polls = 0

        while True:
            if time.monotonic() > deadline:
                raise RuntimeError(f"openhands poll timed out after {deadline_seconds}s")
            r = await c.get(
                f"/api/conversations/{conv_id}/events",
                params={
                    "exclude_hidden": "true",
                    "start_date": start_date,
                },
            )
            if r.status_code >= 400:
                raise RuntimeError(
                    f"openhands GET events -> {r.status_code}: {r.text[:300]}"
                )
            data = r.json()
            events = data if isinstance(data, list) else (data.get("events") or [])
            new_assistant = False
            for ev in events:
                ev_id = str(ev.get("id") or ev.get("event_id") or "")
                if ev_id and ev_id in seen_ids:
                    continue
                seen_ids.add(ev_id)
                # OpenHands emits typed events. The assistant's user-facing
                # text shows up as a ``message`` action with ``source`` set
                # to ``agent``. Tolerate field-name churn (``message`` vs
                # ``content`` vs nested ``args``).
                source = ev.get("source") or ev.get("role")
                kind = ev.get("type") or ev.get("action") or ev.get("kind")
                if source in ("agent", "assistant") and kind in (
                    "message", "MessageAction", "agent_message",
                ):
                    txt = _event_text(ev)
                    if txt:
                        assistant_text.append(txt)
                        new_assistant = True
            if new_assistant:
                idle_polls = 0
            else:
                idle_polls += 1
                if idle_polls >= 3:
                    break
            await asyncio.sleep(2.0)
        return "\n".join(assistant_text).strip()

    async def _stop_quietly(self, c: httpx.AsyncClient, conv_id: str) -> None:
        try:
            await c.post(f"/api/conversations/{conv_id}/stop", json={})
        except httpx.HTTPError as e:
            log.warning("openhands stop %s failed (non-fatal): %s", conv_id, e)


def _event_text(ev: dict) -> str:
    """Best-effort message-text extraction across OpenHands event shapes."""
    args = ev.get("args")
    if isinstance(args, dict):
        c = args.get("content") or args.get("message")
        if isinstance(c, str) and c.strip():
            return c
    c = ev.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        out: list[str] = []
        for block in c:
            t = block.get("text") if isinstance(block, dict) else None
            if t:
                out.append(t)
        return "".join(out)
    m = ev.get("message")
    return m if isinstance(m, str) else ""


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
