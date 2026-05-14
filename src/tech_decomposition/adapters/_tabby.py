"""Adapter backed by Tabby (https://tabby.tabbyml.com).

Tabby is a self-hosted Copilot-style service. Its public REST API
(``openapi.json``) exposes only four endpoints:

  * ``POST /v1/chat/completions``  — OpenAI-compatible chat
  * ``POST /v1/completions``       — code completion
  * ``POST /v1/events``            — telemetry
  * ``GET  /v1/health``            — health probe

The **Answer Engine** feature you see in Tabby's web UI is a UI-level
wrapper, not an additional public endpoint — it composes the same
``/v1/chat/completions`` with Tabby's indexed repository context. So
hitting ``/v1/chat/completions`` from this adapter automatically gets
the Answer Engine's RAG benefit *if* the server has the relevant repos
ingested (configure that via Tabby's web UI or the ``/v1beta/ingestion``
endpoint, which is server-side admin only and outside the bake-off
scope).

Tabby is fundamentally **not an editor** — it returns text, not file
diffs. So:

  * ``ask``       — first-class via chat; benefits from Answer Engine if
                    repos are indexed
  * ``decompose`` — works (the chat endpoint returns whatever JSON we
                    prompt for); quality depends on Tabby's chat model
  * ``implement`` — intentionally ``NotSupported``; pick a real coding
                    agent (claude_sdk, aider, openhands, etc.)

Auth: bearer token in ``settings.tabby_api_key``. The chat model is
configured server-side in Tabby's own config — we don't pass it in the
request.
"""
from __future__ import annotations

import logging
import time

import httpx

from ..models import Decomposition
from ._subprocess import extract_json
from .base import (
    Adapter,
    AdapterAskInput,
    AdapterAskResult,
    AdapterDecomposeInput,
    AdapterDecomposeResult,
    AdapterMetrics,
    Capability,
)

log = logging.getLogger(__name__)


_ASK_SYSTEM = (
    "You are a code Q&A assistant. Answer based on the repositories Tabby "
    "has indexed. Cite file paths with line ranges where possible. Keep the "
    "answer concise and engineer-oriented."
)

_DECOMPOSE_SYSTEM = (
    "You decompose Jira tickets into structured tech work. Use only what you "
    "can verify against the indexed code. Never invent file paths."
)

_DECOMPOSE_USER_TEMPLATE = """Decompose this Jira ticket. Output exactly ONE JSON
object matching this schema and NOTHING else (no prose, no markdown fences):

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
  "decomposition_model": "tabby"
}}

Ticket:
{ticket}
"""


class TabbyAdapter(Adapter):
    name = "tabby"
    capabilities: set[Capability] = {"ask", "decompose"}
    description = (
        "Tabby (tabbyml) — self-hosted Copilot-style chat API. Implement is not "
        "supported (Tabby is not an editor); use a real coding agent for that."
    )

    def health(self) -> dict:
        if not self.settings.tabby_base_url:
            return {"ok": False, "reason": "TABBY_BASE_URL not set"}
        try:
            # Tabby exposes ``GET /v1/health`` on every recent build.
            with httpx.Client(timeout=3.0) as c:
                r = c.get(self.settings.tabby_base_url.rstrip("/") + "/v1/health")
                if r.status_code >= 400:
                    return {"ok": False, "reason": f"tabby /v1/health -> {r.status_code}"}
        except httpx.HTTPError as e:
            return {"ok": False, "reason": f"tabby unreachable: {e}"}
        return {"ok": True}

    async def ask(self, inp: AdapterAskInput) -> AdapterAskResult:
        t = time.monotonic()
        text, usage = await self._chat(
            system=_ASK_SYSTEM, user=inp.query,
            timeout=self.settings.tabby_timeout_seconds,
        )
        return AdapterAskResult(
            adapter=self.name,
            answer=text,
            citations=[],  # Tabby's chat doesn't surface citations in a stable shape
            metrics=AdapterMetrics(
                duration_ms=int((time.monotonic() - t) * 1000),
                tokens_in=usage.get("prompt_tokens"),
                tokens_out=usage.get("completion_tokens"),
                model=usage.get("model"),
            ),
        )

    async def decompose(self, inp: AdapterDecomposeInput) -> AdapterDecomposeResult:
        t = time.monotonic()
        text, usage = await self._chat(
            system=_DECOMPOSE_SYSTEM,
            user=_DECOMPOSE_USER_TEMPLATE.format(ticket=_ticket_blob(inp)),
            timeout=self.settings.tabby_timeout_seconds,
        )
        try:
            decomp = Decomposition.model_validate(extract_json(text))
        except Exception as e:
            raise RuntimeError(
                f"tabby decompose did not return parseable JSON: {e}\n--- raw ---\n{text[:2000]}"
            ) from e
        return AdapterDecomposeResult(
            adapter=self.name,
            decomposition=decomp,
            markdown=text,
            metrics=AdapterMetrics(
                duration_ms=int((time.monotonic() - t) * 1000),
                tokens_in=usage.get("prompt_tokens"),
                tokens_out=usage.get("completion_tokens"),
                model=usage.get("model"),
            ),
        )

    # ----- internals --------------------------------------------------------

    async def _chat(self, *, system: str, user: str, timeout: float) -> tuple[str, dict]:
        """One round-trip to Tabby's OpenAI-compatible chat endpoint.

        Returns ``(assistant_text, usage_dict)``. Raises on non-2xx so the
        adapter wrapper can map to a 502 result. Defensive parsing because
        Tabby's response shape has small differences across versions.
        """
        url = self.settings.tabby_base_url.rstrip("/") + "/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.settings.tabby_api_key:
            headers["Authorization"] = f"Bearer {self.settings.tabby_api_key}"
        body = {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
        }
        async with httpx.AsyncClient(timeout=timeout) as c:
            r = await c.post(url, json=body, headers=headers)
            if r.status_code >= 400:
                raise RuntimeError(
                    f"tabby chat returned {r.status_code}: {r.text[:500]}"
                )
            data = r.json()
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        text = (message.get("content") or "").strip()
        usage = data.get("usage") or {}
        usage["model"] = data.get("model")
        return text, usage


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
