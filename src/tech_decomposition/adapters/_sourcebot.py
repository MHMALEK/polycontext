"""Sourcebot Q&A via agent-node (blocking chat endpoint only)."""
from __future__ import annotations

import time

import httpx

from ..clients.sourcebot import ANSWER_STYLE_SUFFIX, effective_sourcebot_repos_for_ask
from .base import (
    Adapter,
    AdapterAskInput,
    AdapterAskResult,
    AdapterMetrics,
    Capability,
)


class SourcebotAdapter(Adapter):
    name = "sourcebot"
    capabilities: set[Capability] = {"ask"}
    description = (
        "Sourcebot ``/api/chat/blocking`` through agent-node — same answers as "
        "the Sourcebot UI chat."
    )

    def health(self) -> dict:
        if not (self.settings.agent_node_url or "").strip():
            return {"ok": False, "reason": "AGENT_NODE_URL not set"}
        if not (self.settings.sourcebot_url or "").strip():
            return {"ok": False, "reason": "SOURCEBOT_URL not set"}
        if not self.settings.sourcebot_api_key:
            return {"ok": False, "reason": "SOURCEBOT_API_KEY not set"}
        try:
            with httpx.Client(timeout=3.0) as c:
                r = c.get(self.settings.agent_node_url.rstrip("/") + "/health")
                if r.status_code >= 400:
                    return {"ok": False, "reason": f"agent-node /health -> {r.status_code}"}
        except httpx.HTTPError as e:
            base = self.settings.agent_node_url.rstrip("/")
            return {"ok": False, "reason": f"agent-node unreachable ({base}/health): {e}"}
        return {"ok": True}

    async def ask(self, inp: AdapterAskInput) -> AdapterAskResult:
        t0 = time.monotonic()
        repos = effective_sourcebot_repos_for_ask(self.settings, inp.repos)
        sb_base = (self.settings.sourcebot_url_for_agent_node or "").strip() or self.settings.sourcebot_url
        body: dict = {
            "sourcebotUrl": sb_base,
            "sourcebotApiKey": self.settings.sourcebot_api_key,
            "question": inp.query + ANSWER_STYLE_SUFFIX,
            "timeoutSec": int(self.settings.sourcebot_timeout_seconds),
            "maxSteps": 50,
        }
        if repos:
            body["repos"] = repos

        url = self.settings.agent_node_url.rstrip("/") + "/adapters/sourcebot/ask"
        to = float(self.settings.sourcebot_timeout_seconds) + 30
        async with httpx.AsyncClient(timeout=to) as c:
            r = await c.post(url, json=body)
        if r.status_code >= 400:
            raise RuntimeError(f"agent-node sourcebot /ask -> {r.status_code}: {r.text[:500]}")
        data = r.json()
        if data.get("ok") is False:
            raise RuntimeError(f"agent-node sourcebot failed: {data.get('error')}")
        answer = (data.get("answer") or "").strip()
        wall = data.get("wallSeconds")
        metrics = AdapterMetrics(
            duration_ms=int((time.monotonic() - t0) * 1000),
            model=data.get("model"),
            extra={
                "agent_node": True,
                "chat_id": data.get("chatId"),
                "chat_url": data.get("chatUrl"),
                "wall_seconds": wall,
            },
        )
        return AdapterAskResult(adapter=self.name, answer=answer, citations=[], metrics=metrics)
