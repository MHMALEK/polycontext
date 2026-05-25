"""Cursor SDK local runtime via services/agent-node (Node owns @cursor/sdk)."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import httpx

from ._prompts import DECOMPOSE_PREAMBLE, query_blob
from .base import (
    Adapter,
    AdapterAskInput,
    AdapterAskResult,
    AdapterDecomposeInput,
    AdapterMetrics,
    Capability,
    RawDecomposeText,
)

_ASK_SYSTEM = (
    "You are a code Q&A assistant running on local checked-out repositories. "
    "Use tools to inspect files before making claims. Cite file paths with "
    "line ranges. Return only the final answer; no progress narration."
)
_DECOMPOSE_SYSTEM = (
    "You decompose tasks into structured tech work using local repo "
    "evidence. Output exactly one JSON object matching the requested schema."
)


class CursorSDKAdapter(Adapter):
    name = "cursor"
    capabilities: set[Capability] = {"ask", "decompose"}
    description = "Cursor SDK local runtime (agent-node); runs against local repos on disk."

    def health(self) -> dict:
        if not (self.settings.agent_node_url or "").strip():
            return {"ok": False, "reason": "AGENT_NODE_URL not set"}
        if not self.settings.cursor_api_key:
            return {"ok": False, "reason": "CURSOR_API_KEY not set"}
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
        from ..core.explore_directive import wrap_with_explore_directive
        t = time.monotonic()
        # Inject the must-explore directive into the user message so cheap
        # models don't punt with "could you clarify?". See
        # core/explore_directive.py for why this goes here vs system prompt.
        prompt = wrap_with_explore_directive(
            inp.query, tools_on=inp.tools_enabled and not inp.grounded,
        )
        out = await self._run(
            system=_ASK_SYSTEM,
            prompt=prompt,
            cwd=self._cwd_for_repos(inp.repos),
            timeout_seconds=self.settings.agent_node_timeout_seconds,
        )
        return AdapterAskResult(
            adapter=self.name,
            answer=(out.get("answer") or "").strip(),
            citations=[],
            metrics=_metrics_from(out, t),
        )

    async def astream(self, inp: AdapterAskInput):
        """Proxy /adapters/cursor/stream — see streamCursor in agent-node."""
        import json
        from ..core.explore_directive import wrap_with_explore_directive
        prompt = wrap_with_explore_directive(
            inp.query, tools_on=inp.tools_enabled and not inp.grounded,
        )
        body: dict[str, Any] = {
            "systemPrompt": _ASK_SYSTEM,
            "prompt": prompt,
            "cwd": str(self._cwd_for_repos(inp.repos)),
            "apiKey": self.settings.cursor_api_key,
            "modelId": (inp.model or self.settings.cursor_sdk_model),
            "timeoutSec": int(self.settings.agent_node_timeout_seconds),
        }
        url = self.settings.agent_node_url.rstrip("/") + "/adapters/cursor/stream"
        timeout = httpx.Timeout(connect=30.0, read=None, write=30.0, pool=None)
        async with httpx.AsyncClient(timeout=timeout) as c:
            async with c.stream("POST", url, json=body) as resp:
                if resp.status_code >= 400:
                    text = (await resp.aread()).decode("utf-8", "replace")[:500]
                    raise RuntimeError(
                        f"agent-node cursor /stream -> {resp.status_code}: {text}"
                    )
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if not payload:
                        continue
                    try:
                        ev = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    yield ev
                    if isinstance(ev, dict) and ev.get("kind") == "done":
                        return

    async def _decompose_raw_text(self, inp: AdapterDecomposeInput) -> RawDecomposeText:
        t = time.monotonic()
        prompt = DECOMPOSE_PREAMBLE.format(model_tag=self.name) + query_blob(inp)
        out = await self._run(
            system=_DECOMPOSE_SYSTEM,
            prompt=prompt,
            cwd=self._cwd_for_repos(inp.repos),
            timeout_seconds=self.settings.agent_node_timeout_seconds,
        )
        return RawDecomposeText(
            text=(out.get("answer") or ""),
            metrics=_metrics_from(out, t),
        )

    def _cwd_for_repos(self, repos: list[str] | None) -> Path:
        if repos and len(repos) == 1:
            return self.settings.repo_path(repos[0])
        return Path(self.settings.repos_root)

    async def _run(
        self,
        *,
        system: str,
        prompt: str,
        cwd: Path,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        url = self.settings.agent_node_url.rstrip("/") + "/adapters/cursor/run"
        body = {
            "systemPrompt": system,
            "prompt": prompt,
            "cwd": str(cwd),
            "apiKey": self.settings.cursor_api_key,
            "modelId": self.settings.cursor_sdk_model,
            "timeoutSec": int(timeout_seconds),
        }
        async with httpx.AsyncClient(timeout=timeout_seconds + 30) as c:
            r = await c.post(url, json=body)
        if r.status_code >= 400:
            raise RuntimeError(f"agent-node cursor /run -> {r.status_code}: {r.text[:500]}")
        data = r.json()
        if data.get("ok") is False:
            raise RuntimeError(f"agent-node cursor failed: {data.get('error')}")
        return data


def _metrics_from(out: dict[str, Any], start: float) -> AdapterMetrics:
    return AdapterMetrics(
        duration_ms=int((time.monotonic() - start) * 1000),
        tokens_in=_as_int(out.get("tokensIn")),
        tokens_out=_as_int(out.get("tokensOut")),
        model=_as_str(out.get("model")),
        tool_calls=_as_int(out.get("toolCalls")) or 0,
        extra={
            "cursor_mode": "sdk_local",
            "bridge_event_tail_size": len(out.get("events") or []),
            "run_id": out.get("runId"),
            "agent_id": out.get("agentId"),
            "cache_read_tokens": out.get("cacheReadTokens"),
            "cache_write_tokens": out.get("cacheWriteTokens"),
            "agent_node": True,
        },
    )


def _as_int(v: Any) -> int | None:
    return v if isinstance(v, int) else None


def _as_str(v: Any) -> str | None:
    return v if isinstance(v, str) and v else None
