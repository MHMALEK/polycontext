"""Claude Code via @anthropic-ai/claude-agent-sdk in services/agent-node."""
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
    "You are a code Q&A assistant. Ground every claim in real code from the "
    "working directory. Cite file paths with line ranges when possible. "
    "Keep the answer concise and engineer-oriented."
)
_DECOMPOSE_SYSTEM = (
    "You decompose tasks into structured tech work. Use file-reading "
    "tools to verify every file reference; never invent paths. Output EXACTLY "
    "one JSON object matching the schema in the user message — no prose, no "
    "markdown fences."
)


class ClaudeCodeSDKAdapter(Adapter):
    name = "claude_code"
    capabilities: set[Capability] = {"ask", "decompose"}
    description = (
        "Claude Code capabilities via @anthropic-ai/claude-agent-sdk "
        "(agent-node). Requires ANTHROPIC_API_KEY."
    )

    def health(self) -> dict:
        if not (self.settings.agent_node_url or "").strip():
            return {"ok": False, "reason": "AGENT_NODE_URL not set"}
        if not self.settings.anthropic_api_key:
            return {"ok": False, "reason": "ANTHROPIC_API_KEY not set"}
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
        """Proxy /adapters/claude_code/stream. Yields the normalized event
        union (session/status/text.delta/tool.update/error/done) — same
        shape as opencode/gemini astream so the API bridge handles all
        adapters uniformly."""
        import json
        from ..core.explore_directive import wrap_with_explore_directive
        prompt = wrap_with_explore_directive(
            inp.query, tools_on=inp.tools_enabled and not inp.grounded,
        )
        body: dict[str, Any] = {
            "systemPrompt": _ASK_SYSTEM,
            "prompt": prompt,
            "cwd": str(self._cwd_for_repos(inp.repos)),
            "apiKey": self.settings.anthropic_api_key,
            "model": (inp.model or self.settings.claude_code_model),
            "timeoutSec": int(self.settings.agent_node_timeout_seconds),
        }
        url = self.settings.agent_node_url.rstrip("/") + "/adapters/claude_code/stream"
        timeout = httpx.Timeout(connect=30.0, read=None, write=30.0, pool=None)
        async with httpx.AsyncClient(timeout=timeout) as c:
            async with c.stream("POST", url, json=body) as resp:
                if resp.status_code >= 400:
                    text = (await resp.aread()).decode("utf-8", "replace")[:500]
                    raise RuntimeError(
                        f"agent-node claude_code /stream -> {resp.status_code}: {text}"
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
                    # Stop on terminal event — see _gemini.py / _opencode_sdk.py.
                    if isinstance(ev, dict) and ev.get("kind") == "done":
                        return

    async def _decompose_raw_text(self, inp: AdapterDecomposeInput) -> RawDecomposeText:
        t = time.monotonic()
        user_prompt = (
            DECOMPOSE_PREAMBLE.format(model_tag=self.name) + query_blob(inp)
        )
        out = await self._run(
            system=_DECOMPOSE_SYSTEM,
            prompt=user_prompt,
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
        url = self.settings.agent_node_url.rstrip("/") + "/adapters/claude_code/run"
        body: dict[str, Any] = {
            "systemPrompt": system,
            "prompt": prompt,
            "cwd": str(cwd),
            "apiKey": self.settings.anthropic_api_key,
            "model": self.settings.claude_code_model,
            "timeoutSec": int(timeout_seconds),
        }
        async with httpx.AsyncClient(timeout=timeout_seconds + 30) as c:
            r = await c.post(url, json=body)
        if r.status_code >= 400:
            raise RuntimeError(
                f"agent-node claude_code /run -> {r.status_code}: {r.text[:500]}"
            )
        data = r.json()
        if data.get("ok") is False:
            raise RuntimeError(f"agent-node claude_code failed: {data.get('error')}")
        return data


def _metrics_from(out: dict[str, Any], start: float) -> AdapterMetrics:
    return AdapterMetrics(
        duration_ms=int((time.monotonic() - start) * 1000),
        tokens_in=out.get("tokensIn"),
        tokens_out=out.get("tokensOut"),
        cost_usd=out.get("costUsd"),
        model=_as_str(out.get("model")),
        tool_calls=_as_int(out.get("toolCalls")) or 0,
        extra={
            "sdk": "claude-agent-sdk",
            "bridge_event_tail_size": len(out.get("events") or []),
            "agent_node": True,
            "session_id": out.get("sessionId"),
        },
    )


def _as_int(v: Any) -> int | None:
    return v if isinstance(v, int) else None


def _as_str(v: Any) -> str | None:
    return v if isinstance(v, str) and v else None
