"""Cursor SDK local runtime via services/agent-node (Node owns @cursor/sdk)."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import httpx

from ..models import Decomposition
from ._extract_json import extract_json
from ._prompts import DECOMPOSE_PREAMBLE, IMPLEMENT_PREAMBLE, subtask_prompt, query_blob
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

_ASK_SYSTEM = (
    "You are a code Q&A assistant running on local checked-out repositories. "
    "Use tools to inspect files before making claims. Cite file paths with "
    "line ranges. Return only the final answer; no progress narration."
)
_DECOMPOSE_SYSTEM = (
    "You decompose tasks into structured tech work using local repo "
    "evidence. Output exactly one JSON object matching the requested schema."
)
_IMPLEMENT_SYSTEM = (
    "You implement changes in the provided local worktree. Do not commit, push, "
    "or open pull requests."
)


class CursorSDKAdapter(Adapter):
    name = "cursor"
    capabilities: set[Capability] = {"ask", "decompose", "implement"}
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
        t = time.monotonic()
        out = await self._run(
            system=_ASK_SYSTEM,
            prompt=inp.query,
            cwd=self._cwd_for_repos(inp.repos),
            timeout_seconds=self.settings.agent_node_timeout_seconds,
        )
        return AdapterAskResult(
            adapter=self.name,
            answer=(out.get("answer") or "").strip(),
            citations=[],
            metrics=_metrics_from(out, t),
        )

    async def decompose(self, inp: AdapterDecomposeInput) -> AdapterDecomposeResult:
        t = time.monotonic()
        prompt = DECOMPOSE_PREAMBLE.format(model_tag=self.name) + query_blob(inp)
        out = await self._run(
            system=_DECOMPOSE_SYSTEM,
            prompt=prompt,
            cwd=self._cwd_for_repos(inp.repos),
            timeout_seconds=self.settings.agent_node_timeout_seconds,
        )
        answer = out.get("answer") or ""
        decomp = Decomposition.model_validate(extract_json(answer))
        return AdapterDecomposeResult(
            adapter=self.name,
            decomposition=decomp,
            markdown=answer,
            metrics=_metrics_from(out, t),
        )

    async def implement(self, inp: AdapterImplementInput, ctx: ImplementContext) -> AdapterImplementResult:
        t = time.monotonic()
        out = await self._run(
            system=_IMPLEMENT_SYSTEM,
            prompt=IMPLEMENT_PREAMBLE + subtask_prompt(inp),
            cwd=ctx.worktree_path,
            timeout_seconds=self.settings.agent_node_timeout_seconds,
        )
        answer = out.get("answer") or ""
        return AdapterImplementResult(
            adapter=self.name,
            mr_url=None,
            branch=ctx.branch,
            commits=[],
            diff_summary=answer[-1000:].strip(),
            files_changed=[],
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
