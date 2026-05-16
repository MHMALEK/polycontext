"""OpenAI Agents SDK via services/agent-node (`@openai/agents`).

Docs: https://github.com/openai/openai-agents-js
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import httpx

from ..models import Decomposition
from ._extract_json import extract_json
from ._prompts import DECOMPOSE_PREAMBLE, query_blob
from .base import (
    Adapter,
    AdapterAskInput,
    AdapterAskResult,
    AdapterDecomposeInput,
    AdapterDecomposeResult,
    AdapterMetrics,
    Capability,
)

_ASK_SYSTEM = (
    "You are a code Q&A assistant with read-only access to local repositories via "
    "read_file, list_directory, search_files, and grep_search tools. Inspect real files before making claims. "
    "Paths are relative to the workspace root (often REPOS_ROOT with one folder per repo). "
    "Cite paths with line ranges. Return only the final answer; no progress narration."
)
_DECOMPOSE_SYSTEM = (
    "You decompose tasks into structured tech work. Output exactly one "
    "JSON object matching the schema in the user message — no prose, no fences."
)


class OpenAIAgentsAdapter(Adapter):
    name = "openai_agents"
    capabilities: set[Capability] = {"ask", "decompose"}
    description = (
        "OpenAI Agents SDK via agent-node: Agent + read-only workspace tools when cwd is set; "
        "plain Agent otherwise."
    )

    def health(self) -> dict:
        if not (self.settings.agent_node_url or "").strip():
            return {"ok": False, "reason": "AGENT_NODE_URL not set"}
        if not self.settings.openai_api_key:
            return {"ok": False, "reason": "OPENAI_API_KEY not set"}
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
            model_id=self.settings.openai_agents_sdk_model,
            timeout_seconds=float(self.settings.openai_agents_sdk_timeout_seconds),
            cwd=self._cwd_for_repos(inp.repos),
        )
        return AdapterAskResult(
            adapter=self.name,
            answer=(out.get("answer") or "").strip(),
            citations=[],
            metrics=_metrics_from(out, t),
        )

    def _decompose_model_id(self) -> str:
        g = (self.settings.openai_agents_sdk_decompose_model or "").strip()
        if g:
            return g
        d = (self.settings.decompose_model or "").strip()
        return d if d else "gpt-4.1"

    async def decompose(self, inp: AdapterDecomposeInput) -> AdapterDecomposeResult:
        t = time.monotonic()
        prompt = DECOMPOSE_PREAMBLE.format(model_tag=self.name) + query_blob(inp)
        out = await self._run(
            system=_DECOMPOSE_SYSTEM,
            prompt=prompt,
            model_id=self._decompose_model_id(),
            timeout_seconds=float(self.settings.openai_agents_sdk_timeout_seconds),
            cwd=self._cwd_for_repos(inp.repos),
        )
        answer = out.get("answer") or ""
        decomp = Decomposition.model_validate(extract_json(answer))
        return AdapterDecomposeResult(
            adapter=self.name,
            decomposition=decomp,
            markdown=answer,
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
        model_id: str,
        timeout_seconds: float,
        cwd: Path | str | None = None,
    ) -> dict[str, Any]:
        url = self.settings.agent_node_url.rstrip("/") + "/adapters/openai_agents/run"
        body: dict[str, Any] = {
            "systemPrompt": system,
            "prompt": prompt,
            "apiKey": self.settings.openai_api_key,
            "modelId": model_id,
            "timeoutSec": int(timeout_seconds),
        }
        if cwd is not None:
            body["cwd"] = str(cwd)
        async with httpx.AsyncClient(timeout=timeout_seconds + 30) as c:
            r = await c.post(url, json=body)
        if r.status_code >= 400:
            raise RuntimeError(f"agent-node openai_agents /run -> {r.status_code}: {r.text[:500]}")
        data = r.json()
        if data.get("ok") is False:
            raise RuntimeError(f"agent-node openai_agents failed: {data.get('error')}")
        return data


def _metrics_from(out: dict[str, Any], start: float) -> AdapterMetrics:
    return AdapterMetrics(
        duration_ms=int((time.monotonic() - start) * 1000),
        tokens_in=_as_int(out.get("tokensIn")),
        tokens_out=_as_int(out.get("tokensOut")),
        model=_as_str(out.get("model")),
        tool_calls=_as_int(out.get("toolCalls")) or 0,
        extra={"sdk": "@openai/agents", "agent_node": True},
    )


def _as_int(v: Any) -> int | None:
    return v if isinstance(v, int) else None


def _as_str(v: Any) -> str | None:
    return v if isinstance(v, str) and v else None
