"""Cline SDK via services/agent-node (Node owns ``@cline/sdk``)."""
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
    "You are a code Q&A assistant. Before answering, use read_workspace_file, "
    "grep_workspace, and (when present) find_code to inspect the repository at "
    "the configured working directory. Ground every claim in real code; cite "
    "paths with line numbers when possible. Keep the answer concise and "
    "engineer-oriented."
)

_DECOMPOSE_SYSTEM = (
    "You decompose tasks into structured tech work. Use file-reading "
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


class ClineSDKAdapter(Adapter):
    name = "cline_sdk"
    capabilities: set[Capability] = {"ask", "decompose", "implement"}
    description = (
        "Cline via @cline/sdk in agent-node. Token/cost from the SDK. Local "
        "read_workspace_file and grep_workspace use cwd on agent-node; optional "
        "find_code when SOURCEBOT_URL is set there."
    )

    def health(self) -> dict:
        if not (self.settings.agent_node_url or "").strip():
            return {"ok": False, "reason": "AGENT_NODE_URL not set"}
        try:
            with httpx.Client(timeout=3.0) as c:
                r = c.get(self.settings.agent_node_url.rstrip("/") + "/health")
                if r.status_code >= 400:
                    return {"ok": False, "reason": f"agent-node /health -> {r.status_code}"}
        except httpx.HTTPError as e:
            base = self.settings.agent_node_url.rstrip("/")
            return {"ok": False, "reason": f"agent-node unreachable ({base}/health): {e}"}
        if not self._provider_key()[0]:
            return {"ok": False, "reason": "no LLM provider key in env (GEMINI/ANTHROPIC/OPENAI)"}
        return {"ok": True}

    async def ask(self, inp: AdapterAskInput) -> AdapterAskResult:
        t = time.monotonic()
        out = await self._run(
            system=_ASK_SYSTEM,
            prompt=inp.query,
            cwd=self._cwd_for_repos(inp.repos),
            timeout_seconds=self.settings.agent_node_timeout_seconds,
            enable_find_code=True,
        )
        return AdapterAskResult(
            adapter=self.name,
            answer=(out.get("answer") or "").strip(),
            citations=[],
            metrics=_metrics_from(out, t),
        )

    async def decompose(self, inp: AdapterDecomposeInput) -> AdapterDecomposeResult:
        t = time.monotonic()
        user_prompt = (
            DECOMPOSE_PREAMBLE.format(model_tag=self.name) + query_blob(inp)
        )
        out = await self._run(
            system=_DECOMPOSE_SYSTEM,
            prompt=user_prompt,
            cwd=self._cwd_for_repos(inp.repos),
            timeout_seconds=self.settings.agent_node_timeout_seconds,
            enable_find_code=True,
        )
        answer = out.get("answer") or ""
        try:
            decomp = Decomposition.model_validate(extract_json(answer))
        except Exception as e:
            raise RuntimeError(
                f"cline_sdk decompose did not return parseable JSON: {e}\n"
                f"--- raw ---\n{answer[:2000]}"
            ) from e
        return AdapterDecomposeResult(
            adapter=self.name,
            decomposition=decomp,
            markdown=answer,
            metrics=_metrics_from(out, t),
        )

    async def implement(
        self, inp: AdapterImplementInput, ctx: ImplementContext,
    ) -> AdapterImplementResult:
        t = time.monotonic()
        out = await self._run(
            system=_IMPLEMENT_SYSTEM,
            prompt=IMPLEMENT_PREAMBLE + subtask_prompt(inp),
            cwd=ctx.worktree_path,
            timeout_seconds=self.settings.agent_node_timeout_seconds,
            enable_find_code=False,
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

    def _provider_key(self) -> tuple[str | None, str | None, str | None]:
        if self.settings.gemini_api_key:
            return ("gemini", "gemini-2.5-pro", self.settings.gemini_api_key)
        if self.settings.anthropic_api_key:
            return ("anthropic", "claude-sonnet-4-5", self.settings.anthropic_api_key)
        if self.settings.openai_api_key:
            return ("openai-native", "gpt-4o", self.settings.openai_api_key)
        return (None, None, None)

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
        enable_find_code: bool = False,
    ) -> dict[str, Any]:
        provider_id, model_id, api_key = self._provider_key()
        if not provider_id:
            raise RuntimeError("no LLM provider key configured; cline_sdk needs one")
        url = self.settings.agent_node_url.rstrip("/") + "/adapters/cline/run"
        body: dict[str, Any] = {
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
                f"agent-node cline /run -> {r.status_code}: {r.text[:500]}"
            )
        data = r.json()
        if data.get("ok") is False:
            raise RuntimeError(f"agent-node cline failed: {data.get('error')}")
        return data


def _metrics_from(out: dict[str, Any], start: float) -> AdapterMetrics:
    return AdapterMetrics(
        duration_ms=int((time.monotonic() - start) * 1000),
        tokens_in=out.get("tokensIn"),
        tokens_out=out.get("tokensOut"),
        cost_usd=out.get("costUsd"),
        extra={"bridge_event_tail_size": len(out.get("events") or []), "agent_node": True},
    )
