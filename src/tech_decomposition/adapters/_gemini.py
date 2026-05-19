"""Gemini via services/agent-node (`@google/genai` JS SDK).

Both ask and decompose go through agent-node now. Schema enforcement for
the decompose path lives in ``core/decomposition_structurer`` (shared
across every adapter), so the adapter's only job is to drive the model
and hand back the raw text response.

Docs: https://googleapis.github.io/js-genai/ · https://ai.google.dev/gemini-api/docs
"""
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
    "You are a code Q&A assistant with read-only access to local repositories via "
    "the search_files, grep_search, read_file, and list_directory tools.\n\n"
    "MANDATORY: Before answering ANY question about the codebase, you MUST call at "
    "least one of these tools to verify claims against the actual files. Do NOT "
    "answer from training knowledge alone. If you have not read the relevant code, "
    "your answer is wrong by default — answers that invent file paths, class names, "
    "or behavior are unacceptable.\n\n"
    "Workflow:\n"
    "1. Identify what to inspect (relevant files, symbols, or patterns).\n"
    "2. Call tools (typically 3-10 calls) to gather concrete evidence.\n"
    "3. Then write the final answer citing the exact paths and line numbers seen.\n\n"
    "Paths are relative to the workspace root. Return only the final answer; no "
    "progress narration."
)
_DECOMPOSE_SYSTEM = (
    "You decompose tasks into structured tech work. Output exactly one "
    "JSON object matching the schema in the user message — no prose, no fences."
)


class GeminiAdapter(Adapter):
    name = "gemini"
    capabilities: set[Capability] = {"ask", "decompose"}
    description = (
        "Google Gemini via agent-node (`@google/genai`): CallableTool + automatic function "
        "calling for workspace reads when cwd is set."
    )

    def health(self) -> dict:
        if not (self.settings.agent_node_url or "").strip():
            return {"ok": False, "reason": "AGENT_NODE_URL not set"}
        if not self.settings.gemini_api_key:
            return {"ok": False, "reason": "GEMINI_API_KEY not set"}
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
            model_id=self.settings.gemini_sdk_model,
            timeout_seconds=float(self.settings.gemini_sdk_timeout_seconds),
            cwd=self._cwd_for_repos(inp.repos),
        )
        return AdapterAskResult(
            adapter=self.name,
            answer=(out.get("answer") or "").strip(),
            citations=[],
            metrics=_metrics_from(out, t),
        )

    def _decompose_model_id(self) -> str:
        g = (self.settings.gemini_sdk_decompose_model or "").strip()
        if g:
            return g
        d = (self.settings.decompose_model or "").strip()
        return d if d else "gemini-2.5-pro"

    async def _decompose_raw_text(self, inp: AdapterDecomposeInput) -> RawDecomposeText:
        t = time.monotonic()
        prompt = DECOMPOSE_PREAMBLE.format(model_tag=self.name) + query_blob(inp)
        out = await self._run(
            system=_DECOMPOSE_SYSTEM,
            prompt=prompt,
            model_id=self._decompose_model_id(),
            timeout_seconds=float(self.settings.gemini_sdk_timeout_seconds),
            cwd=self._cwd_for_repos(inp.repos),
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
        model_id: str,
        timeout_seconds: float,
        cwd: Path | str | None = None,
    ) -> dict[str, Any]:
        url = self.settings.agent_node_url.rstrip("/") + "/adapters/gemini/run"
        body: dict[str, Any] = {
            "systemPrompt": system,
            "prompt": prompt,
            "apiKey": self.settings.gemini_api_key,
            "modelId": model_id,
            "timeoutSec": int(timeout_seconds),
        }
        if cwd is not None:
            body["cwd"] = str(cwd)
        async with httpx.AsyncClient(timeout=timeout_seconds + 30) as c:
            r = await c.post(url, json=body)
        if r.status_code >= 400:
            raise RuntimeError(f"agent-node gemini /run -> {r.status_code}: {r.text[:500]}")
        data = r.json()
        if data.get("ok") is False:
            raise RuntimeError(f"agent-node gemini failed: {data.get('error')}")
        return data


def _metrics_from(out: dict[str, Any], start: float) -> AdapterMetrics:
    return AdapterMetrics(
        duration_ms=int((time.monotonic() - start) * 1000),
        tokens_in=_as_int(out.get("tokensIn")),
        tokens_out=_as_int(out.get("tokensOut")),
        model=_as_str(out.get("model")),
        tool_calls=_as_int(out.get("toolCalls")) or 0,
        extra={"sdk": "@google/genai", "agent_node": True},
    )


def _as_int(v: Any) -> int | None:
    return v if isinstance(v, int) else None


def _as_str(v: Any) -> str | None:
    return v if isinstance(v, str) and v else None
