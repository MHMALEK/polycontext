"""Gemini adapter.

Two distinct paths:

- **ask** — routed through services/agent-node (`@google/genai` JS SDK). The
  agent loop benefits from agent-node's workspace tools and forced tool-use
  prompt; round-tripping to Python pydantic-ai for ask would lose those.
- **decompose** — direct call from Python via pydantic-ai's ``Agent`` with
  ``output_type=Decomposition``. Gemini's ``responseSchema`` feature is used
  under the hood and the SDK returns a fully-validated ``Decomposition``
  object. No more ``extract_json`` fragility on the decompose path — the
  shape is enforced by the API, not coaxed out of free-form text. (This is
  why ``_extract_json`` is no longer imported here.)

Docs: https://googleapis.github.io/js-genai/ · https://ai.google.dev/gemini-api/docs
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import httpx

from ..models import Decomposition
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

    async def decompose(self, inp: AdapterDecomposeInput) -> AdapterDecomposeResult:
        """Decompose via pydantic-ai → Gemini ``responseSchema``.

        Switched away from the agent-node + extract_json path because the model
        was non-deterministically returning bare ``Subtask`` objects instead of
        the outer ``Decomposition`` wrapper. With ``output_type=Decomposition``,
        the SDK forces the schema via Gemini's structured-output feature and
        returns an already-validated ``Decomposition`` instance.
        """
        t = time.monotonic()
        prompt = DECOMPOSE_PREAMBLE.format(model_tag=self.name) + query_blob(inp)
        model_id = self._decompose_model_id()
        # Lazy import — pydantic-ai is heavy and not needed for ask.
        from pydantic_ai import Agent
        from pydantic_ai.models.gemini import GeminiModel
        from pydantic_ai.settings import ModelSettings

        # The Gemini provider in pydantic-ai reads the API key from env; set it
        # explicitly so it works even when the host shell hasn't exported it.
        os.environ.setdefault("GEMINI_API_KEY", self.settings.gemini_api_key)

        agent = Agent(
            model=GeminiModel(model_id),
            output_type=Decomposition,
            system_prompt=_DECOMPOSE_SYSTEM,
            model_settings=ModelSettings(temperature=0.2),
        )
        result = await agent.run(prompt)
        decomp = result.output  # already a validated Decomposition

        # Fill in the query if the model echoed something different; the caller
        # often relies on this field to correlate.
        if inp.query and not decomp.query:
            decomp = decomp.model_copy(update={"query": inp.query})

        # Re-render as markdown for the UI's "Raw markdown" panel and for any
        # downstream caller (e.g. jira-bridge) that wants a textual rendering.
        markdown = decomp.model_dump_json(indent=2)

        # Best-effort usage extraction — pydantic-ai exposes a Usage object on
        # the result with attributes that vary slightly by version.
        tokens_in: int | None = None
        tokens_out: int | None = None
        try:
            usage_obj = result.usage()
            tokens_in = (
                getattr(usage_obj, "request_tokens", None)
                or getattr(usage_obj, "input_tokens", None)
            )
            tokens_out = (
                getattr(usage_obj, "response_tokens", None)
                or getattr(usage_obj, "output_tokens", None)
            )
        except Exception:
            pass

        return AdapterDecomposeResult(
            adapter=self.name,
            decomposition=decomp,
            markdown=markdown,
            metrics=AdapterMetrics(
                duration_ms=int((time.monotonic() - t) * 1000),
                tokens_in=tokens_in if isinstance(tokens_in, int) else None,
                tokens_out=tokens_out if isinstance(tokens_out, int) else None,
                model=model_id,
                tool_calls=0,
                extra={"sdk": "pydantic-ai", "agent_node": False, "structured_output": True},
            ),
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
