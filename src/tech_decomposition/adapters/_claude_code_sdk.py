"""Claude Code via @anthropic-ai/claude-agent-sdk in services/agent-node."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import httpx

from ..models import Decomposition
from ._ask_pipeline import AskInvocation, AskPipeline, AskPipelinePolicy
from ._extract_json import extract_json
from ._prompts import ASK_PREAMBLE, DECOMPOSE_PREAMBLE, IMPLEMENT_PREAMBLE, subtask_prompt, query_blob
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
_IMPLEMENT_SYSTEM = (
    "You are implementing a single subtask inside a clean git worktree at the "
    "working directory. Edit only files inside this directory. Do not commit, "
    "push, or open MRs — that's handled by the surrounding system after you "
    "finish."
)


class ClaudeCodeSDKAdapter(Adapter):
    name = "claude_code"
    capabilities: set[Capability] = {"ask", "decompose", "implement"}
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

    def _ask_policy(self) -> AskPipelinePolicy:
        return AskPipelinePolicy(
            max_attempts=2,
            min_answer_chars=180,
            enforce_contract=True,
            required_sections=(
                "end-to-end flow",
                "repo-by-repo responsibilities",
                "validation, persistence, and async/background processing",
                "user-visible statuses/errors",
            ),
            require_inline_citations_when_grounded=False,
            retry_without_grounding_on_failure=True,
        )

    async def ask(self, inp: AdapterAskInput) -> AdapterAskResult:
        pipeline = AskPipeline(
            self.settings,
            grounded=False,
            preamble=ASK_PREAMBLE,
            policy=self._ask_policy(),
        )
        t = time.monotonic()
        cwd = self._cwd_for_repos(inp.repos)

        async def _invoke_step(prompt_text: str) -> AskInvocation:
            started = time.monotonic()
            out = await self._run(
                system=_ASK_SYSTEM,
                prompt=prompt_text,
                cwd=cwd,
                timeout_seconds=self.settings.agent_node_timeout_seconds,
            )
            return AskInvocation(
                answer=(out.get("answer") or "").strip(),
                payload=out,
                stage_ms=int((time.monotonic() - started) * 1000),
            )

        run = await pipeline.run(inp, _invoke_step)
        payload = run.invocation.payload or {}
        metrics = _metrics_from(payload, t)
        metrics.extra.setdefault(
            "ask_pipeline",
            {
                "prepare_input_ms": run.prepared_input.stage_ms,
                "retrieve_context_ms": run.retrieved_context.stage_ms,
                "prepare_prompt_ms": run.prompt.stage_ms,
                "invoke_model_ms": run.invocation.stage_ms,
                "format_response_ms": run.response.stage_ms,
                "prompt_chars": len(run.prompt.text),
                "retrieved_snippets": len(run.retrieved_context.snippets),
                "attempts": run.response.attempts,
                "contract_issues": run.response.contract_issues,
                "cwd": str(cwd),
            },
        )
        return AdapterAskResult(
            adapter=self.name,
            answer=run.response.answer,
            citations=run.response.citations,
            metrics=metrics,
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
        )
        answer = out.get("answer") or ""
        try:
            decomp = Decomposition.model_validate(extract_json(answer))
        except Exception as e:
            raise RuntimeError(
                f"claude_code decompose did not return parseable JSON: {e}\n"
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
