"""OpenAI Agents SDK via services/agent-node (`@openai/agents`).

Docs: https://github.com/openai/openai-agents-js
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import httpx

from ..models import Decomposition
from ._ask_pipeline import AskInvocation, AskPipeline, AskPipelinePolicy
from ._extract_json import extract_json
from ._prompts import ASK_PREAMBLE, DECOMPOSE_PREAMBLE, IMPLEMENT_PREAMBLE, subtask_prompt, ticket_blob
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
    "You are a code Q&A assistant with read-only access to local repositories via "
    "read_file and list_directory tools. Inspect real files before making claims. "
    "Paths are relative to the workspace root (often REPOS_ROOT with one folder per repo). "
    "Cite paths with line ranges. Return only the final answer; no progress narration."
)
_DECOMPOSE_SYSTEM = (
    "You decompose Jira tickets into structured tech work. Output exactly one "
    "JSON object matching the schema in the user message — no prose, no fences."
)
_IMPLEMENT_SYSTEM = (
    "You are guiding implementation in a git worktree. Do not commit, push, "
    "or open pull requests."
)


class OpenAIAgentsAdapter(Adapter):
    name = "openai_agents"
    capabilities: set[Capability] = {"ask", "decompose", "implement"}
    description = (
        "OpenAI Agents SDK via agent-node: SandboxAgent + UnixLocalSandboxClient when cwd is set; "
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
                model_id=self.settings.openai_agents_sdk_model,
                timeout_seconds=float(self.settings.openai_agents_sdk_timeout_seconds),
                cwd=cwd,
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

    def _decompose_model_id(self) -> str:
        g = (self.settings.openai_agents_sdk_decompose_model or "").strip()
        if g:
            return g
        d = (self.settings.decompose_model or "").strip()
        return d if d else "gpt-4.1"

    async def decompose(self, inp: AdapterDecomposeInput) -> AdapterDecomposeResult:
        t = time.monotonic()
        prompt = DECOMPOSE_PREAMBLE.format(model_tag=self.name) + ticket_blob(inp)
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

    async def implement(self, inp: AdapterImplementInput, ctx: ImplementContext) -> AdapterImplementResult:
        t0 = time.monotonic()
        worktree_note = (
            f"\n\nWorktree directory (read-only context for paths): {ctx.worktree_path}\n"
        )
        out = await self._run(
            system=_IMPLEMENT_SYSTEM,
            prompt=IMPLEMENT_PREAMBLE + worktree_note + subtask_prompt(inp),
            model_id=self.settings.openai_agents_sdk_model,
            timeout_seconds=float(self.settings.openai_agents_sdk_timeout_seconds),
            cwd=ctx.worktree_path,
        )
        answer = out.get("answer") or ""
        return AdapterImplementResult(
            adapter=self.name,
            mr_url=None,
            branch=ctx.branch,
            commits=[],
            diff_summary=answer[-1000:].strip(),
            files_changed=[],
            metrics=_metrics_from(out, t0),
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
