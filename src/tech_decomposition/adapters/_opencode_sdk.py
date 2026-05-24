"""OpenCode SDK via agent-node (@opencode-ai/sdk v2).

Uses session API with structured ``json_schema`` output for decomposition (best
practice per OpenCode SDK docs — clear schemas, retries, StructuredOutput errors).
Prefer ``OPENCODE_SDK_BASE_URL`` / ``createOpencodeClient`` in production when a
server is already running; otherwise agent-node shells ``opencode serve``.
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
    # IMPORTANT: cheap models (DeepSeek V3.2, Qwen3-235B, GLM-4.6) tend to
    # punt with "could you clarify?" when given a short or ambiguous query
    # — they default to asking the user instead of exploring. Stronger
    # cheap-model nudge: explicit ban on clarification questions + a
    # mandatory first-tool-call directive so the agent loop actually fires.
    "You are a code Q&A assistant with read-only access to a multi-repo "
    "monorepo via tools (read, grep, glob, ls). "
    "\n\n"
    "## Operating rules\n"
    "1. **NEVER ask the user for clarification.** If the question is vague "
    "(e.g. 'what is roles', 'how does upload work'), interpret it as a "
    "request to explore the codebase for that concept. Make reasonable "
    "assumptions and investigate.\n"
    "2. **ALWAYS start with at least one tool call.** Before writing ANY "
    "answer, use ``glob`` or ``grep`` to locate relevant files. Do not "
    "answer from training knowledge alone. A single tool call is non-"
    "negotiable.\n"
    "3. **Read 2-3 files before answering.** After your initial search, "
    "``read`` the most-likely files to confirm your understanding. Cite "
    "specific file paths and line numbers.\n"
    "4. **Be concise.** Engineer-readable bullets, real file paths, brief "
    "explanations of the why. No marketing fluff.\n"
    "5. **If after exploration you genuinely cannot find the answer**, say "
    "so explicitly with the search terms you tried — do NOT punt with "
    "'could you clarify?'."
)
_DECOMPOSE_SYSTEM = (
    "You produce tech work breakdowns grounded in workspace tools. Prefer structured "
    "output enforced by schema; paths must correspond to checked-out repositories."
)


class OpencodeSDKAdapter(Adapter):
    name = "opencode"
    capabilities: set[Capability] = {"ask", "decompose"}
    description = (
        "OpenCode via @opencode-ai/sdk v2 (agent-node). Structured JSON decomposition; "
        "set OPENCODE_SDK_BASE_URL to connect to an existing server (recommended)."
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
        pid, api = self._credentials_for_model()
        if not (self.settings.opencode_sdk_base_url or "").strip():
            if not pid or not api:
                return {
                    "ok": False,
                    "reason": (
                        "Configure OPENCODE_SDK_BASE_URL or set credentials for provider "
                        f"{self.settings.opencode_sdk_model.split('/', 1)[0] if '/' in self.settings.opencode_sdk_model else '?'} "
                        "(e.g. ANTHROPIC_API_KEY for anthropic models)"
                    ),
                }
        return {"ok": True}

    async def ask(self, inp: AdapterAskInput) -> AdapterAskResult:
        t = time.monotonic()
        # Resolve credentials for the per-request model override when present,
        # else for the configured default. e.g. ``openrouter/qwen/...`` →
        # provider="openrouter" → uses OPENROUTER_API_KEY.
        model_override = (inp.model or "").strip() or None
        pid, ak = self._credentials_for_model(model_override=model_override)
        out = await self._run(
            system=_ASK_SYSTEM,
            prompt=inp.query,
            cwd=self._cwd_for_repos(inp.repos),
            timeout_seconds=self.settings.agent_node_timeout_seconds,
            structured=False,
            provider_id=pid,
            api_key=ak,
            structured_retry=self.settings.opencode_sdk_structured_retry_count,
            model=model_override,
            tools_enabled=inp.tools_enabled,
        )
        return AdapterAskResult(
            adapter=self.name,
            answer=(out.get("answer") or "").strip(),
            citations=[],
            metrics=_metrics_from(out, t),
        )

    async def _decompose_raw_text(self, inp: AdapterDecomposeInput) -> RawDecomposeText:
        """Drive opencode's chat without structured output.

        OpenCode's ``format: { type: json_schema, ... }`` mode was failing
        unpredictably for some provider+schema combinations ("UnknownError:
        Unexpected server error"). Since the shared structurer
        (``core/decomposition_structurer.py``) now normalises every
        adapter's output, the adapter no longer needs to enforce schema
        itself — emitting free-text JSON the model wants to write and
        letting the structurer reshape is more reliable.
        """
        t = time.monotonic()
        user_prompt = (
            DECOMPOSE_PREAMBLE.format(model_tag=self.name) + query_blob(inp)
        )
        pid, ak = self._credentials_for_model()
        out = await self._run(
            system=_DECOMPOSE_SYSTEM,
            prompt=user_prompt,
            cwd=self._cwd_for_repos(inp.repos),
            timeout_seconds=self.settings.agent_node_timeout_seconds,
            structured=False,
            provider_id=pid,
            api_key=ak,
            structured_retry=self.settings.opencode_sdk_structured_retry_count,
        )
        return RawDecomposeText(
            text=(out.get("answer") or ""),
            metrics=_metrics_from(out, t),
        )

    def _cwd_for_repos(self, repos: list[str] | None) -> Path:
        if repos and len(repos) == 1:
            return self.settings.repo_path(repos[0])
        return Path(self.settings.repos_root)

    def _credentials_for_model(
        self, *, model_override: str | None = None,
    ) -> tuple[str | None, str | None]:
        """Resolve (providerID, apiKey) for the model. When ``model_override``
        is set (per-request UI/SDK override), use its provider prefix to
        select the credential instead of the configured default — so a user
        can pick ``openrouter/deepseek/...`` from the UI even if the env's
        default model is ``anthropic/claude-...``."""
        spec = (model_override or self.settings.opencode_sdk_model).strip()
        if "/" not in spec:
            return (None, None)
        pid, _mid = spec.split("/", 1)
        pid_low = pid.lower().strip()
        if pid_low == "anthropic":
            k = self.settings.anthropic_api_key
            return (pid.strip(), k) if k else (None, None)
        if pid_low in ("google", "gemini"):
            k = self.settings.gemini_api_key
            return (pid.strip(), k) if k else (None, None)
        if pid_low in ("openai", "openrouter"):
            k = (
                self.settings.openai_api_key
                if pid_low == "openai"
                else self.settings.openrouter_api_key
            )
            return (pid.strip(), k) if k else (None, None)
        # Other providers rely on OAuth or server-side secrets on OpenCode itself.
        return (None, None)

    async def _run(
        self,
        *,
        system: str,
        prompt: str,
        cwd: Path,
        timeout_seconds: float,
        structured: bool,
        structured_retry: int,
        provider_id: str | None,
        api_key: str | None,
        model: str | None = None,
        tools_enabled: bool = True,
    ) -> dict[str, Any]:
        url = self.settings.agent_node_url.rstrip("/") + "/adapters/opencode/run"
        body: dict[str, Any] = {
            "systemPrompt": system,
            "prompt": prompt,
            "cwd": str(cwd),
            "timeoutSec": int(timeout_seconds),
            "structured": structured,
            "model": (model or self.settings.opencode_sdk_model).strip(),
        }
        base = self.settings.opencode_sdk_base_url.strip()
        if base:
            body["baseUrl"] = base.rstrip("/")
        if structured and structured_retry >= 0:
            body["structuredRetryCount"] = structured_retry
        if provider_id and api_key:
            body["providerID"] = provider_id
            body["apiKey"] = api_key
        # When tools_enabled=False the user has asked for "grounded-only" mode
        # — i.e. answer single-shot from the prefetched context. Tell agent-node
        # to mask the read-side tools so the model can't call them even if its
        # prompt would otherwise lead it to. The agent-node handler maps this
        # to ``tools: { read: false, grep: false, ... }`` in session.prompt.
        body["toolsEnabled"] = bool(tools_enabled)
        async with httpx.AsyncClient(timeout=timeout_seconds + 45) as c:
            r = await c.post(url, json=body)
        if r.status_code >= 400:
            raise RuntimeError(f"agent-node opencode /run -> {r.status_code}: {r.text[:800]}")
        data = r.json()
        if data.get("ok") is False:
            raise RuntimeError(f"agent-node opencode failed: {data.get('error')}")
        return data


def _metrics_from(out: dict[str, Any], start: float) -> AdapterMetrics:
    extra: dict[str, Any] = {"agent_node": True, "opencode_mode": True}
    if out.get("structuredOutputFailed"):
        extra["structured_output_failed"] = True
    if out.get("toolNames"):
        extra["opencode_tool_names"] = list(out["toolNames"])
    # Surface the files the agent actually read so the context_recall scorer
    # can compute recall against each case's labeled ``expected_files``. The
    # opencode adapter retrieves on-demand via tools, not via a prefetch
    # block — without this, recall would be uninformative for it.
    paths = out.get("groundingPaths")
    if isinstance(paths, list):
        extra["grounding_paths"] = sorted(p for p in paths if isinstance(p, str))
    tc = out.get("toolCalls")
    return AdapterMetrics(
        duration_ms=int((time.monotonic() - start) * 1000),
        tokens_in=out.get("tokensIn"),
        tokens_out=out.get("tokensOut"),
        cost_usd=out.get("costUsd"),
        model=out.get("model"),
        tool_calls=int(tc) if isinstance(tc, int) else 0,
        extra=extra,
    )
