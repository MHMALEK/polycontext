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

def _strip_gemini_prefix(spec: str | None) -> str | None:
    """Accept the same per-request model id formats other adapters use.

    The gemini adapter (via @google/genai) wants a bare model id like
    ``gemini-2.5-flash`` or ``gemini-2.5-pro``. Strip provider prefixes
    so the UI's Model picker can send ``gemini:gemini-2.5-pro``,
    ``google/gemini-2.5-flash``, or the bare id — all work."""
    if not spec:
        return None
    s = spec.strip()
    if not s:
        return None
    if s.startswith("gemini:"):
        return s.split(":", 1)[1].strip() or None
    if s.startswith("google/"):
        return s.split("/", 1)[1].strip() or None
    return s


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
_DECOMPOSE_SYSTEM = """\
You decompose engineering tickets into structured tech work for a real
developer to pick up. You have read-only filesystem tools (read_file,
list_directory, search_files, grep_search) over a workspace on disk that
contains multiple repos.

# Output contract

The user message describes the output JSON schema. Your final response
must be exactly one JSON object matching it — no prose, no markdown
fences. Every file path in the output must be one you actually opened
or saw in a search/list result. DO NOT invent paths.

# Required workflow (read this carefully)

You are not done until you have done all of these. Skipping steps
produces decompositions that mis-place code in the wrong repo or cite
files that don't exist — both of which waste a developer's afternoon.

## Step 1 — Map the workspace

Call list_directory(".") FIRST. Look at every top-level repo name.

For a "move / port / create new code" ticket, every repo name with a
serverless/microservice flavor is a candidate destination — names like
`*-cloud-functions`, `*-cf`, `*-functions`, `*-lambdas`, `services/*`,
`*-workers`, `*-jobs`. Note all of them.

## Step 2 — Find the SOURCE (where the existing code is)

search_files / grep_search for the class, function, or DAG name the
ticket mentions. read_file on the hits. Note the actual classes you
find and the modules they live in.

## Step 3 — Find the DESTINATION (where the new code goes)

DO NOT skip this step. Domain repos (e.g. `traceability/`) are usually
NOT where new Cloud Functions belong — even when the ticket talks about
"traceability validation", new CF code typically goes in a dedicated
CF repo.

For each candidate destination repo from Step 1:
  a. list_directory(<repo>/src) (or wherever the source code lives).
  b. If you see existing entrypoints / cloud function modules, read
     their main.py. Look for stub functions: `def process_X(...): pass`,
     `TODO`, `DEV-XXXX`, `raise NotImplementedError`. These are explicit
     extension points that the new code should slot into.
  c. Read at least one EXISTING sibling implementation to understand
     the pattern to mirror.

## Step 4 — Emit the JSON

When citing files, use the FULL path relative to the workspace root
(e.g. `data-cloud-functions/src/cloud_functions/composer_dag_trigger/main.py`,
NOT bare `main.py`). When you name a class or function, name the one
you actually saw in a read_file output.

# Worked example (study this — it shows what good exploration looks like)

User ticket: "Move the foo-bar batch job out of Airflow and into a Cloud
Function".

Good exploration sequence:
  1. list_directory(".") → sees `data/`, `data-cloud-functions/`, `frontend/`, `services/`
  2. search_files("foo_bar") → finds `data/src/dags/foo_bar_job.py`
  3. read_file("data/src/dags/foo_bar_job.py") → confirms it's the source DAG
  4. list_directory("data-cloud-functions/src") → sees `cloud_functions/`
  5. list_directory("data-cloud-functions/src/cloud_functions") → sees several existing CFs and a `composer_dag_trigger/`
  6. read_file("data-cloud-functions/src/cloud_functions/composer_dag_trigger/main.py")
     → finds `def process_foo_bar(...): pass` near the bottom — THE EXTENSION POINT
  7. Emit Decomposition citing `data/src/dags/foo_bar_job.py` (source) AND
     `data-cloud-functions/src/cloud_functions/composer_dag_trigger/main.py`
     (destination — replace the `process_foo_bar` stub).

Bad exploration (DO NOT do this):
  1. list_directory(".") → sees the repos
  2. search_files / grep_search the source
  3. read_file the source
  4. ASSUMES the new code goes in a repo named after the domain
  5. Emits Decomposition citing fabricated path like `<domain-repo>/src/main.py`

If you find yourself about to emit JSON without having read any file
inside the destination repo's `src/` directory, STOP and do Step 3.
"""


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
        # Honor per-request model override (from UI's Model picker or API
        # caller's inp.model). Falls through to GEMINI_SDK_MODEL otherwise.
        # The gemini adapter accepts bare Gemini ids (``gemini-2.5-flash``)
        # — strip ``gemini:`` or ``google/`` provider prefixes if present.
        override = _strip_gemini_prefix(inp.model)
        return await self.ask_configured(
            inp, model_id=override, tools_enabled=inp.tools_enabled,
        )

    async def astream(self, inp: AdapterAskInput):
        """Proxy the SSE bridge in services/agent-node /adapters/gemini/stream.

        Yields the normalized event union (kind: session/status/text.delta/
        tool.update/error/done). Same shape as opencode.astream so the UI's
        client reducer handles both transparently. We don't open a separate
        websocket; httpx streams the chunked SSE response.
        """
        import json
        from ..core.explore_directive import wrap_with_explore_directive
        override = _strip_gemini_prefix(inp.model) or self.settings.gemini_sdk_model
        prompt = wrap_with_explore_directive(
            inp.query, tools_on=inp.tools_enabled and not inp.grounded,
        )
        body: dict[str, Any] = {
            "systemPrompt": _ASK_SYSTEM,
            "prompt": prompt,
            "apiKey": self.settings.gemini_api_key,
            "modelId": override,
            "timeoutSec": int(self.settings.gemini_sdk_timeout_seconds),
            "cwd": str(self._cwd_for_repos(inp.repos)),
            "toolsEnabled": bool(inp.tools_enabled),
        }
        url = self.settings.agent_node_url.rstrip("/") + "/adapters/gemini/stream"
        timeout = httpx.Timeout(connect=30.0, read=None, write=30.0, pool=None)
        async with httpx.AsyncClient(timeout=timeout) as c:
            async with c.stream("POST", url, json=body) as resp:
                if resp.status_code >= 400:
                    text = (await resp.aread()).decode("utf-8", "replace")[:500]
                    raise RuntimeError(
                        f"agent-node gemini /stream -> {resp.status_code}: {text}"
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
                    # Stop reading on the terminal ``done`` event. Continuing
                    # to ``aiter_lines`` after this would race against
                    # agent-node closing its chunked-encoding response and
                    # raise RemoteProtocolError on a half-closed read.
                    if isinstance(ev, dict) and ev.get("kind") == "done":
                        return

    async def ask_configured(
        self,
        inp: AdapterAskInput,
        *,
        model_id: str | None = None,
        max_tool_rounds: int | None = None,
        tools_enabled: bool = True,
    ) -> AdapterAskResult:
        from ..core.explore_directive import wrap_with_explore_directive
        t = time.monotonic()
        # Wrap query with the "must-explore" directive when tools are on
        # and grounding wasn't already prepended at the API level. The
        # SDK's system parameter is unreliably honored by cheap models,
        # so we put the directive in the user message where the model
        # can't miss it.
        prompt = wrap_with_explore_directive(
            inp.query, tools_on=tools_enabled and not inp.grounded,
        )
        out = await self._run(
            system=_ASK_SYSTEM,
            prompt=prompt,
            model_id=model_id or self.settings.gemini_sdk_model,
            timeout_seconds=float(self.settings.gemini_sdk_timeout_seconds),
            cwd=self._cwd_for_repos(inp.repos),
            max_tool_rounds=max_tool_rounds,
            tools_enabled=tools_enabled,
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
        return await self.decompose_raw_configured(inp)

    async def decompose_raw_configured(
        self,
        inp: AdapterDecomposeInput,
        *,
        model_id: str | None = None,
        max_tool_rounds: int | None = None,
    ) -> RawDecomposeText:
        t = time.monotonic()
        prompt = DECOMPOSE_PREAMBLE.format(model_tag=self.name) + query_blob(inp)
        out = await self._run(
            system=_DECOMPOSE_SYSTEM,
            prompt=prompt,
            model_id=model_id or self._decompose_model_id(),
            timeout_seconds=float(self.settings.gemini_sdk_timeout_seconds),
            cwd=self._cwd_for_repos(inp.repos),
            response_schema=_gemini_decomposition_schema(),
            max_tool_rounds=max_tool_rounds,
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
        response_schema: dict[str, Any] | None = None,
        max_tool_rounds: int | None = None,
        tools_enabled: bool = True,
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
        if response_schema is not None:
            body["responseSchema"] = response_schema
        if max_tool_rounds is not None:
            body["maxToolRounds"] = max_tool_rounds
        # Forward the per-request tools-mask. agent-node's gemini handler
        # routes to the no-tools generateContent path when this is false.
        if not tools_enabled:
            body["toolsEnabled"] = False
        async with httpx.AsyncClient(timeout=timeout_seconds + 30) as c:
            r = await c.post(url, json=body)
        if r.status_code >= 400:
            raise RuntimeError(f"agent-node gemini /run -> {r.status_code}: {r.text[:500]}")
        data = r.json()
        if data.get("ok") is False:
            raise RuntimeError(f"agent-node gemini failed: {data.get('error')}")
        return data


def _metrics_from(out: dict[str, Any], start: float) -> AdapterMetrics:
    extra: dict[str, Any] = {"sdk": "@google/genai", "agent_node": True}
    if isinstance(out.get("toolTrace"), list):
        # The adapter records every workspace tool call (name, args, result
        # preview, latency). Surfacing it through metrics.extra makes
        # "did gemini actually call list_directory?" answerable without
        # adding more logging — the runstore row already has it.
        extra["tool_trace"] = out["toolTrace"]
    if isinstance(out.get("thoughtsTokens"), int):
        extra["thoughts_tokens"] = out["thoughtsTokens"]
    return AdapterMetrics(
        duration_ms=int((time.monotonic() - start) * 1000),
        tokens_in=_as_int(out.get("tokensIn")),
        tokens_out=_as_int(out.get("tokensOut")),
        model=_as_str(out.get("model")),
        tool_calls=_as_int(out.get("toolCalls")) or 0,
        extra=extra,
    )


def _as_int(v: Any) -> int | None:
    return v if isinstance(v, int) else None


def _as_str(v: Any) -> str | None:
    return v if isinstance(v, str) and v else None


# ---------------------------------------------------------------------------
# Decomposition responseSchema for Gemini
# ---------------------------------------------------------------------------
#
# Gemini's ``responseSchema`` is a subset of JSON Schema — it doesn't accept
# ``$ref`` / ``$defs`` (which pydantic uses heavily for nested models) and is
# strict about properties. Hand-rolling a flat schema is safer than trying to
# inline-resolve pydantic's auto-generated one.

_GEMINI_DECOMPOSITION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "Echo of the user's question."},
        "overview": {
            "type": "string",
            "description": "2–4 sentence engineer-readable framing of what needs to happen.",
        },
        "affected_repos": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Repo names touched by this work (must exist on disk).",
        },
        "risks": {"type": "array", "items": {"type": "string"}},
        "open_questions": {"type": "array", "items": {"type": "string"}},
        "subtasks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "repo": {"type": "string"},
                    "files": {"type": "array", "items": {"type": "string"}},
                    "file_links": {"type": "array", "items": {"type": "string"}},
                    "acceptance_criteria": {"type": "array", "items": {"type": "string"}},
                    "estimated_complexity": {
                        "type": "string",
                        "enum": ["small", "medium", "large", "unknown"],
                    },
                },
                "required": ["title", "description", "repo", "estimated_complexity"],
            },
        },
        "enrichment_model": {"type": "string"},
        "decomposition_model": {"type": "string"},
    },
    "required": ["query", "overview", "affected_repos", "subtasks"],
}


def _gemini_decomposition_schema() -> dict[str, Any]:
    """Return the Gemini-compatible JSON schema for ``Decomposition``."""
    return _GEMINI_DECOMPOSITION_SCHEMA
