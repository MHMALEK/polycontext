"""LangChain adapter — an in-process agent over the shared workspace tools.

Why this exists
---------------
Every other agentic adapter either round-trips through ``services/agent-node``
(cursor, cline_sdk, claude_code, gemini, openai_agents, opencode) or talks to a
remote service (sourcebot). This one is pure-Python and runs the whole
tool-calling loop inside the FastAPI process, the same way ``_pipeline.py`` does
its synthesis — but using **LangChain** instead of pydantic-ai as the agent
framework.

It's primarily a reference / learning implementation: it shows that the
project's adapter contract is framework-agnostic. The *only* thing an adapter
owes the rest of the system is:

  - ``ask``                → return an ``AdapterAskResult`` (answer + metrics)
  - ``_decompose_raw_text`` → return raw model text; the shared structurer in
    ``core/decomposition_structurer.py`` turns it into a validated
    ``Decomposition`` (we never parse JSON here).

How it maps onto LangChain (the three moving parts)
---------------------------------------------------
1. **Tools.** We wrap the four read-only workspace helpers from
   ``core/agent_tools.py`` (``read_file``, ``list_directory``, ``glob``,
   ``grep_search``) as LangChain tools with the ``@tool`` decorator. The
   workspace root is captured in a closure, so the model only ever passes
   repo-relative paths — identical safety model to the pydantic-ai tools.

2. **Model.** ``init_chat_model`` gives us one provider-agnostic constructor.
   We parse the project-standard ``provider:model`` spec (reusing
   ``ModelSpec`` from ``core/llm_registry.py``) and translate it to the
   langchain provider id + the right credential. OpenRouter is just the
   OpenAI provider pointed at OpenRouter's base URL — same trick the registry
   uses for pydantic-ai.

3. **Agent.** ``create_agent`` (langchain v1's prebuilt ReAct-style agent,
   built on langgraph) wires the model + tools + system prompt into a graph.
   ``agent.ainvoke({"messages": [...]})`` runs the full think→call-tool→observe
   loop and returns the message list; the final ``AIMessage`` is the answer and
   every ``AIMessage`` carries ``usage_metadata`` for token accounting.

Docs: https://docs.langchain.com/oss/python/langchain/quickstart
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ..core.agent_tools import (
    DEFAULT_MAX_BYTES,
    GREP_MAX_MATCHES,
    Workspace,
    _glob,
    _grep_search,
    _list_directory,
    _read_file,
)
from ..core.explore_directive import wrap_with_explore_directive
from ..core.llm_registry import ModelSpec, estimate_cost_usd
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
    "You are a code Q&A assistant with read-only access to local repositories "
    "via the read_file, grep_search, glob, and list_directory tools.\n\n"
    "MANDATORY: Before answering ANY question about the codebase, call at least "
    "one tool to verify your claims against the actual files. Do NOT answer from "
    "training knowledge alone — answers that invent file paths, class names, or "
    "behavior are unacceptable.\n\n"
    "Workflow:\n"
    "1. Use glob/grep_search to locate relevant files.\n"
    "2. read_file 2-3 of them to confirm.\n"
    "3. Write the final answer citing exact paths and line numbers.\n\n"
    "Paths are relative to the workspace root. Return only the final answer."
)

_DECOMPOSE_SYSTEM = (
    "You decompose engineering tickets into structured tech work. You have "
    "read-only filesystem tools (read_file, list_directory, glob, grep_search) "
    "over a workspace containing multiple repos.\n\n"
    "Map the workspace first (list_directory '.'), find where the relevant code "
    "lives, and read the files you intend to cite — never invent paths. Then emit "
    "exactly one JSON object matching the schema described in the user message: "
    "no prose, no markdown fences."
)


class LangChainAdapter(Adapter):
    """In-process agent backed by LangChain's ``create_agent``."""

    name = "langchain"
    capabilities: set[Capability] = {"ask", "decompose"}
    description = (
        "In-process LangChain agent (create_agent + init_chat_model) over the "
        "shared read-only workspace tools. No agent-node round-trip; model is the "
        "project-standard ``provider:model`` spec (gemini/anthropic/openai/openrouter)."
    )

    # ----- readiness ---------------------------------------------------------

    def health(self) -> dict:
        try:
            import langchain  # noqa: F401
            from langchain.agents import create_agent  # noqa: F401
        except ImportError as e:
            return {"ok": False, "reason": f"langchain not installed: {e} (pip install -e '.[langchain]')"}
        spec = (self.settings.langchain_model or "").strip()
        if not spec:
            return {"ok": False, "reason": "LANGCHAIN_MODEL not set"}
        try:
            ms = ModelSpec.parse(spec)
        except ValueError as e:
            return {"ok": False, "reason": f"bad LANGCHAIN_MODEL {spec!r}: {e}"}
        missing = self._missing_key_for(ms.provider)
        if missing:
            return {"ok": False, "reason": f"{missing} required for default model {spec!r}"}
        return {"ok": True}

    def _missing_key_for(self, provider: str) -> str | None:
        """Return the name of the missing credential for ``provider``, or None."""
        needs = {
            "gemini": ("gemini_api_key", "GEMINI_API_KEY"),
            "anthropic": ("anthropic_api_key", "ANTHROPIC_API_KEY"),
            "openai": ("openai_api_key", "OPENAI_API_KEY"),
            "openrouter": ("openrouter_api_key", "OPENROUTER_API_KEY"),
        }
        pair = needs.get(provider)
        if not pair:
            return None
        attr, env_name = pair
        return None if getattr(self.settings, attr, "") else env_name

    # ----- model construction ------------------------------------------------

    def _make_model(self, spec: str | None):
        """Translate a ``provider:model`` spec into a langchain chat model.

        We reuse ``ModelSpec`` (the same parser the pydantic-ai registry uses)
        so the spec syntax is identical across adapters, then hand off to
        ``init_chat_model`` with the provider-specific credential. OpenRouter
        and any custom OpenAI-compatible endpoint ride the OpenAI provider with
        a ``base_url`` override — exactly how ``core/llm_registry.get_model``
        does it for pydantic-ai.
        """
        from langchain.chat_models import init_chat_model

        ms = ModelSpec.parse((spec or self.settings.langchain_model))
        timeout = float(self.settings.langchain_timeout_seconds)

        if ms.provider == "gemini":
            if not self.settings.gemini_api_key:
                raise RuntimeError("GEMINI_API_KEY required for provider 'gemini'")
            return init_chat_model(
                ms.name,
                model_provider="google_genai",
                google_api_key=self.settings.gemini_api_key,
                timeout=timeout,
            )
        if ms.provider == "anthropic":
            if not self.settings.anthropic_api_key:
                raise RuntimeError("ANTHROPIC_API_KEY required for provider 'anthropic'")
            return init_chat_model(
                ms.name,
                model_provider="anthropic",
                api_key=self.settings.anthropic_api_key,
                timeout=timeout,
            )
        if ms.provider == "openai":
            if not self.settings.openai_api_key:
                raise RuntimeError("OPENAI_API_KEY required for provider 'openai'")
            return init_chat_model(
                ms.name,
                model_provider="openai",
                api_key=self.settings.openai_api_key,
                timeout=timeout,
            )
        if ms.provider == "openrouter":
            if not self.settings.openrouter_api_key:
                raise RuntimeError("OPENROUTER_API_KEY required for provider 'openrouter'")
            return init_chat_model(
                ms.name,
                model_provider="openai",
                api_key=self.settings.openrouter_api_key,
                base_url="https://openrouter.ai/api/v1",
                timeout=timeout,
            )
        if ms.provider in ("custom", "openai-compat"):
            base_url = self.settings.custom_llm_base_url
            if not base_url:
                raise RuntimeError("CUSTOM_LLM_BASE_URL required for provider 'custom'")
            return init_chat_model(
                ms.name,
                model_provider="openai",
                api_key=self.settings.custom_llm_api_key or "sk-noop",
                base_url=base_url,
                timeout=timeout,
            )
        raise RuntimeError(f"unknown provider {ms.provider!r} in LANGCHAIN_MODEL")

    def _build_tools(self, ws: Workspace) -> list:
        """Wrap the shared workspace helpers as LangChain tools.

        ``@tool`` reads the function name + docstring + type hints to build the
        tool schema the model sees. The ``ws`` root is closed over, so the model
        only ever supplies repo-relative paths — ``core/agent_tools`` clamps them
        inside the workspace.
        """
        from langchain_core.tools import tool

        @tool
        async def read_file(path: str, max_bytes: int = DEFAULT_MAX_BYTES) -> str:
            """Read a UTF-8 text file under the workspace.

            Args:
                path: File path relative to the workspace root (e.g. 'src/foo.py').
                max_bytes: Optional cap on bytes returned (default 50000).
            """
            return await _read_file(ws, path, max_bytes)

        @tool
        async def list_directory(path: str = ".") -> str:
            """List non-hidden files and folders in a directory under the workspace.

            Args:
                path: Directory path relative to the workspace root; '.' for root.
            """
            return await _list_directory(ws, path)

        @tool
        async def glob(pattern: str) -> str:
            """Find files whose path contains the given case-insensitive substring.
            Skips build/dependency directories. Returns repo-relative paths.

            Args:
                pattern: Substring to match within file paths (e.g. 'tests/auth').
            """
            return await _glob(ws, pattern)

        @tool
        async def grep_search(query: str, max_matches: int = GREP_MAX_MATCHES) -> str:
            """Search file contents for a literal string or regex pattern.
            Returns 'path:line:content' rows.

            Args:
                query: Literal string or regex pattern.
                max_matches: Optional cap on matches returned (default 60).
            """
            return await _grep_search(ws, query, max_matches)

        return [read_file, list_directory, glob, grep_search]

    def _cwd_for_repos(self, repos: list[str] | None) -> Path:
        if repos and len(repos) == 1:
            return self.settings.repo_path(repos[0])
        return Path(self.settings.repos_root)

    async def _run_agent(
        self,
        *,
        system: str,
        prompt: str,
        model_spec: str | None,
        repos: list[str] | None,
    ) -> dict[str, Any]:
        """Build a one-shot agent, run the tool-calling loop, return raw output.

        Returns a dict the ask/decompose paths share: ``answer``, ``model``,
        ``tokens_in/out``, ``tool_calls``, ``tool_trace``.
        """
        from langchain.agents import create_agent

        ws = Workspace(root=self._cwd_for_repos(repos).resolve())
        model = self._make_model(model_spec)
        tools = self._build_tools(ws)
        agent = create_agent(model, tools=tools, system_prompt=system)

        # create_agent counts each LLM step AND each tool step against the
        # recursion limit. ~2 nodes per round + a final answer node.
        rounds = max(1, int(self.settings.langchain_max_tool_rounds))
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": prompt}]},
            config={"recursion_limit": 2 * rounds + 1},
        )
        messages = result.get("messages", []) if isinstance(result, dict) else []
        return self._collect(messages)

    @staticmethod
    def _collect(messages: list) -> dict[str, Any]:
        """Pull the answer + telemetry out of the langgraph message list."""
        answer = ""
        tokens_in = 0
        tokens_out = 0
        tool_calls = 0
        tool_trace: list[dict[str, Any]] = []
        model_name: str | None = None

        for m in messages:
            usage = getattr(m, "usage_metadata", None)
            if isinstance(usage, dict):
                tokens_in += int(usage.get("input_tokens") or 0)
                tokens_out += int(usage.get("output_tokens") or 0)
            meta = getattr(m, "response_metadata", None)
            if isinstance(meta, dict) and not model_name:
                model_name = meta.get("model_name") or meta.get("model")
            for call in getattr(m, "tool_calls", None) or []:
                tool_calls += 1
                if isinstance(call, dict):
                    tool_trace.append({"name": call.get("name"), "args": call.get("args")})

        # The final message is the agent's answer (an AIMessage with no further
        # tool calls). Concatenate its text content robustly — content may be a
        # plain string or a list of content blocks.
        if messages:
            answer = _message_text(messages[-1])

        return {
            "answer": answer.strip(),
            "model": model_name,
            "tokens_in": tokens_in or None,
            "tokens_out": tokens_out or None,
            "tool_calls": tool_calls,
            "tool_trace": tool_trace,
        }

    def _metrics(self, out: dict[str, Any], start: float, spec: str) -> AdapterMetrics:
        tin = out.get("tokens_in")
        tout = out.get("tokens_out")
        cost = None
        if isinstance(tin, int) and isinstance(tout, int):
            cost = estimate_cost_usd(spec, tin, tout)
        return AdapterMetrics(
            duration_ms=int((time.monotonic() - start) * 1000),
            tokens_in=tin,
            tokens_out=tout,
            cost_usd=cost,
            model=out.get("model"),
            tool_calls=int(out.get("tool_calls") or 0),
            extra={
                "framework": "langchain",
                "in_process": True,
                "model_spec": spec,
                "tool_trace": out.get("tool_trace") or [],
            },
        )

    # ----- the two operations ------------------------------------------------

    async def ask(self, inp: AdapterAskInput) -> AdapterAskResult:
        t = time.monotonic()
        spec = (inp.model or self.settings.langchain_model).strip()
        # Mirror the gemini/opencode paths: the "must-explore" nudge goes INTO
        # the user message (cheap models ignore swapped system prompts) — but
        # only when tools are on and grounding hasn't already prepended snippets.
        prompt = wrap_with_explore_directive(
            inp.query, tools_on=inp.tools_enabled and not inp.grounded,
        )
        out = await self._run_agent(
            system=_ASK_SYSTEM, prompt=prompt, model_spec=spec, repos=inp.repos,
        )
        return AdapterAskResult(
            adapter=self.name,
            answer=out["answer"],
            citations=[],
            metrics=self._metrics(out, t, spec),
        )

    def _decompose_spec(self) -> str:
        d = (self.settings.langchain_decompose_model or "").strip()
        return d or self.settings.langchain_model

    async def _decompose_raw_text(self, inp: AdapterDecomposeInput) -> RawDecomposeText:
        t = time.monotonic()
        spec = self._decompose_spec()
        prompt = DECOMPOSE_PREAMBLE.format(model_tag=self.name) + query_blob(inp)
        out = await self._run_agent(
            system=_DECOMPOSE_SYSTEM, prompt=prompt, model_spec=spec, repos=inp.repos,
        )
        return RawDecomposeText(text=out["answer"], metrics=self._metrics(out, t, spec))


def _message_text(msg: Any) -> str:
    """Extract plain text from a langchain message whose ``content`` may be a
    string or a list of content blocks (dicts with a ``text`` key, or objects
    exposing ``.text``)."""
    content = getattr(msg, "content", msg)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
            elif isinstance(getattr(block, "text", None), str):
                parts.append(block.text)
        return "".join(parts)
    return str(content)
