"""Local code Q&A using our own agentic loop.

Same tool inventory as `deep_decompose` (read_file, grep, list_directory,
find_symbol, get_file_overview, find_referencing_symbols, who_imports,
what_does_this_import) — but the output schema is `Answer`, not
`Decomposition`. The model produces a free-form prose answer with explicit
citations, rather than typed subtasks.

This is the fallback for `--ask` when Sourcebot's `/api/ask` (an EE feature)
isn't available. Once you're on Sourcebot Enterprise, you can switch to the
`SourcebotAskAdapter` in `ask.py` and skip this entirely."""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
from collections import Counter
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import UsageLimits

from ..config import Settings
from ._deep_decompose_impl import DEEP_MAX_FILE_BYTES, DEEP_MAX_GREP_HITS, DEEP_MAX_LIST_ENTRIES, DeepDeps, _call_serena
from ..import_index import load_or_build_index, who_imports as index_who_imports

log = logging.getLogger(__name__)

# Q&A needs more exploration than ticket decomposition — the answer often
# requires verifying 3-5 sources before being grounded.
LOCAL_ASK_MAX_REQUESTS = 30


class Citation(BaseModel):
    repo: str
    path: str
    line_start: int | None = Field(default=None)
    line_end: int | None = Field(default=None)
    snippet: str | None = Field(default=None, description="Short code excerpt that anchors the claim. ~1-5 lines.")


class Answer(BaseModel):
    """Free-form Q&A output. Used by `--ask` when Sourcebot's /api/ask is unavailable."""
    answer: str = Field(description="The actual answer, in Markdown. Be direct — do not produce subtasks.")
    citations: list[Citation] = Field(
        default_factory=list,
        description="Every factual claim about the codebase should have a citation here.",
    )
    confidence: str = Field(
        default="medium",
        description="low / medium / high. low if the answer is partial or you couldn't ground key claims.",
    )
    open_questions: list[str] = Field(
        default_factory=list,
        description="Things you couldn't determine from the available tools.",
    )


_SYSTEM = """\
You answer questions about the Tract codebase. You have tools that let you
read code from four repos:
  - traceability  (Python API)
  - frontend  (TypeScript / Next.js)
  - data  (Python; Airflow DAGs under src/composer/dag/)
  - data-cloud-functions  (Python; CFs under src/cloud_functions/<name>/)

Your job is to give a DIRECT answer to the user's question, grounded in the
code. Do NOT produce a project plan or task breakdown. Produce a Markdown
answer plus citations for every claim.

EXPLORATION STRATEGY:
1. Identify what the user is asking. If they name files/symbols, read those
   first via `read_file` (use `what_does_this_import` to see deps).
2. For "where is X defined?", use `find_symbol`. For "who calls X?", use
   `find_referencing_symbols` or `who_imports`. For "is this string in the
   code?", use `grep`.
3. Read enough to give a precise answer. Typical Q&A: 5-15 tool calls.

GROUNDING — HARD RULES:
- Before claiming a function/class/constant DOES NOT EXIST, you MUST call
  `find_symbol(name_path_pattern=<name>)`. Saying "X is missing" without
  find_symbol evidence is a hard violation.
- Every factual claim about the codebase MUST be backed by a Citation with
  a real `repo`, `path`, and `line_start` from a tool result. Do not invent
  paths or line numbers.
- If the answer has multiple valid sources or conflicting rules, say so
  explicitly — list each variant with its own citation.
- If you can't fully answer, set confidence="low" and put the gaps in
  `open_questions`.

OUTPUT SHAPE:
- `answer`: Markdown prose. Start with the direct answer to the question.
  Then explain. Use code spans / blocks where appropriate.
- `citations`: one per factual claim about the codebase, with a short
  snippet quoting the line(s) you're citing.
- Be honest about uncertainty — partial answers are better than fabricated
  ones.
"""


def _build_agent(settings: Settings) -> Agent[DeepDeps, Answer]:
    if settings.gemini_api_key:
        os.environ["GEMINI_API_KEY"] = settings.gemini_api_key
    model = GeminiModel(settings.decompose_model)
    agent = Agent(
        model=model,
        deps_type=DeepDeps,
        output_type=Answer,
        system_prompt=_SYSTEM,
        model_settings=ModelSettings(temperature=0.0),
    )

    # Identical tool set to deep_decompose. We could share via a helper, but
    # duplicating keeps this module standalone and easy to tune separately.

    @agent.tool
    async def read_file(ctx: RunContext[DeepDeps], repo: str, path: str) -> str:
        """Read a file from one of the configured repos. Path is repo-relative."""
        ctx.deps.tool_calls["read_file"] += 1
        root = ctx.deps.resolve_repo(repo)
        if root is None:
            return f"error: repo {repo!r} not configured"
        target = (root / path).resolve()
        try:
            target.relative_to(root.resolve())
        except ValueError:
            return f"error: path {path!r} escapes repo root"
        if not target.is_file():
            return f"error: file does not exist: {repo}/{path}"
        data = target.read_bytes()[:DEEP_MAX_FILE_BYTES].decode("utf-8", errors="replace")
        lines = data.splitlines()
        return "\n".join(f"{i+1:5d} | {line}" for i, line in enumerate(lines))

    @agent.tool
    async def grep(ctx: RunContext[DeepDeps], pattern: str, repo: str | None = None) -> str:
        """Search for a literal pattern across one or all repos."""
        ctx.deps.tool_calls["grep"] += 1
        if not shutil.which("rg"):
            return "error: ripgrep not installed"
        repos = [repo] if repo else list(ctx.deps.settings.repos)
        out: list[str] = []
        for r in repos:
            root = ctx.deps.resolve_repo(r)
            if root is None:
                continue
            cmd = [
                "rg", "--no-heading", "--smart-case", "--max-count", "5",
                "--max-columns", "200",
                "-g", "!{node_modules,dist,build,.next,.git,*.lock,*.svg,*.png,*.jpg}",
                "-e", pattern, str(root),
            ]
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await proc.communicate()
            for line in stdout.decode(errors="replace").splitlines():
                if not line:
                    continue
                try:
                    abs_path, lineno, rest = line.split(":", 2)
                except ValueError:
                    continue
                try:
                    rel = str(Path(abs_path).relative_to(root))
                except ValueError:
                    rel = abs_path
                out.append(f"{r}:{rel}:{lineno}: {rest.strip()}")
                if len(out) >= DEEP_MAX_GREP_HITS:
                    break
            if len(out) >= DEEP_MAX_GREP_HITS:
                break
        return "\n".join(out) if out else f"(no matches for {pattern!r})"

    @agent.tool
    async def list_directory(ctx: RunContext[DeepDeps], repo: str, path: str = "") -> str:
        """List files and subdirectories at `<repo>/<path>`."""
        ctx.deps.tool_calls["list_directory"] += 1
        root = ctx.deps.resolve_repo(repo)
        if root is None:
            return f"error: repo {repo!r} not configured"
        target = (root / path).resolve() if path else root
        try:
            target.relative_to(root.resolve())
        except ValueError:
            return f"error: path {path!r} escapes repo root"
        if not target.is_dir():
            return f"error: not a directory: {repo}/{path}"
        excluded = {".git", "node_modules", ".venv", "venv", "__pycache__", "dist", "build", ".next"}
        entries: list[str] = []
        for entry in sorted(target.iterdir()):
            if entry.name in excluded:
                continue
            suffix = "/" if entry.is_dir() else ""
            entries.append(entry.name + suffix)
            if len(entries) >= DEEP_MAX_LIST_ENTRIES:
                entries.append(f"... (truncated; >{DEEP_MAX_LIST_ENTRIES} entries)")
                break
        return "\n".join(entries) if entries else "(empty)"

    @agent.tool
    async def find_symbol(ctx: RunContext[DeepDeps], name_path_pattern: str) -> str:
        """Find a code symbol by name via Serena LSP. Use for precise lookups."""
        ctx.deps.tool_calls["find_symbol"] += 1
        result = await _call_serena(
            ctx.deps.settings, "find_symbol",
            {
                "name_path_pattern": name_path_pattern,
                "include_body": False,
                "substring_matching": False,
                "max_matches": 10,
                "max_answer_chars": 4000,
            },
        )
        return result or "(no results from serena)"

    @agent.tool
    async def get_file_overview(ctx: RunContext[DeepDeps], repo: str, path: str) -> str:
        """Structural outline of a file (classes, methods) via Serena."""
        ctx.deps.tool_calls["get_file_overview"] += 1
        rel = f"{repo}/{path}"
        result = await _call_serena(
            ctx.deps.settings, "get_symbols_overview",
            {"relative_path": rel, "depth": 1, "max_answer_chars": 4000},
        )
        return result or "(no overview from serena)"

    @agent.tool
    async def find_referencing_symbols(ctx: RunContext[DeepDeps], name_path: str, relative_path: str) -> str:
        """Find all places that reference a given symbol (Serena)."""
        ctx.deps.tool_calls["find_referencing_symbols"] += 1
        result = await _call_serena(
            ctx.deps.settings, "find_referencing_symbols",
            {"name_path": name_path, "relative_path": relative_path, "max_answer_chars": 4000},
        )
        return result or "(no references found)"

    @agent.tool
    async def what_does_this_import(ctx: RunContext[DeepDeps], repo: str, path: str) -> str:
        """Return the imports a file declares (instant lookup, no LLM)."""
        ctx.deps.tool_calls["what_does_this_import"] += 1
        if repo not in ctx.deps.settings.repos:
            return f"error: repo {repo!r} not configured"
        idx = load_or_build_index(repo, ctx.deps.settings)
        entry = idx.files.get(path)
        if entry is None:
            return f"(no index entry for {repo}/{path})"
        return "\n".join(entry.imports) if entry.imports else "(no imports)"

    @agent.tool
    async def who_imports_tool(ctx: RunContext[DeepDeps], repo: str, path: str) -> str:
        """Return the list of files in <repo> that import <path>."""
        ctx.deps.tool_calls["who_imports"] += 1
        if repo not in ctx.deps.settings.repos:
            return f"error: repo {repo!r} not configured"
        idx = load_or_build_index(repo, ctx.deps.settings)
        callers = index_who_imports(idx, path)
        return "\n".join(callers) if callers else f"(nobody in {repo} imports {path})"

    return agent


async def local_ask(question: str, settings: Settings):
    """Run the local agentic Q&A. Returns the Pydantic AI result + DeepDeps.

    If the request budget is exhausted before the model produces a final
    Answer, returns a synthesized partial answer with confidence=low rather
    than crashing — better UX than a stack trace."""
    from pydantic_ai.exceptions import UsageLimitExceeded

    deps = DeepDeps(settings)
    agent = _build_agent(settings)

    user = (
        f"Question:\n{question.strip()}\n\n"
        f"Explore as needed (read files, follow imports, grep, etc.) and then "
        f"produce the structured Answer. Be direct — answer the question first, "
        f"then explain."
    )

    try:
        result = await agent.run(
            user, deps=deps,
            usage_limits=UsageLimits(request_limit=LOCAL_ASK_MAX_REQUESTS),
        )
        return result, deps
    except UsageLimitExceeded:
        log.warning("local_ask exhausted request budget; returning partial answer")
        # Fabricate a placeholder Answer noting the budget exhaustion. Callers
        # detect this via confidence="low" and the message in `answer`.
        tools_str = ", ".join(f"{k}={v}" for k, v in sorted(deps.tool_calls.items()))
        placeholder = type("PR", (), {
            "output": Answer(
                answer=(
                    f"_(Exploration exceeded the {LOCAL_ASK_MAX_REQUESTS}-step budget without "
                    f"producing a grounded answer. The agent made {tools_str}. Increase "
                    f"`LOCAL_ASK_MAX_REQUESTS` or rephrase the question to be more specific.)_"
                ),
                citations=[], confidence="low",
                open_questions=["The original question, unresolved."],
            ),
            "usage": lambda *a, **kw: None,
        })()
        return placeholder, deps
