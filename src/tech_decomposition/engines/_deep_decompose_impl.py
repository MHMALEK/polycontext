"""Agentic deep decompose.

Same input (Ticket + EnrichedQuery) and same output (Decomposition) as the
cheap pipeline, but the model drives retrieval itself via tools:

- read_file(repo, path)          — full file read
- grep(query, repo?)             — keyword search across repos (ripgrep)
- list_directory(repo, path)     — what's at this path
- find_symbol(name)              — Serena LSP-backed symbol lookup
- get_file_overview(repo, path)  — Serena structural outline (classes, methods)

This is the same pattern Cursor and Claude Code use: a model loop that
reads → thinks → reads more until it can answer. Costs 5-25x cheap mode
because the model takes multiple turns, but quality on hard tickets is
materially better.

Bounded by `request_limit` (max model invocations) — set conservatively
so a runaway exploration can't burn the bill."""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
from collections import Counter
from pathlib import Path

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import UsageLimits

from ..config import Settings
from ..import_index import load_or_build_index, transitive_imports, who_imports
from ..models import Decomposition, EnrichedQuery

log = logging.getLogger(__name__)

# Per-request caps.
DEEP_MAX_REQUESTS = 20            # max model turns; runaway protection
DEEP_MAX_FILE_BYTES = 24_000      # truncate huge files when read
DEEP_MAX_GREP_HITS = 30
DEEP_MAX_LIST_ENTRIES = 60


_SYSTEM = """\
You are a senior Tract engineer producing a tech decomposition for a task.
You have tools that let you read code from four repos:
  - traceability  (Python API)
  - frontend  (TypeScript / Next.js)
  - data  (Python; Airflow DAGs under src/composer/dag/)
  - data-cloud-functions  (Python; CFs under src/cloud_functions/<name>/)

EXPLORATION STRATEGY:
1. Read any file explicitly named in the ticket first.
2. Use `what_does_this_import(repo, path)` on anchor files to discover their
   dependencies. Read the dependencies you need.
3. For "who calls this?" questions use `who_imports(repo, path)` or Serena's
   `find_symbol`. Do not guess.
4. Before deciding a file is too long to read, call `get_file_overview` for
   its structural outline first.
5. Stop when you can produce a grounded decomposition. Typical good decomp:
   8-20 tool calls.

SYMBOL EXISTENCE — HARD RULE:
- Before claiming any function, class, method, or constant DOES NOT EXIST,
  you MUST first call `find_symbol(name_path_pattern=<name>)` to verify.
- If find_symbol returns nothing, only then may you say "does not exist."
- Saying "X is missing" or "X is not defined" without calling find_symbol is
  a hard violation of these rules.

GROUNDING — HARD RULES:
- Every file path you reference in your final output MUST be a path you have
  actually read (read_file) or listed (list_directory) via tools. If you
  can't find a file, mark the subtask "requires investigation" rather than
  guessing a name.
- Do not invent function or class names. Only reference symbols that appear
  in tool output you've seen.
- Do not propose subtasks whose acceptance criteria contradict criteria in
  the source ticket. If a subtask would, drop or reshape it.

TOOL CHOICE GUIDE:
- "Where is X defined?"     → find_symbol (precise) or grep (cheap)
- "Who calls X?"            → find_referencing_symbols or who_imports
- "What does file F use?"   → what_does_this_import
- "What's in file F?"       → get_file_overview before read_file
- "What's at path P?"       → list_directory
- "Is this string anywhere?" → grep

When you're done exploring, produce the final Decomposition.
"""


def _git_head(repo_path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_path, text=True
        ).strip()
    except Exception:
        return "HEAD"


class DeepDeps:
    """RunContext deps for the deep agent."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.tool_calls: Counter = Counter()
        self.head_by_repo: dict[str, str] = {}
        # Lazy serena session — opened on first symbol-tool use.
        self._serena_lock = asyncio.Lock()

    def resolve_repo(self, repo: str) -> Path | None:
        if repo not in self.settings.repos:
            return None
        p = self.settings.repo_path(repo)
        return p if p.is_dir() else None

    def head_sha(self, repo: str) -> str:
        if repo not in self.head_by_repo:
            p = self.resolve_repo(repo)
            self.head_by_repo[repo] = _git_head(p) if p else "HEAD"
        return self.head_by_repo[repo]


def _build_agent(settings: Settings) -> Agent[DeepDeps, Decomposition]:
    if settings.gemini_api_key:
        os.environ["GEMINI_API_KEY"] = settings.gemini_api_key
    model = GeminiModel(settings.decompose_model)
    agent = Agent(
        model=model,
        deps_type=DeepDeps,
        output_type=Decomposition,
        system_prompt=_SYSTEM,
        model_settings=ModelSettings(temperature=0.0),
    )

    @agent.tool
    async def read_file(ctx: RunContext[DeepDeps], repo: str, path: str) -> str:
        """Read a file from one of the configured repos. Path is relative to the repo root.
        Returns the file content (truncated to ~24KB if large)."""
        ctx.deps.tool_calls["read_file"] += 1
        root = ctx.deps.resolve_repo(repo)
        if root is None:
            return f"error: repo {repo!r} not configured (configured: {ctx.deps.settings.repos})"
        target = (root / path).resolve()
        try:
            target.relative_to(root.resolve())
        except ValueError:
            return f"error: path {path!r} escapes repo root"
        if not target.is_file():
            return f"error: file does not exist: {repo}/{path}"
        data = target.read_bytes()[:DEEP_MAX_FILE_BYTES].decode("utf-8", errors="replace")
        # Prefix with line numbers so the model can cite ranges back.
        lines = data.splitlines()
        return "\n".join(f"{i+1:5d} | {line}" for i, line in enumerate(lines))

    @agent.tool
    async def grep(
        ctx: RunContext[DeepDeps], pattern: str, repo: str | None = None
    ) -> str:
        """Search for a literal pattern across one or all repos.
        Returns up to 30 matches as `repo:path:line: snippet`."""
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
                # rg gives absolute paths; rewrite to repo-relative.
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
    async def list_directory(
        ctx: RunContext[DeepDeps], repo: str, path: str = ""
    ) -> str:
        """List files and subdirectories at `<repo>/<path>`. Use path='' for the repo root."""
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
        """Find a code symbol (class, function, method) by name via Serena LSP.
        Use this for precise lookups when you know the symbol name.
        Returns symbol locations across all 4 repos as one flat list."""
        ctx.deps.tool_calls["find_symbol"] += 1
        result = await _call_serena(
            ctx.deps.settings,
            "find_symbol",
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
    async def get_file_overview(
        ctx: RunContext[DeepDeps], repo: str, path: str
    ) -> str:
        """Get a structural overview of a file (top-level classes, functions) via Serena.
        Much cheaper than read_file for getting the shape of a large file before
        deciding whether to read it in full."""
        ctx.deps.tool_calls["get_file_overview"] += 1
        rel = f"{repo}/{path}"
        result = await _call_serena(
            ctx.deps.settings,
            "get_symbols_overview",
            {"relative_path": rel, "depth": 1, "max_answer_chars": 4000},
        )
        return result or "(no overview from serena)"

    @agent.tool
    async def find_referencing_symbols(
        ctx: RunContext[DeepDeps], name_path: str, relative_path: str
    ) -> str:
        """Find all places that reference a given symbol. Use this for 'who calls X'
        questions. `name_path` is the symbol name (or Class/method path);
        `relative_path` is the file the symbol is defined in (repo/path/to/file)."""
        ctx.deps.tool_calls["find_referencing_symbols"] += 1
        result = await _call_serena(
            ctx.deps.settings,
            "find_referencing_symbols",
            {
                "name_path": name_path,
                "relative_path": relative_path,
                "max_answer_chars": 4000,
            },
        )
        return result or "(no references found)"

    @agent.tool
    async def what_does_this_import(
        ctx: RunContext[DeepDeps], repo: str, path: str
    ) -> str:
        """Return the list of repo-relative paths this file imports (from the
        pre-built import index). Instant. Use this BEFORE read_file when you
        want to know dependencies without reading the file. Returns empty
        list if the file isn't in the index (binary, or outside source dirs)."""
        ctx.deps.tool_calls["what_does_this_import"] += 1
        if repo not in ctx.deps.settings.repos:
            return f"error: repo {repo!r} not configured"
        idx = load_or_build_index(repo, ctx.deps.settings)
        entry = idx.files.get(path)
        if entry is None:
            return f"(no index entry for {repo}/{path} — may be binary or outside source dirs)"
        if not entry.imports:
            return "(no imports)"
        return "\n".join(entry.imports)

    @agent.tool
    async def who_imports_tool(
        ctx: RunContext[DeepDeps], repo: str, path: str
    ) -> str:
        """Return the list of files in <repo> that import <path>. Instant lookup
        in the pre-built reverse-import index. Cheaper and broader than
        find_referencing_symbols when you just want 'who depends on this file'."""
        ctx.deps.tool_calls["who_imports"] += 1
        if repo not in ctx.deps.settings.repos:
            return f"error: repo {repo!r} not configured"
        idx = load_or_build_index(repo, ctx.deps.settings)
        callers = who_imports(idx, path)
        if not callers:
            return f"(no files import {path} — either it's a leaf, or path is wrong; double-check with list_directory)"
        return "\n".join(callers)

    return agent


async def _call_serena(settings: Settings, tool_name: str, args: dict) -> str:
    """One-shot Serena MCP call: open SSE, activate_project, call tool, close."""
    if not settings.serena_url:
        return "(serena not configured)"
    try:
        from mcp import ClientSession
        from mcp.client.sse import sse_client
    except ImportError:
        return "(mcp library not installed)"

    try:
        async with sse_client(settings.serena_url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                await session.call_tool(
                    "activate_project",
                    arguments={"project": "/workspaces/projects"},
                )
                result = await session.call_tool(tool_name, arguments=args)
                parts = []
                for item in (result.content or []):
                    text = getattr(item, "text", None)
                    if text:
                        parts.append(text)
                return "\n".join(parts)[:8000]
    except Exception as e:
        log.warning("serena call %s failed: %s", tool_name, e)
        return f"(serena error: {e})"


async def deep_decompose(
    *,
    query_text: str,
    enriched: EnrichedQuery,
    settings: Settings,
):
    """Run the agentic deep decompose. Returns the Pydantic AI result object
    plus the DeepDeps so the caller can read tool_calls + iteration count."""
    deps = DeepDeps(settings)
    agent = _build_agent(settings)

    user = (
        f"Ticket key: {ticket.key or '(none)'}\n"
        f"Ticket title: {ticket.title}\n"
        f"Ticket URL: {ticket.url or '(none)'}\n\n"
        f"Ticket body:\n{ticket.body or '(empty)'}\n\n"
        f"Enrichment summary: {enriched.summary}\n"
        f"Suspected repos (Flash's guess, verify with tools): "
        f"{', '.join(enriched.suspected_repos) or '(none)'}\n"
        f"Search-term hints: {', '.join(enriched.search_queries) or '(none)'}\n\n"
        f"Now: read the files named in the ticket, then explore as needed, "
        f"then produce the structured decomposition."
    )

    result = await agent.run(
        user,
        deps=deps,
        usage_limits=UsageLimits(request_limit=DEEP_MAX_REQUESTS),
    )
    # Patch back the ticket metadata the cheap path also fills in.
    decomp = result.output
    decomp.ticket_key = decomp.ticket_key or ticket.key
    decomp.ticket_title = decomp.ticket_title or ticket.title
    decomp.ticket_url = decomp.ticket_url or ticket.url
    decomp.enrichment_model = settings.enrich_model
    decomp.decomposition_model = settings.decompose_model
    return result, deps
