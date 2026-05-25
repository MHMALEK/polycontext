"""Workspace tools registered with pydantic-AI agents.

Why these exist
---------------
We used to have six per-SDK streamers (opencode, gemini, claude_code,
openai_agents, cursor, cline) — each wrapping its own SDK's tool taxonomy
to give the user a streamed answer with live tool calls. That worked but
was ~2000 lines of translation code and six SDK deps to keep current.

This module replaces all six. Pydantic-AI's ``Agent.tool`` decorator
registers Python functions as tools the model can call; ``run_stream_events``
gives us text deltas + tool start/result events in one unified shape that
works for Gemini, Anthropic, OpenAI, and any OpenRouter-hosted model.

The tools mirror the read-only workspace exploration tools every adapter
exposes today (read_file, grep_search, glob, list_directory). They live
in one place so we don't have to re-implement them inside each SDK
wrapper. Path safety is enforced by clamping every path under the
workspace root.

Design notes
------------
- All tools take a ``ctx: RunContext[Workspace]`` so the cwd is injected
  per-request (the model never sees absolute paths).
- Each tool returns a string the model can read directly. Errors are
  returned as strings starting with "ERROR:" so the model can recover.
- File reads are capped at 50 KB by default so the context window doesn't
  blow up on huge files (configurable per-call via the ``max_bytes`` arg).
- Glob/grep cap result count to keep the output bounded.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Tool result caps — keep model outputs bounded so a single read/grep
# call can't fill the context window.
DEFAULT_MAX_BYTES = 50_000
GREP_MAX_MATCHES = 60
GLOB_MAX_MATCHES = 80
LS_MAX_ENTRIES = 200
# Directories we never recurse into — skip the usual giant noise sinks
# so glob/grep don't return useless matches from build artifacts.
_SKIP_DIRS = {
    ".git",
    "node_modules",
    ".venv",
    "venv",
    "__pycache__",
    ".tox",
    "dist",
    "build",
    ".next",
    "target",
    ".cargo",
    ".idea",
    ".vscode",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
}


@dataclass
class Workspace:
    """Dependency injected into every tool call. Carries the absolute
    workspace root so each tool can resolve relative paths safely."""

    root: Path

    def resolve(self, rel: str) -> Path:
        """Resolve ``rel`` against ``root``, ensuring the result stays
        within the workspace. Raises ValueError on escape attempts."""
        rel = (rel or "").strip()
        if not rel or rel == ".":
            return self.root
        # Reject absolute paths and explicit traversal — the model
        # shouldn't be reading outside the workspace.
        if rel.startswith("/") or ".." in rel.split("/"):
            raise ValueError(f"path must be relative + inside workspace: {rel}")
        resolved = (self.root / rel).resolve()
        try:
            resolved.relative_to(self.root.resolve())
        except ValueError as e:
            raise ValueError(f"path escapes workspace root: {rel}") from e
        return resolved


# ---------------------------------------------------------------------------
# Tool implementations (plain async functions; registered by build_agent_tools)
# ---------------------------------------------------------------------------


async def _read_file(ws: Workspace, path: str, max_bytes: int = DEFAULT_MAX_BYTES) -> str:
    """Read a UTF-8 text file. Returns the file contents (truncated to
    ``max_bytes`` if larger), prefixed with the resolved path so the model
    can cite it in its answer."""
    try:
        p = ws.resolve(path)
    except ValueError as e:
        return f"ERROR: {e}"
    if not p.exists():
        return f"ERROR: file not found: {path}"
    if not p.is_file():
        return f"ERROR: not a regular file: {path}"
    try:
        cap = min(max(1, int(max_bytes)), 500_000)
    except (TypeError, ValueError):
        cap = DEFAULT_MAX_BYTES
    try:
        data = p.read_bytes()[:cap]
        text = data.decode("utf-8", errors="replace")
    except OSError as e:
        return f"ERROR: read failed: {e}"
    truncated = "" if p.stat().st_size <= cap else f"\n[truncated to {cap} bytes; file is {p.stat().st_size} bytes]"
    return f"# {path}\n{text}{truncated}"


async def _list_directory(ws: Workspace, path: str = ".") -> str:
    """List non-hidden files and folders in a directory. Returns a
    newline-separated listing with ``/`` suffix for directories."""
    try:
        p = ws.resolve(path)
    except ValueError as e:
        return f"ERROR: {e}"
    if not p.exists():
        return f"ERROR: directory not found: {path}"
    if not p.is_dir():
        return f"ERROR: not a directory: {path}"
    try:
        entries = []
        for child in sorted(p.iterdir()):
            if child.name.startswith("."):
                continue
            if child.name in _SKIP_DIRS:
                continue
            entries.append(f"{child.name}{'/' if child.is_dir() else ''}")
        if not entries:
            return f"(empty: {path})"
        if len(entries) > LS_MAX_ENTRIES:
            entries = entries[:LS_MAX_ENTRIES] + [f"... ({len(entries) - LS_MAX_ENTRIES} more)"]
        return "\n".join(entries)
    except OSError as e:
        return f"ERROR: list failed: {e}"


async def _glob(ws: Workspace, pattern: str) -> str:
    """Find files whose path contains ``pattern`` (case-insensitive
    substring). Skips standard build/dependency directories. Returns
    repo-relative paths, one per line."""
    pat = (pattern or "").strip().lower()
    if not pat:
        return "ERROR: empty pattern"
    matches: list[str] = []
    try:
        for root, dirs, files in _walk(ws.root):
            for f in files:
                full = root / f
                rel = str(full.relative_to(ws.root))
                if pat in rel.lower():
                    matches.append(rel)
                    if len(matches) >= GLOB_MAX_MATCHES:
                        break
            if len(matches) >= GLOB_MAX_MATCHES:
                break
    except OSError as e:
        return f"ERROR: glob failed: {e}"
    if not matches:
        return f"(no matches for `{pattern}`)"
    return "\n".join(matches)


async def _grep_search(ws: Workspace, query: str, max_matches: int = GREP_MAX_MATCHES) -> str:
    """Search file contents for ``query`` (literal string, case-sensitive).
    Returns ``path:line:content`` rows for matching lines, capped at
    ``max_matches``."""
    q = (query or "").strip()
    if not q:
        return "ERROR: empty query"
    try:
        cap = min(max(1, int(max_matches)), 200)
    except (TypeError, ValueError):
        cap = GREP_MAX_MATCHES
    pattern = re.escape(q) if not _looks_like_regex(q) else q
    try:
        compiled = re.compile(pattern)
    except re.error:
        compiled = re.compile(re.escape(q))
    rows: list[str] = []
    try:
        for root, dirs, files in _walk(ws.root):
            for f in files:
                full = root / f
                if not _looks_textual(full):
                    continue
                try:
                    for i, line in enumerate(
                        full.read_text(encoding="utf-8", errors="replace").splitlines(),
                        start=1,
                    ):
                        if compiled.search(line):
                            rel = full.relative_to(ws.root)
                            rows.append(f"{rel}:{i}:{line.strip()[:200]}")
                            if len(rows) >= cap:
                                break
                except OSError:
                    continue
                if len(rows) >= cap:
                    break
            if len(rows) >= cap:
                break
    except OSError as e:
        return f"ERROR: grep failed: {e}"
    if not rows:
        return f"(no matches for `{query}`)"
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _walk(root: Path):
    """os.walk replacement that prunes the skip-dirs in-place. Yields
    (Path, dirs, files) triples like os.walk."""
    import os

    for dirpath, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS and not d.startswith(".")]
        yield Path(dirpath), dirs, files


def _looks_textual(p: Path) -> bool:
    """Cheap heuristic: skip files whose extension is binary or whose
    size exceeds 1 MB. Avoids loading huge JSON / binary blobs into a
    line-iterating grep."""
    if p.stat().st_size > 1_000_000:
        return False
    binary_exts = {
        ".png", ".jpg", ".jpeg", ".gif", ".webp", ".pdf", ".ico",
        ".zip", ".tar", ".gz", ".tgz", ".bz2", ".xz",
        ".exe", ".dll", ".so", ".dylib", ".bin",
        ".pyc", ".pyo", ".class", ".jar",
        ".mp3", ".mp4", ".mov", ".wav",
        ".db", ".sqlite", ".sqlite3",
    }
    return p.suffix.lower() not in binary_exts


def _looks_like_regex(s: str) -> bool:
    return any(c in s for c in ".*+?^$()[]{}|\\")


# ---------------------------------------------------------------------------
# Agent wiring
# ---------------------------------------------------------------------------


def attach_workspace_tools(agent: Any, ws: Workspace) -> None:
    """Register the four workspace tools on a pydantic-AI Agent. The
    workspace is captured in the closure so the model never sees the
    absolute path — it just calls ``read_file('src/foo.py')`` and the
    tool resolves against the configured root."""

    @agent.tool_plain
    async def read_file(path: str, max_bytes: int = DEFAULT_MAX_BYTES) -> str:
        """Read a UTF-8 text file under the workspace.

        Args:
            path: File path relative to the workspace root (e.g. 'src/foo.py').
            max_bytes: Optional cap on bytes returned (default 50000).
        """
        return await _read_file(ws, path, max_bytes)

    @agent.tool_plain
    async def list_directory(path: str = ".") -> str:
        """List non-hidden files and folders in a directory under the workspace.

        Args:
            path: Directory path relative to the workspace root; use '.' for the root.
        """
        return await _list_directory(ws, path)

    @agent.tool_plain
    async def glob(pattern: str) -> str:
        """Find files whose path contains the given substring pattern.
        Skips standard build/dep directories. Returns repo-relative paths.

        Args:
            pattern: Case-insensitive substring to match within file paths
              (e.g. 'tests/auth' or 'config.py').
        """
        return await _glob(ws, pattern)

    @agent.tool_plain
    async def grep_search(query: str, max_matches: int = GREP_MAX_MATCHES) -> str:
        """Search file contents for a literal string or regex pattern.
        Returns 'path:line:content' rows.

        Args:
            query: Literal string or regex pattern.
            max_matches: Optional cap on matches returned (default 60).
        """
        return await _grep_search(ws, query, max_matches)
