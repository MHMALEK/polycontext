"""Adapter backed by Cursor Agent CLI (``cursor-agent``).

The cursor-agent CLI ships as part of the Cursor install and runs the same
background agent that powers Cursor's editor flow. Headless form:

    cursor-agent -p "<prompt>"        # non-interactive, prints to stdout

Auth happens once on the host via ``cursor-agent login`` — we don't try to
manage that here. ``health()`` only checks the binary is on PATH; a fresh
install will succeed at that and then fail at first invocation with an auth
error, which we propagate as-is so the user knows what to fix.
"""
from __future__ import annotations

import logging
import time

from ..models import Decomposition
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
from ._subprocess import (
    CliFailed,
    CliNotFound,
    CliTimeout,
    extract_json,
    resolve_binary,
    run_cli,
)

log = logging.getLogger(__name__)


_ASK_PREAMBLE = """Answer the following question about the codebase under this working
directory. Ground every claim in real code; cite file paths and line ranges.

Question:
"""

_DECOMPOSE_PREAMBLE = """Decompose this Jira ticket into a tech decomposition. Use
file-reading tools to ground every reference; never invent paths.

Output ONE JSON object matching this schema and nothing else (no prose, no fences):

{{
  "ticket_key": str | null, "ticket_title": str, "ticket_url": str | null,
  "overview": str, "affected_repos": [str], "risks": [str], "open_questions": [str],
  "subtasks": [
    {{"title": str, "description": str, "repo": str, "files": [str],
      "file_links": [str], "acceptance_criteria": [str],
      "estimated_complexity": "small"|"medium"|"large"|"unknown"}}
  ],
  "enrichment_model": "", "decomposition_model": "cursor"
}}

Ticket:
"""

_IMPLEMENT_PREAMBLE = """You are implementing a single subtask inside a clean git
worktree at this working directory. Edit only files inside this directory. Do not
commit, push, or open MRs.

Subtask:
"""


class CursorAdapter(Adapter):
    name = "cursor"
    capabilities: set[Capability] = {"ask", "decompose", "implement"}
    description = "Cursor Agent CLI (cursor-agent) — Cursor's background agent in headless mode."

    def _bin(self) -> str:
        return resolve_binary("cursor-agent", override=self.settings.cursor_bin)

    def health(self) -> dict:
        try:
            self._bin()
        except CliNotFound as e:
            return {"ok": False, "reason": str(e)}
        return {"ok": True}

    async def ask(self, inp: AdapterAskInput) -> AdapterAskResult:
        t = time.monotonic()
        res = await self._invoke(
            prompt=_ASK_PREAMBLE + inp.query,
            cwd=self.settings.repos_root,
            timeout=self.settings.cursor_timeout_seconds,
        )
        return AdapterAskResult(
            adapter=self.name,
            answer=res.stdout.strip(),
            citations=[],
            metrics=AdapterMetrics(duration_ms=int((time.monotonic() - t) * 1000)),
        )

    async def decompose(self, inp: AdapterDecomposeInput) -> AdapterDecomposeResult:
        t = time.monotonic()
        res = await self._invoke(
            prompt=_DECOMPOSE_PREAMBLE + _ticket_blob(inp),
            cwd=self.settings.repos_root,
            timeout=self.settings.cursor_timeout_seconds,
        )
        try:
            decomp = Decomposition.model_validate(extract_json(res.stdout))
        except Exception as e:
            raise RuntimeError(
                f"cursor decompose did not return parseable JSON: {e}\n--- raw ---\n{res.stdout[:2000]}"
            ) from e
        return AdapterDecomposeResult(
            adapter=self.name,
            decomposition=decomp,
            markdown=res.stdout,
            metrics=AdapterMetrics(duration_ms=int((time.monotonic() - t) * 1000)),
        )

    async def implement(
        self, inp: AdapterImplementInput, ctx: ImplementContext,
    ) -> AdapterImplementResult:
        t = time.monotonic()
        res = await self._invoke(
            prompt=_IMPLEMENT_PREAMBLE + _subtask_prompt(inp),
            cwd=ctx.worktree_path,
            timeout=self.settings.cursor_timeout_seconds,
        )
        return AdapterImplementResult(
            adapter=self.name,
            mr_url=None,
            branch=ctx.branch,
            commits=[],
            diff_summary=res.stdout[-1000:].strip(),
            files_changed=[],
            metrics=AdapterMetrics(duration_ms=int((time.monotonic() - t) * 1000)),
        )

    async def _invoke(self, *, prompt: str, cwd, timeout: float):
        cmd = [self._bin(), "-p", prompt]
        try:
            return await run_cli(cmd=cmd, cwd=cwd, timeout=timeout)
        except CliTimeout as e:
            raise RuntimeError(str(e)) from e
        except CliFailed as e:
            raise RuntimeError(
                f"cursor-agent failed (exit {e.returncode}): {e.stderr[:500]}"
            ) from e


def _ticket_blob(inp: AdapterDecomposeInput) -> str:
    parts = []
    if inp.ticket_key:
        parts.append(f"Ticket key: {inp.ticket_key}")
    if inp.ticket_url:
        parts.append(f"Ticket URL: {inp.ticket_url}")
    if inp.ticket_text:
        parts.append(inp.ticket_text)
    if inp.repos:
        parts.append(f"Repos to consider: {', '.join(inp.repos)}")
    return "\n\n".join(parts)


def _subtask_prompt(inp: AdapterImplementInput) -> str:
    if inp.subtask:
        st = inp.subtask
        out = [f"# {st.title}", "", st.description]
        if st.acceptance_criteria:
            out.append("")
            out.append("Acceptance criteria:")
            out.extend(f"- {ac}" for ac in st.acceptance_criteria)
        return "\n".join(out)
    return inp.free_text or "Implement the requested task."
