"""Adapter backed by OpenCode (https://opencode.ai).

OpenCode is a provider-agnostic terminal agent. We drive it through its
non-interactive ``opencode run`` mode, which takes a prompt and prints the
final response to stdout.

All three jobs use the same shape:
  * ``ask``       — cwd=repos_root, prompt = question + "answer with citations"
  * ``decompose`` — cwd=repos_root, prompt asks for the JSON schema
  * ``implement`` — cwd=worktree_path, prompt = subtask

Provider/model config lives in OpenCode's own config (``~/.config/opencode/``)
or via env vars (``ANTHROPIC_API_KEY``, etc.). We don't try to override it —
the user picks what their OpenCode is talking to.
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
directory. Ground every claim in real code; cite file paths with line ranges where
relevant. Keep the answer concise and engineer-oriented.

Question:
"""

_DECOMPOSE_PREAMBLE = """Decompose this Jira ticket into a tech decomposition. Use
file-reading tools to ground every reference; never invent paths.

Output ONE JSON object matching this schema and nothing else (no prose, no
fences, no markdown):

{{
  "ticket_key": str | null, "ticket_title": str, "ticket_url": str | null,
  "overview": str, "affected_repos": [str], "risks": [str], "open_questions": [str],
  "subtasks": [
    {{"title": str, "description": str, "repo": str, "files": [str],
      "file_links": [str], "acceptance_criteria": [str],
      "estimated_complexity": "small"|"medium"|"large"|"unknown"}}
  ],
  "enrichment_model": "", "decomposition_model": "opencode"
}}

Ticket:
"""

_IMPLEMENT_PREAMBLE = """You are implementing a single subtask inside a clean git
worktree at this working directory. Edit only files inside this directory. Do not
commit, push, or open MRs — the surrounding system does that after you finish.

Subtask:
"""


class OpenCodeAdapter(Adapter):
    name = "opencode"
    capabilities: set[Capability] = {"ask", "decompose", "implement"}
    description = "OpenCode (opencode.ai) — provider-agnostic terminal agent, via opencode run."

    def _bin(self) -> str:
        return resolve_binary("opencode", override=self.settings.opencode_bin)

    def health(self) -> dict:
        try:
            self._bin()
        except CliNotFound as e:
            return {"ok": False, "reason": str(e)}
        return {"ok": True}

    async def ask(self, inp: AdapterAskInput) -> AdapterAskResult:
        prompt = _ASK_PREAMBLE + inp.query
        t = time.monotonic()
        res = await self._invoke(
            prompt=prompt,
            cwd=self.settings.repos_root,
            timeout=self.settings.opencode_timeout_seconds,
        )
        return AdapterAskResult(
            adapter=self.name,
            answer=res.stdout.strip(),
            citations=[],
            metrics=AdapterMetrics(
                duration_ms=int((time.monotonic() - t) * 1000),
                extra={"stderr_excerpt": res.stderr[:200] if res.stderr else ""},
            ),
        )

    async def decompose(self, inp: AdapterDecomposeInput) -> AdapterDecomposeResult:
        prompt = _DECOMPOSE_PREAMBLE + _ticket_blob(inp)
        t = time.monotonic()
        res = await self._invoke(
            prompt=prompt,
            cwd=self.settings.repos_root,
            timeout=self.settings.opencode_timeout_seconds,
        )
        try:
            decomp = Decomposition.model_validate(extract_json(res.stdout))
        except Exception as e:
            raise RuntimeError(
                f"opencode decompose did not return parseable JSON: {e}\n--- raw ---\n{res.stdout[:2000]}"
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
        prompt = _IMPLEMENT_PREAMBLE + _subtask_prompt(inp)
        t = time.monotonic()
        res = await self._invoke(
            prompt=prompt,
            cwd=ctx.worktree_path,
            timeout=self.settings.opencode_timeout_seconds,
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
        # ``opencode run "<prompt>"`` is already the non-interactive form per
        # https://opencode.ai/docs/cli/. No --print flag; --dir is the cwd
        # override, but we pass cwd via the OS so it works identically. Add
        # ``--format json`` later if we want structured event output.
        cmd = [self._bin(), "run", str(prompt)]
        try:
            return await run_cli(cmd=cmd, cwd=cwd, timeout=timeout)
        except CliTimeout as e:
            raise RuntimeError(str(e)) from e
        except CliFailed as e:
            raise RuntimeError(
                f"opencode failed (exit {e.returncode}): {e.stderr[:500]}"
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
