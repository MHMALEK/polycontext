"""Adapter backed by Cursor Agent CLI (``cursor-agent``).

The cursor-agent CLI ships as part of the Cursor install and runs the same
background agent that powers Cursor's editor flow. Headless form:

    cursor-agent -p "<prompt>"        # non-interactive, prints to stdout

Auth happens once on the host via ``cursor-agent login`` — we don't try to
manage that here. ``health()`` only checks the binary is on PATH; a fresh
install will succeed at that and then fail at first invocation with an
auth error, which we propagate as-is so the user knows what to fix.

Shared prompt assembly, subprocess wiring, and metrics live in
``_cli_agent_base.py``.
"""
from __future__ import annotations

from ._cli_agent_base import CliAgentAdapter, JobKind


class CursorAdapter(CliAgentAdapter):
    name = "cursor"
    description = "Cursor Agent CLI (cursor-agent) — Cursor's background agent in headless mode."

    _bin_name = "cursor-agent"
    _model_tag = "cursor"

    def _bin_override(self) -> str | None:
        return self.settings.cursor_bin

    def _timeout(self) -> float:
        return self.settings.cursor_timeout_seconds

    def _build_cmd(self, *, prompt: str, kind: JobKind) -> list[str]:
        return [self._bin(), "-p", prompt]
