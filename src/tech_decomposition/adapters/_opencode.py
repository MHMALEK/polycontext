"""Adapter backed by OpenCode (https://opencode.ai).

OpenCode is a provider-agnostic terminal agent. We drive it through its
non-interactive ``opencode run`` mode, which takes a prompt and prints
the final response to stdout.

Provider/model config lives in OpenCode's own config (``~/.config/opencode/``)
or via env vars (``ANTHROPIC_API_KEY`` etc.). We don't override it — the
user picks what their OpenCode is talking to.

The shared prompt assembly, subprocess wiring, and metrics live in
``_cli_agent_base.py``. This file only declares the binary name, the
argv shape, and (eventually) a JSON-event parser for richer metrics.
"""
from __future__ import annotations

from ._cli_agent_base import (
    CliAgentAdapter,
    JobKind,
    ParsedAgentRun,
    parse_jsonl_events,
)


class OpenCodeAdapter(CliAgentAdapter):
    name = "opencode"
    description = "OpenCode (opencode.ai) — provider-agnostic terminal agent, via opencode run."

    _bin_name = "opencode"
    _model_tag = "opencode"

    def _bin_override(self) -> str | None:
        return self.settings.opencode_bin

    def _timeout(self) -> float:
        return self.settings.opencode_timeout_seconds

    def _build_cmd(self, *, prompt: str, kind: JobKind) -> list[str]:
        # ``opencode run --format json`` emits raw JSON events so we can
        # pick up tokens / cost / model / tool-call counts. Per
        # ``opencode run --help``: "--format: default (formatted) or
        # json (raw JSON events)".
        return [self._bin(), "run", "--format", "json", str(prompt)]

    def _parse_result(self, *, stdout: str, kind: JobKind) -> ParsedAgentRun:
        parsed = parse_jsonl_events(stdout)
        if parsed is not None and parsed.answer:
            return parsed
        # Fallback — pre-JSON-format builds or unexpected stdout shape.
        return ParsedAgentRun(answer=stdout.strip())


class OpenCodeGroundedAdapter(OpenCodeAdapter):
    """OpenCode with a Sourcebot+ripgrep retrieval prelude on ask/decompose."""

    name = "opencode_grounded"
    description = (
        "OpenCode with cross-repo Sourcebot + ripgrep retrieval injected "
        "into the prompt — gives the agent file pointers it can't find on its own."
    )

    def __init__(self, settings):
        super().__init__(settings, grounded=True)
