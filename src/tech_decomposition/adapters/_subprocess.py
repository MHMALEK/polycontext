"""Shared subprocess runner for CLI-backed adapters (OpenCode, Goose, Cursor, ...).

Every CLI we wrap has its own quirks (flag names, output format, auth model),
but the *shape* of "run this binary inside a directory with these env vars and
a prompt on stdin, time out cleanly, capture stdout/stderr" is the same. This
module owns that shape.

Design choices:
  * Subprocess runs via ``asyncio.create_subprocess_exec`` so FastAPI's event
    loop isn't blocked.
  * Stdin is the prompt by default — most modern code-agent CLIs accept this.
    Adapters that need ``--message "..."`` instead pass the prompt as an arg.
  * Stderr is captured separately and only surfaced on non-zero exit; chatty
    CLIs write progress to stderr and we don't want that polluting the result.
  * Timeouts are enforced via ``asyncio.wait_for`` with a graceful kill on the
    process group so subagents don't get orphaned.
"""
from __future__ import annotations

import asyncio
import logging
import os
import shlex
import shutil
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

log = logging.getLogger(__name__)


class CliNotFound(RuntimeError):
    """The CLI binary isn't on PATH (or at the configured location)."""


class CliFailed(RuntimeError):
    """The CLI exited non-zero. ``stderr`` and ``stdout`` are attached."""

    def __init__(self, cmd: list[str], returncode: int, stdout: str, stderr: str):
        self.cmd = cmd
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        super().__init__(
            f"{shlex.join(cmd)} exited {returncode}\n--- stderr ---\n{stderr[:2000]}"
        )


class CliTimeout(RuntimeError):
    """The CLI didn't finish within ``timeout`` seconds."""


@dataclass
class CliResult:
    stdout: str
    stderr: str
    duration_ms: int
    returncode: int


def resolve_binary(name: str, override: str | None = None) -> str:
    """Find a CLI binary on PATH or at an explicit override path.

    Used by adapter ``health()`` checks so the UI can tell the user
    "install ``foo`` to enable this adapter" before they fire off a request.
    """
    if override:
        if Path(override).is_file() and os.access(override, os.X_OK):
            return override
        raise CliNotFound(f"{name!r} override {override!r} is not an executable file")
    path = shutil.which(name)
    if not path:
        raise CliNotFound(f"{name!r} not on PATH")
    return path


async def run_cli(
    *,
    cmd: list[str],
    cwd: Path,
    stdin: str | None = None,
    env: Mapping[str, str] | None = None,
    timeout: float = 300.0,
) -> CliResult:
    """Run a subprocess with stdin/cwd/env and bounded time.

    Raises:
        CliNotFound: ``cmd[0]`` is missing.
        CliFailed:   process exited non-zero.
        CliTimeout:  ``timeout`` elapsed before the process finished.
    """
    if not Path(cmd[0]).is_absolute() and shutil.which(cmd[0]) is None:
        raise CliNotFound(f"{cmd[0]!r} not on PATH")

    full_env = dict(os.environ)
    if env:
        full_env.update(env)

    log.info("subprocess: cwd=%s cmd=%s timeout=%s", cwd, shlex.join(cmd), timeout)
    t0 = time.monotonic()

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(cwd),
        env=full_env,
        stdin=asyncio.subprocess.PIPE if stdin is not None else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        # New process group so a hung child gets the whole tree killed cleanly.
        start_new_session=True,
    )
    try:
        stdout_b, stderr_b = await asyncio.wait_for(
            proc.communicate(input=(stdin.encode() if stdin else None)),
            timeout=timeout,
        )
    except asyncio.TimeoutError as e:
        # Try a soft kill first; some agents catch SIGTERM to flush state.
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            await asyncio.wait_for(proc.wait(), timeout=5.0)
        except (asyncio.TimeoutError, ProcessLookupError):
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
        raise CliTimeout(f"{shlex.join(cmd)} timed out after {timeout}s") from e

    duration_ms = int((time.monotonic() - t0) * 1000)
    stdout = stdout_b.decode(errors="replace")
    stderr = stderr_b.decode(errors="replace")
    if proc.returncode != 0:
        raise CliFailed(cmd=cmd, returncode=proc.returncode or -1, stdout=stdout, stderr=stderr)
    return CliResult(stdout=stdout, stderr=stderr, duration_ms=duration_ms, returncode=0)


# ---------------------------------------------------------------------------
# Output parsing helpers (shared across CLI adapters)
# ---------------------------------------------------------------------------


import json
import re

_JSON_OBJECT_RE = re.compile(r"\{(?:[^{}]|\{[^{}]*\})*\}", re.DOTALL)


def extract_json(text: str) -> dict:
    """Pull the largest top-level JSON object out of a possibly-noisy answer.

    Same heuristic CLI adapters use — agent CLIs love to wrap JSON
    in fenced code blocks or sandwich it between explanation paragraphs.
    """
    # Strip common ```json ... ``` fences first; they wreck the regex above.
    stripped = re.sub(r"```(?:json)?\s*", "", text)
    stripped = stripped.replace("```", "")
    candidates = _JSON_OBJECT_RE.findall(stripped)
    candidates.sort(key=len, reverse=True)
    for c in candidates:
        try:
            obj = json.loads(c)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise ValueError("no JSON object found in CLI output")
