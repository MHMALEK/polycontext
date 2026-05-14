"""Tests for the shared CLI runner using portable shell builtins.

We avoid mocking ``asyncio.create_subprocess_exec`` because the real thing is
cheap and the contract (cwd, env, stdin, timeout) is exactly what we want to
verify behaves correctly across OSes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from tech_decomposition.adapters._subprocess import (
    CliFailed,
    CliNotFound,
    CliTimeout,
    extract_json,
    resolve_binary,
    run_cli,
)


@pytest.mark.asyncio
async def test_run_cli_captures_stdout(tmp_path: Path) -> None:
    res = await run_cli(cmd=[sys.executable, "-c", "print('hi from stdout')"], cwd=tmp_path)
    assert "hi from stdout" in res.stdout
    assert res.returncode == 0
    assert res.duration_ms >= 0


@pytest.mark.asyncio
async def test_run_cli_captures_stdin(tmp_path: Path) -> None:
    res = await run_cli(
        cmd=[sys.executable, "-c", "import sys; print(sys.stdin.read())"],
        cwd=tmp_path,
        stdin="payload\n",
    )
    assert "payload" in res.stdout


@pytest.mark.asyncio
async def test_run_cli_propagates_env(tmp_path: Path) -> None:
    res = await run_cli(
        cmd=[sys.executable, "-c", "import os; print(os.environ.get('TD_TEST_VAR', 'unset'))"],
        cwd=tmp_path,
        env={"TD_TEST_VAR": "the-value"},
    )
    assert "the-value" in res.stdout


@pytest.mark.asyncio
async def test_run_cli_raises_on_nonzero_exit(tmp_path: Path) -> None:
    with pytest.raises(CliFailed) as ei:
        await run_cli(
            cmd=[sys.executable, "-c", "import sys; sys.stderr.write('boom'); sys.exit(7)"],
            cwd=tmp_path,
        )
    assert ei.value.returncode == 7
    assert "boom" in ei.value.stderr


@pytest.mark.asyncio
async def test_run_cli_raises_on_missing_binary(tmp_path: Path) -> None:
    with pytest.raises(CliNotFound):
        await run_cli(cmd=["definitely-not-a-real-binary-xyzzy"], cwd=tmp_path)


@pytest.mark.asyncio
async def test_run_cli_times_out(tmp_path: Path) -> None:
    with pytest.raises(CliTimeout):
        await run_cli(
            cmd=[sys.executable, "-c", "import time; time.sleep(5)"],
            cwd=tmp_path,
            timeout=0.3,
        )


def test_resolve_binary_finds_python() -> None:
    path = resolve_binary("python3") if _on_path("python3") else resolve_binary("python")
    assert Path(path).exists()


def test_resolve_binary_raises_on_missing() -> None:
    with pytest.raises(CliNotFound):
        resolve_binary("nope-not-real-zzz")


def test_resolve_binary_honors_override(tmp_path: Path) -> None:
    fake = tmp_path / "tool"
    fake.write_text("#!/bin/sh\necho hi\n")
    fake.chmod(0o755)
    assert resolve_binary("anything", override=str(fake)) == str(fake)


def test_resolve_binary_rejects_nonexec_override(tmp_path: Path) -> None:
    fake = tmp_path / "not-exec"
    fake.write_text("nope")
    with pytest.raises(CliNotFound):
        resolve_binary("x", override=str(fake))


def test_extract_json_handles_fenced_block() -> None:
    text = """Some preamble.
```json
{"foo": "bar", "n": 3}
```
And some trailing prose."""
    obj = extract_json(text)
    assert obj == {"foo": "bar", "n": 3}


def test_extract_json_picks_largest_object() -> None:
    text = '{"small": 1} and then later {"bigger": {"nested": true}, "more": "yes"}'
    obj = extract_json(text)
    assert "bigger" in obj


def test_extract_json_raises_when_none() -> None:
    with pytest.raises(ValueError):
        extract_json("no json here at all")


def _on_path(name: str) -> bool:
    import shutil
    return shutil.which(name) is not None
