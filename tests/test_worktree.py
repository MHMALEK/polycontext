"""Tests for the git worktree helper using a real ephemeral repo on tmp_path."""
from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

import pytest

from tech_decomposition.core import worktree as wt_mod


def _make_repo(path: Path) -> None:
    """Initialize a tiny git repo with one commit on ``main``."""
    subprocess.run(["git", "init", "-q", "-b", "main", str(path)], check=True)
    # Local-only repos won't have an ``origin/main`` ref; the worktree helper
    # falls back to the local branch when fetch fails, so we don't need to
    # configure a remote.
    subprocess.run(["git", "-C", str(path), "config", "user.email", "t@t"], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "t"], check=True)
    (path / "README.md").write_text("hello\n")
    subprocess.run(["git", "-C", str(path), "add", "."], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-qm", "init"], check=True)


@pytest.mark.asyncio
async def test_create_then_remove_worktree(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _make_repo(repo)

    wt = await wt_mod.create_worktree(
        source_path=repo,
        parent_dir=tmp_path / "worktrees",
        run_id="abc123",
        repo="myrepo",
        base_branch="main",
    )
    assert wt.path.exists()
    assert (wt.path / "README.md").exists()
    assert wt.branch == "tech-decomp/abc123"

    await wt_mod.remove_worktree(wt, keep_branch=False)
    assert not wt.path.exists()


@pytest.mark.asyncio
async def test_status_porcelain_returns_changed_paths(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _make_repo(repo)

    wt = await wt_mod.create_worktree(
        source_path=repo,
        parent_dir=tmp_path / "worktrees",
        run_id="zzz",
        repo="myrepo",
        base_branch="main",
    )
    try:
        (wt.path / "new_file.txt").write_text("data")
        (wt.path / "README.md").write_text("changed\n")
        changed = await wt_mod.git_status_porcelain(wt)
        assert "new_file.txt" in changed
        assert "README.md" in changed
    finally:
        await wt_mod.remove_worktree(wt)


@pytest.mark.asyncio
async def test_commit_and_diff_summary(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _make_repo(repo)

    wt = await wt_mod.create_worktree(
        source_path=repo,
        parent_dir=tmp_path / "worktrees",
        run_id="q1",
        repo="myrepo",
        base_branch="main",
    )
    try:
        (wt.path / "x.py").write_text("print('hi')\n")
        sha = await wt_mod.git_commit_all(wt, "feat: add x", author_name="t", author_email="t@t")
        assert len(sha) >= 7
        summary = await wt_mod.git_diff_summary(wt)
        assert "x.py" in summary
    finally:
        await wt_mod.remove_worktree(wt)


@pytest.mark.asyncio
async def test_create_fails_on_missing_repo(tmp_path: Path) -> None:
    with pytest.raises(wt_mod.WorktreeError):
        await wt_mod.create_worktree(
            source_path=tmp_path / "nope",
            parent_dir=tmp_path / "worktrees",
            run_id="x",
            repo="x",
        )
