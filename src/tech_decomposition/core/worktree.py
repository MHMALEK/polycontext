"""Isolated git worktrees for ``implement`` runs.

Each implement call gets its own worktree on a fresh branch so:
  * the main clone Sourcebot reads from stays clean
  * concurrent implement runs don't fight over the index
  * a failed run is trivially cleaned up

Worktrees are created under ``{settings.output_dir}/worktrees/{run_id}`` by
default. The shared parent directory is created lazily.

Caller responsibility:
    The adapter only edits files inside ``path``. Commit/push/MR is handled
    by ``core.implement_runner``.
"""
from __future__ import annotations

import asyncio
import logging
import shutil
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator

log = logging.getLogger(__name__)


class WorktreeError(RuntimeError):
    pass


@dataclass
class Worktree:
    repo: str           # logical repo name (key in settings.gitlab_projects)
    source_path: Path   # the long-lived clone under REPOS_ROOT
    path: Path          # the isolated worktree dir (lives under output_dir/worktrees)
    branch: str         # branch name created on this worktree
    base_branch: str    # what we branched off


async def _run(cwd: Path, *args: str) -> str:
    """Run a git command, return stdout, raise on non-zero exit."""
    proc = await asyncio.create_subprocess_exec(
        "git", *args,
        cwd=str(cwd),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise WorktreeError(
            f"git {' '.join(args)} failed in {cwd}: {stderr.decode(errors='replace').strip()}"
        )
    return stdout.decode(errors="replace")


async def create_worktree(
    *,
    source_path: Path,
    parent_dir: Path,
    run_id: str,
    repo: str,
    base_branch: str = "main",
    branch_prefix: str = "tech-decomp",
) -> Worktree:
    """Provision an isolated worktree.

    Equivalent to:
        cd {source_path} && git fetch origin {base_branch} \\
            && git worktree add {parent_dir}/{run_id} \\
                -b {branch_prefix}/{run_id} origin/{base_branch}

    Returns a ``Worktree`` describing the result. The directory at ``path``
    is guaranteed to exist on success.
    """
    if not source_path.exists():
        raise WorktreeError(f"source repo not found: {source_path}")
    if not (source_path / ".git").exists():
        raise WorktreeError(f"{source_path} is not a git checkout")

    branch = f"{branch_prefix}/{run_id}"
    parent_dir.mkdir(parents=True, exist_ok=True)
    target = parent_dir / run_id
    if target.exists():
        # Stale dir from a previous run — wipe before reusing the name. Worktrees
        # outside the repo's normal worktree list can be left over if a process
        # was killed before cleanup ran.
        await _force_remove(source_path, target)

    # Fetch to make sure origin/{base_branch} is up to date. Best-effort: if there's
    # no remote configured (local-only repos in dev), fall back to the local branch.
    try:
        await _run(source_path, "fetch", "origin", base_branch)
        base_ref = f"origin/{base_branch}"
    except WorktreeError as e:
        log.warning("worktree: fetch failed (%s); falling back to local %s", e, base_branch)
        base_ref = base_branch

    await _run(source_path, "worktree", "add", str(target), "-b", branch, base_ref)
    return Worktree(
        repo=repo,
        source_path=source_path,
        path=target,
        branch=branch,
        base_branch=base_branch,
    )


async def remove_worktree(wt: Worktree, *, keep_branch: bool = True) -> None:
    """Tear down a worktree. By default the branch is kept on the local repo
    (and on remote if it was pushed) so reviewers can still find it.
    """
    try:
        await _run(wt.source_path, "worktree", "remove", "--force", str(wt.path))
    except WorktreeError as e:
        log.warning("worktree: remove --force failed (%s); falling back to rm -rf", e)
        await _force_remove(wt.source_path, wt.path)
    if not keep_branch:
        # Best-effort branch deletion. If the branch was pushed and the user wants
        # it removed remotely too, that's a separate call.
        try:
            await _run(wt.source_path, "branch", "-D", wt.branch)
        except WorktreeError as e:
            log.warning("worktree: local branch delete failed: %s", e)


async def _force_remove(source_path: Path, target: Path) -> None:
    """Last-resort cleanup when ``git worktree remove`` won't work."""
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)
    # Prune dangling worktree metadata so subsequent ``worktree add`` doesn't complain.
    try:
        await _run(source_path, "worktree", "prune")
    except WorktreeError:
        pass


@asynccontextmanager
async def worktree_for_run(
    *,
    source_path: Path,
    parent_dir: Path,
    run_id: str,
    repo: str,
    base_branch: str = "main",
    keep_branch: bool = True,
) -> AsyncIterator[Worktree]:
    """Async context manager: create + cleanup."""
    wt = await create_worktree(
        source_path=source_path,
        parent_dir=parent_dir,
        run_id=run_id,
        repo=repo,
        base_branch=base_branch,
    )
    try:
        yield wt
    finally:
        await remove_worktree(wt, keep_branch=keep_branch)


# ---------------------------------------------------------------------------
# Helpers used by the implement_runner after the adapter has written files.
# Kept here so all git plumbing lives in one module.
# ---------------------------------------------------------------------------


async def git_status_porcelain(wt: Worktree) -> list[str]:
    """Return the list of changed paths (porcelain v1, just the path part)."""
    out = await _run(wt.path, "status", "--porcelain")
    paths: list[str] = []
    for line in out.splitlines():
        # Format is ``XY path`` or ``XY path -> newpath``; we keep the final token.
        if not line.strip():
            continue
        parts = line[3:].split(" -> ")
        paths.append(parts[-1].strip())
    return paths


async def git_commit_all(wt: Worktree, message: str, *, author_name: str | None = None,
                        author_email: str | None = None) -> str:
    """Stage everything and create one commit. Returns the new commit SHA."""
    env_args: list[str] = []
    if author_name and author_email:
        # ``git -c`` flags scope only to this invocation; safer than mutating the local config.
        env_args = ["-c", f"user.name={author_name}", "-c", f"user.email={author_email}"]

    await _run(wt.path, "add", "-A")
    await _run(wt.path, *env_args, "commit", "-m", message)
    out = await _run(wt.path, "rev-parse", "HEAD")
    return out.strip()


async def git_push(wt: Worktree, *, remote: str = "origin", set_upstream: bool = True) -> None:
    args = ["push"]
    if set_upstream:
        args += ["-u", remote, wt.branch]
    else:
        args += [remote, wt.branch]
    await _run(wt.path, *args)


async def git_diff_summary(wt: Worktree) -> str:
    """``git diff --stat`` against base — for the MR description."""
    return await _run(wt.path, "diff", "--stat", f"{wt.base_branch}...HEAD")
