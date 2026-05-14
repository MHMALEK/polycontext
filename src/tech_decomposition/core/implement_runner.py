"""Glue: worktree → adapter.implement() → commit → push → open MR.

The same flow runs for every adapter; only the editing step (the adapter
itself) differs. Keeping the orchestration here means each adapter stays
small and focused on "edit files in this directory".

Failure semantics:
  * Worktree provisioning failure → propagate as 5xx; no cleanup needed
    (worktree creation is atomic from the caller's POV).
  * Adapter raises → mark the run failed, attempt cleanup, propagate.
  * Adapter returns but ``git status`` is empty → return a result with
    ``mr_url=None`` and a clear ``diff_summary='no changes'``. The bake-off UI
    will show this as a soft failure rather than crash.
  * Commit/push/MR failure → propagate; cleanup still runs.
"""
from __future__ import annotations

import logging
from pathlib import Path

from ..adapters.base import (
    Adapter,
    AdapterImplementInput,
    AdapterImplementResult,
    AdapterMetrics,
    ImplementContext,
)
from ..clients.gitlab import GitLabClient, GitLabError
from ..config import Settings
from . import worktree as wt_mod

log = logging.getLogger(__name__)


async def run_implement(
    *,
    adapter: Adapter,
    inp: AdapterImplementInput,
    settings: Settings,
    run_id: str,
) -> AdapterImplementResult:
    """End-to-end implement run.

    Caller (typically the API route) supplies ``run_id`` so it matches the
    run row in the runstore. The returned ``AdapterImplementResult`` is
    augmented with branch/commits/mr_url that the adapter itself never sees.
    """
    if inp.repo not in settings.gitlab_projects:
        raise ValueError(
            f"unknown repo {inp.repo!r}; configure GITLAB_PROJECTS to map it"
        )
    source_path = settings.repo_path(inp.repo)
    worktrees_root = Path(settings.output_dir) / "worktrees"

    async with wt_mod.worktree_for_run(
        source_path=source_path,
        parent_dir=worktrees_root,
        run_id=run_id,
        repo=inp.repo,
        base_branch=inp.base_branch,
    ) as wt:
        ctx = ImplementContext(
            worktree_path=wt.path,
            branch=wt.branch,
            repo=wt.repo,
            base_branch=wt.base_branch,
        )

        # 1) Let the adapter edit files.
        adapter_result = await adapter.implement(inp, ctx)

        # 2) Did anything change?
        changed = await wt_mod.git_status_porcelain(wt)
        if not changed:
            return AdapterImplementResult(
                adapter=adapter.name,
                mr_url=None,
                branch=wt.branch,
                commits=[],
                diff_summary="no changes",
                files_changed=[],
                metrics=adapter_result.metrics or AdapterMetrics(),
            )

        # 3) Commit. Single squashed commit per run for now — easier to revert
        #    and keeps the MR small. If an adapter wants multi-commit, we'll
        #    revisit (probably by letting the adapter commit internally and
        #    skipping this step).
        commit_msg = _commit_message(inp, run_id)
        sha = await wt_mod.git_commit_all(
            wt,
            commit_msg,
            author_name=settings.git_author_name or None,
            author_email=settings.git_author_email or None,
        )

        # 4) Push.
        await wt_mod.git_push(wt, remote="origin", set_upstream=True)

        # 5) Open the MR (if a token is configured).
        mr_url: str | None = None
        if settings.gitlab_token:
            project_path = settings.gitlab_projects[inp.repo]
            description = _mr_description(inp, run_id, adapter.name)
            title = (inp.subtask.title if inp.subtask else inp.free_text or "Implement task")[:200]
            async with GitLabClient(settings.gitlab_base_url, settings.gitlab_token) as gl:
                try:
                    mr = await gl.create_merge_request(
                        project_path=project_path,
                        source_branch=wt.branch,
                        target_branch=wt.base_branch,
                        title=title,
                        description=description,
                        draft=inp.draft,
                    )
                    mr_url = mr.get("web_url")
                except GitLabError as e:
                    log.warning("MR creation failed (%s); branch is pushed: %s", e, wt.branch)

        diff_summary = await wt_mod.git_diff_summary(wt)

        return AdapterImplementResult(
            adapter=adapter.name,
            mr_url=mr_url,
            branch=wt.branch,
            commits=[sha],
            diff_summary=diff_summary,
            files_changed=changed,
            metrics=adapter_result.metrics or AdapterMetrics(),
        )


def _commit_message(inp: AdapterImplementInput, run_id: str) -> str:
    title = (inp.subtask.title if inp.subtask else inp.free_text or "Implement task").strip()
    return f"{title}\n\n[td run {run_id}]"


def _mr_description(inp: AdapterImplementInput, run_id: str, adapter_name: str) -> str:
    lines = [
        f"Generated by tech-decomposition via the **{adapter_name}** adapter.",
        f"Run id: `{run_id}`",
    ]
    if inp.ticket_key:
        lines.append(f"Ticket: `{inp.ticket_key}`")
    if inp.subtask:
        lines.append("")
        lines.append("## Subtask")
        lines.append(f"**{inp.subtask.title}**")
        lines.append("")
        lines.append(inp.subtask.description)
        if inp.subtask.acceptance_criteria:
            lines.append("")
            lines.append("## Acceptance criteria")
            for ac in inp.subtask.acceptance_criteria:
                lines.append(f"- {ac}")
    elif inp.free_text:
        lines.append("")
        lines.append("## Task")
        lines.append(inp.free_text)
    return "\n".join(lines)
