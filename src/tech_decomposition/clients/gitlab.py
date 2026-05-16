"""Minimal GitLab MR client.

Just enough to open an MR after the ``implement`` wrapper has pushed a branch.
Uses raw httpx rather than ``python-gitlab`` to avoid a heavier dep for what
amounts to one POST. If/when we need richer GitLab interaction (review
comments, pipeline status, …), swap to ``python-gitlab`` here.

Auth: a project-scoped or personal access token with ``api`` scope, read from
``settings.gitlab_token``. We don't attempt OAuth — service accounts are the
expected use case.
"""
from __future__ import annotations

import logging
from typing import Any

import httpx

log = logging.getLogger(__name__)


class GitLabError(RuntimeError):
    pass


class GitLabClient:
    def __init__(self, base_url: str, token: str, *, timeout: float = 20.0):
        if not token:
            raise GitLabError("GITLAB_TOKEN is empty; set it to enable implement")
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=f"{self.base_url}/api/v4",
            headers={"PRIVATE-TOKEN": token},
            timeout=timeout,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "GitLabClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    async def create_merge_request(
        self,
        *,
        project_path: str,
        source_branch: str,
        target_branch: str,
        title: str,
        description: str = "",
        draft: bool = True,
        remove_source_branch: bool = False,
        labels: list[str] | None = None,
    ) -> dict[str, Any]:
        """Open an MR. Returns the GitLab MR JSON (including ``web_url``).

        ``project_path`` is the full namespace path, e.g.
        ``your-company/backend-api``. GitLab accepts URL-encoded
        paths as the ``id`` segment.
        """
        from urllib.parse import quote

        title_final = f"Draft: {title}" if draft and not title.lower().startswith("draft:") else title
        payload: dict[str, Any] = {
            "source_branch": source_branch,
            "target_branch": target_branch,
            "title": title_final,
            "description": description,
            "remove_source_branch": remove_source_branch,
        }
        if labels:
            payload["labels"] = ",".join(labels)

        project_id = quote(project_path, safe="")
        try:
            resp = await self._client.post(f"/projects/{project_id}/merge_requests", json=payload)
        except httpx.HTTPError as e:
            raise GitLabError(f"network error creating MR: {e}") from e
        if resp.status_code == 409:
            # An MR for this source→target pair already exists; return the existing one.
            existing = await self._find_open_mr(project_path, source_branch, target_branch)
            if existing:
                return existing
            raise GitLabError(f"409 from GitLab but no existing MR found: {resp.text}")
        if resp.status_code >= 300:
            raise GitLabError(f"{resp.status_code} from GitLab: {resp.text}")
        return resp.json()

    async def _find_open_mr(
        self, project_path: str, source_branch: str, target_branch: str,
    ) -> dict[str, Any] | None:
        from urllib.parse import quote

        project_id = quote(project_path, safe="")
        resp = await self._client.get(
            f"/projects/{project_id}/merge_requests",
            params={
                "state": "opened",
                "source_branch": source_branch,
                "target_branch": target_branch,
            },
        )
        if resp.status_code >= 300:
            return None
        items = resp.json() or []
        return items[0] if items else None
