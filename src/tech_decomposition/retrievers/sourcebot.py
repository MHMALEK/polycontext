from __future__ import annotations

import asyncio
import logging
import re as _re
import subprocess
from pathlib import Path

import httpx

from ..config import Settings
from ..models import EnrichedQuery, RepoContext, Snippet
from .base import Retriever

log = logging.getLogger(__name__)


def _git_head(repo_path: Path) -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_path, text=True)
        return out.strip()
    except Exception:
        return "HEAD"


class SourcebotRetriever(Retriever):
    """
    Calls Sourcebot's `POST /api/search`. Maps results to Snippet.

    If `SOURCEBOT_URL` or `SOURCEBOT_API_KEY` is unset, this retriever no-ops
    so the pipeline still runs (the ripgrep retriever takes over).
    """
    name = "sourcebot"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.enabled = bool(settings.sourcebot_url and settings.sourcebot_api_key)

    async def retrieve(self, repo: str, query: EnrichedQuery) -> RepoContext:
        repo_path = self.settings.repo_path(repo)
        head = _git_head(repo_path)
        empty = RepoContext(repo=repo, head_sha=head, snippets=[])

        if not self.enabled:
            return empty

        terms = list(dict.fromkeys(query.search_queries + query.code_keywords))
        terms = [t for t in terms if t and len(t) >= 3][:5]
        if not terms:
            return empty

        url = self.settings.sourcebot_url.rstrip("/") + "/api/search"
        headers = {
            "Authorization": f"Bearer {self.settings.sourcebot_api_key}",
            "Content-Type": "application/json",
        }
        # Per-repo scoping via Sourcebot's `repo:` filter. Repo name in Sourcebot
        # for a file:///data/repos/<dir> connection is the directory name.
        per_term_cap = max(2, self.settings.retrieval_max_hits // len(terms))

        async with httpx.AsyncClient(timeout=self.settings.sourcebot_timeout_seconds) as client:
            results = await asyncio.gather(
                *[
                    self._search_one(client, url, headers, term, repo, per_term_cap)
                    for term in terms
                ],
                return_exceptions=True,
            )

        snippets: list[Snippet] = []
        for r in results:
            if isinstance(r, Exception):
                log.warning("sourcebot search failed: %s", r)
                continue
            snippets.extend(r)

        seen: set[tuple[str, int]] = set()
        deduped: list[Snippet] = []
        for s in snippets:
            key = (s.path, s.line_start)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(s)
        deduped.sort(key=lambda s: -s.score)
        deduped = deduped[: self.settings.retrieval_max_hits]

        return RepoContext(repo=repo, head_sha=head, snippets=deduped)

    def _expected_repo_name(self, repo: str) -> str | None:
        """Sourcebot indexes the repos under their git origin
        (e.g. `gitlab.com/tract1/application/frontend`)."""
        project = self.settings.gitlab_projects.get(repo)
        if project:
            return f"gitlab.com/{project}"
        return None

    async def _search_one(
        self,
        client: httpx.AsyncClient,
        url: str,
        headers: dict[str, str],
        term: str,
        repo: str,
        cap: int,
    ) -> list[Snippet]:
        # Use literal mode + post-filter on the repo name. Sourcebot's `repo:`
        # filter does substring match, which would conflate `data` with
        # `data-cloud-functions`. We use it just to narrow, then filter exactly
        # on the parsed response. Cheaper than escaping for zoekt's regex engine.
        expected = self._expected_repo_name(repo)
        repo_hint = self.settings.gitlab_projects.get(repo, repo)
        body = {
            "query": f"{term} repo:{repo_hint}",
            "matches": cap,
            "contextLines": self.settings.retrieval_snippet_lines,
            "isRegexEnabled": False,
            "isCaseSensitivityEnabled": False,
        }
        try:
            resp = await client.post(url, headers=headers, json=body)
        except httpx.HTTPError as e:
            log.warning("sourcebot HTTP error for term %r: %s", term, e)
            return []
        if resp.status_code != 200:
            log.warning("sourcebot %s returned %s: %s", term, resp.status_code, resp.text[:200])
            return []
        data = resp.json()
        return self._parse_response(data, repo)

    def _parse_response(self, data: dict, repo: str) -> list[Snippet]:
        expected = self._expected_repo_name(repo)
        out: list[Snippet] = []
        for f in data.get("files", []):
            # Exact-match the repo name to avoid `data` matching `data-cloud-functions`.
            if expected and f.get("repository") != expected:
                continue
            path = (f.get("fileName") or {}).get("text") or ""
            if not path:
                continue
            for chunk in f.get("chunks", []):
                content = chunk.get("content") or ""
                if not content:
                    continue
                start_obj = chunk.get("contentStart") or {}
                line_start = int(start_obj.get("lineNumber", 1) or 1)
                line_end = line_start + content.count("\n")
                out.append(Snippet(
                    repo=repo,
                    path=path,
                    line_start=line_start,
                    line_end=line_end,
                    content=content,
                    score=2.0,  # Sourcebot results outrank ripgrep by default
                    source="sourcebot",
                ))
        return out
