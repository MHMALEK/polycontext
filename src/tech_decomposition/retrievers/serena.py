from __future__ import annotations

import asyncio
import json
import logging
import re
import subprocess
from pathlib import Path

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


# Serena returns text content with file paths and line ranges. We parse them out
# rather than relying on a stable JSON shape, since the MCP tools mostly emit
# human-readable text.
_PATH_LINE_RE = re.compile(
    r"^\s*(?P<path>[^\s:]+\.[a-zA-Z0-9]+):(?P<start>\d+)(?:-(?P<end>\d+))?\b"
)


class SerenaRetriever(Retriever):
    """
    Connects to a Serena MCP server over SSE and calls `search_for_pattern`.

    Serena is configured with one project root (REPOS_ROOT) so it sees all 4
    repos as a single project. We post-filter results back into per-repo
    snippets by matching the path prefix.

    If `SERENA_URL` is unset OR the connection fails, this retriever no-ops.
    """
    name = "serena"

    # Path Serena sees inside its container (set by docker-compose volume mount).
    PROJECT_PATH_IN_CONTAINER = "/workspaces/projects"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.enabled = bool(settings.serena_url)

    async def retrieve(self, repo: str, query: EnrichedQuery) -> RepoContext:
        repo_path = self.settings.repo_path(repo)
        head = _git_head(repo_path)
        empty = RepoContext(repo=repo, head_sha=head, snippets=[])

        if not self.enabled:
            return empty

        terms = list(dict.fromkeys(query.code_keywords + query.search_queries))
        terms = [t for t in terms if t and len(t) >= 3][:4]
        if not terms:
            return empty

        try:
            results = await asyncio.wait_for(
                self._call_serena(terms, repo),
                timeout=self.settings.serena_timeout_seconds,
            )
        except asyncio.TimeoutError:
            log.warning("serena timed out after %ss", self.settings.serena_timeout_seconds)
            return empty
        except Exception as e:
            log.warning("serena call failed: %s", e)
            return empty

        seen: set[tuple[str, int]] = set()
        deduped: list[Snippet] = []
        for s in results:
            key = (s.path, s.line_start)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(s)

        return RepoContext(repo=repo, head_sha=head, snippets=deduped)

    async def _call_serena(self, terms: list[str], repo: str) -> list[Snippet]:
        from mcp import ClientSession  # type: ignore[import-not-found]
        from mcp.client.sse import sse_client  # type: ignore[import-not-found]

        snippets: list[Snippet] = []
        async with sse_client(self.settings.serena_url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                try:
                    await session.call_tool(
                        "activate_project",
                        arguments={"project": self.PROJECT_PATH_IN_CONTAINER},
                    )
                except Exception as e:
                    log.warning("serena activate_project failed: %s", e)
                    return []
                for term in terms:
                    try:
                        result = await session.call_tool(
                            "search_for_pattern",
                            arguments={
                                "substring_pattern": term,
                                "relative_path": repo,
                                "max_answer_chars": 8_000,
                                "context_lines_before": self.settings.retrieval_snippet_lines // 2,
                                "context_lines_after": self.settings.retrieval_snippet_lines // 2,
                            },
                        )
                    except Exception as e:
                        log.warning("serena search_for_pattern(%r) failed: %s", term, e)
                        continue
                    snippets.extend(self._parse_tool_result(result, repo))
                    if len(snippets) >= self.settings.retrieval_max_hits:
                        break
        return snippets

    def _parse_tool_result(self, result, repo: str) -> list[Snippet]:
        out: list[Snippet] = []
        contents = getattr(result, "content", []) or []
        for item in contents:
            text = getattr(item, "text", None)
            if not text:
                continue
            # Serena results are sometimes JSON, sometimes structured text.
            # Try JSON first; fall back to line-by-line parsing.
            parsed = self._parse_json_payload(text, repo)
            if parsed:
                out.extend(parsed)
                continue
            out.extend(self._parse_text_payload(text, repo))
        return out[: self.settings.serena_max_results_per_query]

    def _parse_json_payload(self, text: str, repo: str) -> list[Snippet]:
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return []
        if not isinstance(data, dict):
            return []
        out: list[Snippet] = []
        # Serena's search_for_pattern returns:
        #   {"<path>": ["  >  N:line\n... N+1:line\n...", ...]}
        # Each value is a list of multi-line snippet strings. Match lines start
        # with "> N:" and context lines with "... N:".
        for path, matches in data.items():
            rel = self._strip_repo_prefix(path, repo)
            if rel is None or not isinstance(matches, list):
                continue
            for m in matches:
                if isinstance(m, str):
                    snip = self._snippet_from_serena_string(repo, rel, m)
                    if snip:
                        out.append(snip)
                elif isinstance(m, dict):
                    out.extend(self._snippet_from_match(repo, rel, m))
        return out

    @staticmethod
    def _snippet_from_serena_string(repo: str, rel: str, raw: str) -> Snippet | None:
        """Parse a Serena snippet string like:
           "  >   0:from foo import bar\n...   1:next line\n  >   2:another match"
        Returns the cleaned content + first/last line numbers."""
        lines = raw.splitlines()
        cleaned: list[str] = []
        first = last = None
        for line in lines:
            if not line:
                continue
            tag = line[:5]
            if tag.lstrip().startswith((">", ".")):
                rest = line[5:] if len(line) > 5 else ""
                if ":" in rest:
                    num_str, _, content = rest.partition(":")
                    try:
                        n = int(num_str.strip())
                    except ValueError:
                        cleaned.append(rest); continue
                    if first is None:
                        first = n
                    last = n
                    cleaned.append(content)
                else:
                    cleaned.append(rest)
            else:
                cleaned.append(line)
        content = "\n".join(cleaned).strip("\n")
        if not content or first is None:
            return None
        return Snippet(
            repo=repo, path=rel,
            line_start=first + 1,  # Serena uses 0-indexed; our convention is 1-indexed
            line_end=(last or first) + 1,
            content=content, score=2.5, source="serena",
        )

    def _snippet_from_match(self, repo: str, path: str, m: dict) -> list[Snippet]:
        rel = self._strip_repo_prefix(path, repo)
        if rel is None:
            return []
        line = int(m.get("line", m.get("line_number", m.get("startLine", 1))) or 1)
        content = m.get("content") or m.get("snippet") or m.get("text") or ""
        if not content:
            return []
        line_end = line + content.count("\n")
        return [Snippet(
            repo=repo, path=rel, line_start=line, line_end=line_end,
            content=content, score=2.5, source="serena",
        )]

    def _parse_text_payload(self, text: str, repo: str) -> list[Snippet]:
        out: list[Snippet] = []
        lines = text.splitlines()
        i = 0
        while i < len(lines):
            line = lines[i]
            m = _PATH_LINE_RE.match(line)
            if not m:
                i += 1
                continue
            path = m.group("path")
            rel = self._strip_repo_prefix(path, repo)
            if rel is None:
                i += 1
                continue
            start = int(m.group("start"))
            end = int(m.group("end") or start)
            # Capture body until the next path:line header or blank section break.
            body: list[str] = []
            j = i + 1
            while j < len(lines):
                if _PATH_LINE_RE.match(lines[j]):
                    break
                body.append(lines[j])
                j += 1
            content = "\n".join(body).strip("\n")
            if content:
                out.append(Snippet(
                    repo=repo, path=rel, line_start=start, line_end=end or start,
                    content=content, score=2.5, source="serena",
                ))
            i = j
        return out

    def _strip_repo_prefix(self, path: str, repo: str) -> str | None:
        """Serena sees paths under the project root, e.g. 'traceability/src/x.py'.
        Strip the leading '<repo>/' so we end up with repo-relative paths matching
        ripgrep / Sourcebot results."""
        if path.startswith(f"{repo}/"):
            return path[len(repo) + 1:]
        # Some serena outputs may already be repo-relative (e.g. when the project
        # root IS the repo). Accept those as-is.
        if not path.startswith(("/", ".")) and "/" in path:
            return path
        return None
