from __future__ import annotations

import asyncio
import json
import re
import shutil
import subprocess
from pathlib import Path

from ..config import Settings
from ..models import EnrichedQuery, RepoContext, Snippet
from .base import Retriever


def _git_head(repo_path: Path) -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_path, text=True)
        return out.strip()
    except Exception:
        return "HEAD"


def _read_lines(path: Path, start: int, end: int) -> str:
    try:
        with path.open("r", errors="replace") as f:
            lines = f.readlines()
        # 1-indexed
        s = max(1, start) - 1
        e = min(len(lines), end)
        return "".join(lines[s:e]).rstrip("\n")
    except Exception:
        return ""


class RipgrepRetriever(Retriever):
    name = "ripgrep"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.enabled = bool(shutil.which("rg"))
        if not self.enabled:
            import logging
            logging.getLogger(__name__).warning(
                "ripgrep (rg) not on PATH — RipgrepRetriever disabled. brew install ripgrep"
            )

    async def retrieve(self, repo: str, query: EnrichedQuery) -> RepoContext:
        repo_path = self.settings.repo_path(repo)
        head = _git_head(repo_path)
        if not self.enabled or not repo_path.is_dir():
            return RepoContext(repo=repo, head_sha=head, snippets=[])

        terms = list(dict.fromkeys(query.search_queries + query.code_keywords + query.entities))
        terms = [t for t in terms if t and len(t) >= 3][:10]
        if not terms:
            return RepoContext(repo=repo, head_sha=head, snippets=[])

        snippets: list[Snippet] = []
        # Run searches concurrently per term, cap hits per term.
        per_term_cap = max(2, self.settings.retrieval_max_hits // max(1, len(terms)))
        results = await asyncio.gather(
            *[self._search_term(repo_path, t, per_term_cap) for t in terms],
            return_exceptions=False,
        )
        for hits, term in zip(results, terms):
            for h in hits:
                snip = self._build_snippet(repo, repo_path, h, term)
                if snip is not None:
                    snippets.append(snip)

        # Dedupe by (path, line_start) and cap.
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

    async def _search_term(self, repo_path: Path, term: str, cap: int) -> list[dict]:
        cmd = [
            "rg", "--json", "--no-heading", "--smart-case",
            "--max-count", str(cap), "--max-columns", "300",
            "-g", "!{node_modules,dist,build,.next,.git,*.lock,*.svg,*.png,*.jpg}",
            "-e", term, str(repo_path),
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await proc.communicate()
        hits: list[dict] = []
        for line in stdout.decode(errors="replace").splitlines():
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("type") == "match":
                hits.append(obj["data"])
        return hits

    def _build_snippet(self, repo: str, repo_path: Path, data: dict, term: str) -> Snippet | None:
        path_text = data.get("path", {}).get("text") or ""
        if not path_text:
            return None
        try:
            rel = str(Path(path_text).resolve().relative_to(repo_path.resolve()))
        except ValueError:
            rel = path_text
        line_no = int(data.get("line_number", 0) or 0)
        if line_no <= 0:
            return None
        ctx = self.settings.retrieval_snippet_lines
        start = max(1, line_no - ctx // 2)
        end = line_no + (ctx - (line_no - start))
        content = _read_lines(repo_path / rel, start, end)
        if not content:
            return None
        # crude scoring: shorter terms aren't great; symbol-y terms are good
        score = 1.0 + (1.5 if re.search(r"[A-Za-z_][A-Za-z0-9_]{4,}", term) else 0.0)
        score += 0.3 if any(c.isupper() for c in term) else 0.0
        return Snippet(
            repo=repo, path=rel, line_start=start, line_end=end,
            content=content, score=score, source="ripgrep",
        )
