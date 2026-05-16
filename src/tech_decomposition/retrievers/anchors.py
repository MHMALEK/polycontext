"""Anchor-file retrieval: extract any file paths or filenames mentioned in the
ticket text and read them whole. These anchors get a high score so they
survive context trimming and ground the LLM's decomposition.

This runs before the keyword retrievers — Sourcebot/ripgrep may miss
the file even when it's named verbatim in the ticket (we saw this on SCRUM-18:
the DAG file was in the title but didn't appear in any retrieved snippet)."""
from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path

from ..config import Settings
from ..import_index import load_or_build_index, transitive_imports
from ..models import RepoContext, Snippet

log = logging.getLogger(__name__)

# File-like tokens: <name>.<ext> where ext is a likely source/config extension.
# We deliberately keep this conservative so we don't pull in random log/csv mentions.
_FILE_TOKEN_RE = re.compile(
    r"(?<![a-zA-Z0-9_/])"
    r"([A-Za-z0-9_\-./]+\.(?:py|ts|tsx|js|jsx|json|yml|yaml|md|sh|sql|toml))"
    r"(?![a-zA-Z0-9_])"
)
# Cap per-anchor read so a giant file doesn't blow the context budget.
_MAX_BYTES_PER_FILE = 24_000


def _git_head(repo_path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_path, text=True
        ).strip()
    except Exception:
        return "HEAD"


def extract_file_tokens(ticket: Ticket) -> list[str]:
    """Pull every file-like token out of the title + body, deduped, ordered by first appearance."""
    text = f"{ticket.title}\n{ticket.body or ''}"
    seen: set[str] = set()
    out: list[str] = []
    for m in _FILE_TOKEN_RE.finditer(text):
        tok = m.group(1).strip().strip(".,;:)")
        # filter junk: too short, contains spaces, just an extension
        if len(tok) < 5 or tok.startswith(".") or "/" not in tok and "." not in tok:
            continue
        if tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
    return out


def _find_in_repos(token: str, settings: Settings) -> list[tuple[str, Path]]:
    """Find files in any configured repo matching the token.

    Two strategies:
      1. Exact relative-path match (e.g. "src/utils/common.py" inside the repo).
      2. Basename match if the token is just a filename (e.g. "common.py").
    """
    matches: list[tuple[str, Path]] = []
    is_basename_only = "/" not in token
    target = token.lstrip("./")
    for repo in settings.repos:
        repo_path = settings.repo_path(repo)
        if not repo_path.is_dir():
            continue
        # Strategy 1: token as relative path (or trailing fragment thereof).
        direct = repo_path / target
        if direct.is_file():
            matches.append((repo, direct))
            continue
        # Strategy 2: basename search (rglob) — cap so a wildcard doesn't explode.
        basename = Path(target).name
        if not is_basename_only and basename == target:
            continue  # already tried direct
        found = list(repo_path.rglob(basename))
        # Filter out venv / node_modules / .git noise.
        excluded = {".git", "node_modules", ".venv", "venv", "__pycache__", "dist", "build", ".next"}
        for p in found[:5]:
            if any(part in excluded for part in p.parts):
                continue
            if p.is_file():
                matches.append((repo, p))
    return matches


def gather_anchor_snippets(ticket: Ticket, settings: Settings) -> dict[str, RepoContext]:
    """Returns per-repo RepoContext with anchor snippets keyed by repo name.

    Empty dict if the ticket mentions no file paths. Anchors score 10.0 so they
    rank above any keyword-retriever hit and survive context trimming."""
    tokens = extract_file_tokens(ticket)
    if not tokens:
        return {}

    log.info("anchor tokens extracted from ticket: %s", tokens)
    out: dict[str, RepoContext] = {}
    seen: set[tuple[str, str]] = set()  # (repo, rel_path)

    def _add_file(repo: str, path: Path, score: float) -> None:
        try:
            rel = str(path.relative_to(settings.repo_path(repo)))
        except ValueError:
            return
        if (repo, rel) in seen:
            return
        seen.add((repo, rel))
        try:
            data = path.read_bytes()[:_MAX_BYTES_PER_FILE].decode("utf-8", errors="replace")
        except OSError as e:
            log.warning("could not read anchor %s: %s", path, e)
            return
        line_count = data.count("\n") + 1
        snip = Snippet(
            repo=repo, path=rel,
            line_start=1, line_end=line_count,
            content=data, score=score, source="anchor",
        )
        ctx = out.get(repo)
        if ctx is None:
            ctx = RepoContext(repo=repo, head_sha=_git_head(settings.repo_path(repo)), snippets=[])
            out[repo] = ctx
        ctx.snippets.append(snip)

    for token in tokens:
        for repo, path in _find_in_repos(token, settings):
            _add_file(repo, path, score=10.0)

    # Transitive import follow using the prebuilt index (up to 3 hops).
    # Per-hop scoring: hop 1 = 8.0, hop 2 = 7.0, hop 3 = 6.0 — all still above
    # keyword-retrieval defaults (1.0-2.5), so they survive context trimming.
    for repo, ctx in list(out.items()):
        index = load_or_build_index(repo, settings)
        repo_root = settings.repo_path(repo)
        for snip in list(ctx.snippets):
            for hop, dep_rel in _walk_with_hops(index, snip.path, max_hops=3, max_files=10):
                dep_path = repo_root / dep_rel
                score = max(6.0, 9.0 - hop)
                _add_file(repo, dep_path, score=score)

    return out


def _walk_with_hops(index, start_path: str, max_hops: int, max_files: int):
    """BFS yielding (hop_depth, repo_relative_path) so callers can apply
    distance-based scoring."""
    seen = {start_path}
    frontier = [start_path]
    yielded = 0
    for hop in range(1, max_hops + 1):
        next_frontier: list[str] = []
        for f in frontier:
            entry = index.files.get(f)
            if not entry:
                continue
            for imp in entry.imports:
                if imp in seen:
                    continue
                seen.add(imp)
                yield (hop, imp)
                yielded += 1
                next_frontier.append(imp)
                if yielded >= max_files:
                    return
        frontier = next_frontier
        if not frontier:
            return
