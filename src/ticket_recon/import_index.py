"""Pre-built per-repo import index.

For each indexed file we record:
  - which repo-relative paths it imports
  - which repo-relative paths import it (reverse edges, computed)

Cached to `outputs/.import_index/<repo>-<sha8>.json`. Invalidated when HEAD
moves. Used by:

- anchor retrieval — instant multi-hop follow without re-parsing
- deep agent — `who_imports` and `what_does_this_import` tools
"""
from __future__ import annotations

import json
import logging
import subprocess
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from .config import Settings

log = logging.getLogger(__name__)

# Cap the number of files indexed per repo (defensive).
_MAX_FILES_PER_REPO = 20_000
# Skip noise.
_EXCLUDED_DIRS = {
    ".git", "node_modules", ".venv", "venv", "__pycache__", "dist", "build",
    ".next", ".pytest_cache", ".ruff_cache", "coverage", ".turbo",
}
# Per-language file suffixes we'll attempt to parse.
_SUFFIXES = {".py", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"}


@dataclass
class FileEntry:
    path: str                          # repo-relative
    imports: list[str] = field(default_factory=list)  # repo-relative

    def to_dict(self) -> dict:
        return {"path": self.path, "imports": self.imports}


@dataclass
class RepoIndex:
    repo: str
    head_sha: str
    files: dict[str, FileEntry] = field(default_factory=dict)
    # reverse: path -> list of files that import it
    imported_by: dict[str, list[str]] = field(default_factory=dict)
    built_at: float = 0.0

    def to_dict(self) -> dict:
        return {
            "repo": self.repo,
            "head_sha": self.head_sha,
            "files": {p: e.to_dict() for p, e in self.files.items()},
            "imported_by": self.imported_by,
            "built_at": self.built_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RepoIndex":
        files = {p: FileEntry(**e) for p, e in d["files"].items()}
        return cls(
            repo=d["repo"], head_sha=d["head_sha"], files=files,
            imported_by=d.get("imported_by", {}), built_at=d.get("built_at", 0.0),
        )


def _git_head(repo_path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_path, text=True
        ).strip()
    except Exception:
        return "HEAD"


def _walk_source_files(repo_root: Path) -> Iterable[Path]:
    count = 0
    for path in repo_root.rglob("*"):
        if count >= _MAX_FILES_PER_REPO:
            break
        if not path.is_file():
            continue
        if path.suffix.lower() not in _SUFFIXES:
            continue
        if any(part in _EXCLUDED_DIRS for part in path.parts):
            continue
        count += 1
        yield path


def _cache_path(settings: Settings, repo: str, sha: str) -> Path:
    cache_dir = settings.output_dir / ".import_index"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{repo}-{sha[:8]}.json"


def build_repo_index(repo: str, settings: Settings) -> RepoIndex:
    """Build a fresh index for one repo. Slow (walks the tree); use load_or_build."""
    # Lazy import to avoid a circular dependency:
    # retrievers/anchors.py imports this module, which would re-enter
    # retrievers/__init__.py before it's done initialising.
    from .retrievers.import_follow import follow_imports

    repo_root = settings.repo_path(repo)
    sha = _git_head(repo_root)
    index = RepoIndex(repo=repo, head_sha=sha, built_at=time.time())

    for f in _walk_source_files(repo_root):
        try:
            rel = str(f.relative_to(repo_root))
        except ValueError:
            continue
        deps = follow_imports(f, repo_root)
        dep_rels: list[str] = []
        for d in deps:
            try:
                dep_rels.append(str(d.relative_to(repo_root)))
            except ValueError:
                continue
        index.files[rel] = FileEntry(path=rel, imports=dep_rels)

    # Build reverse-edge index.
    reverse: dict[str, list[str]] = defaultdict(list)
    for path, entry in index.files.items():
        for imp in entry.imports:
            reverse[imp].append(path)
    index.imported_by = dict(reverse)

    return index


def load_or_build_index(repo: str, settings: Settings) -> RepoIndex:
    """Load from cache if HEAD matches, otherwise rebuild + cache."""
    repo_root = settings.repo_path(repo)
    if not repo_root.is_dir():
        return RepoIndex(repo=repo, head_sha="HEAD")
    sha = _git_head(repo_root)
    path = _cache_path(settings, repo, sha)
    if path.exists():
        try:
            return RepoIndex.from_dict(json.loads(path.read_text()))
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            log.warning("import-index cache for %s corrupt, rebuilding: %s", repo, e)

    t0 = time.monotonic()
    index = build_repo_index(repo, settings)
    elapsed = time.monotonic() - t0
    log.info("built import index for %s: %d files in %.2fs", repo, len(index.files), elapsed)
    try:
        path.write_text(json.dumps(index.to_dict()))
    except OSError as e:
        log.warning("could not cache import index: %s", e)
    return index


def load_all_indices(settings: Settings) -> dict[str, RepoIndex]:
    return {r: load_or_build_index(r, settings) for r in settings.repos}


def transitive_imports(
    index: RepoIndex, start_path: str, max_hops: int = 3, max_files: int = 15
) -> list[str]:
    """BFS the import graph from start_path. Returns repo-relative paths
    of files reachable in up to `max_hops` hops, capped at `max_files`."""
    seen = {start_path}
    frontier = [start_path]
    out: list[str] = []
    for _hop in range(max_hops):
        next_frontier: list[str] = []
        for f in frontier:
            entry = index.files.get(f)
            if not entry:
                continue
            for imp in entry.imports:
                if imp in seen:
                    continue
                seen.add(imp)
                out.append(imp)
                next_frontier.append(imp)
                if len(out) >= max_files:
                    return out
        frontier = next_frontier
        if not frontier:
            break
    return out


def who_imports(index: RepoIndex, path: str) -> list[str]:
    """Files in the repo that directly import `path`."""
    return list(index.imported_by.get(path, []))
