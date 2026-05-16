from __future__ import annotations

import asyncio

from ..config import Settings
from ..models import EnrichedQuery, RepoContext, RetrievedContext, Snippet
from .anchors import gather_anchor_snippets
from .base import Retriever
from .ripgrep import RipgrepRetriever
from .serena import SerenaRetriever
from .sourcebot import SourcebotRetriever


_TEST_PATH_HINTS = ("/tests/", "/test/", "/__tests__/", "/spec/")


def _adjust_score_for_path(snippet: Snippet) -> None:
    """Down-rank test files. Tests are useful confirmation but they shouldn't
    crowd out the implementation file in a limited context window."""
    p = snippet.path
    is_test = (
        any(h in f"/{p}" for h in _TEST_PATH_HINTS)
        or Path_basename_starts_with_test(p)
    )
    if is_test:
        snippet.score -= 1.0


def Path_basename_starts_with_test(path: str) -> bool:
    base = path.rsplit("/", 1)[-1]
    return base.startswith("test_") or base.endswith(".test.ts") or base.endswith(".test.tsx") \
        or base.endswith(".spec.ts") or base.endswith(".spec.tsx") \
        or base.endswith("_test.py") or base.endswith(".test.js")


def build_retrievers(settings: Settings) -> list[Retriever]:
    """Order matters — first one runs first; later ones can augment."""
    return [
        RipgrepRetriever(settings),
        SourcebotRetriever(settings),
        SerenaRetriever(settings),
    ]


async def gather_context(
    repos: list[str],
    query: EnrichedQuery,
    settings: Settings,
    ticket: str | None = None,
) -> RetrievedContext:
    """Run anchor pre-retrieval (whole-file reads of any path mentioned in the
    ticket) THEN keyword retrievers in parallel. Anchors get score 10.0 so they
    rank above any keyword hit and survive context trimming."""
    retrievers = build_retrievers(settings)

    # Anchor pre-retrieval — only when we have the ticket text (always in normal use).
    anchor_ctx: dict[str, RepoContext] = {}
    if ticket is not None:
        anchor_ctx = gather_anchor_snippets(ticket, settings)

    async def for_repo(repo: str) -> RepoContext:
        merged = anchor_ctx.get(repo) or RepoContext(repo=repo, head_sha="HEAD", snippets=[])
        for r in retrievers:
            ctx = await r.retrieve(repo, query)
            # Sync head_sha if we didn't have one from anchors.
            if not anchor_ctx.get(repo) and ctx.head_sha != "HEAD":
                merged.head_sha = ctx.head_sha
            merged.snippets.extend(ctx.snippets)
        # Down-rank test files across all sources.
        for s in merged.snippets:
            _adjust_score_for_path(s)
        return merged

    target_repos = query.suspected_repos or repos
    target_repos = [r for r in target_repos if r in repos] or repos
    # Always include any repo that had anchor hits, even if enrichment didn't suspect it.
    for r in anchor_ctx:
        if r not in target_repos and r in repos:
            target_repos.append(r)

    results = await asyncio.gather(*[for_repo(r) for r in target_repos])
    total_snips = sum(len(rc.snippets) for rc in results)
    total_chars = sum(len(s.content) for rc in results for s in rc.snippets)
    return RetrievedContext(repos=results, total_snippets=total_snips, total_chars=total_chars)


__all__ = [
    "Retriever",
    "RipgrepRetriever",
    "SerenaRetriever",
    "SourcebotRetriever",
    "build_retrievers",
    "gather_context",
    "gather_anchor_snippets",
]
