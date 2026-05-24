"""Retrieval coverage scoring — decides prefetch-only vs agent fallback."""
from __future__ import annotations

from dataclasses import dataclass

from .grounding import GroundedContext, GroundingSnippet
from .router import RouteDecision, TaskTier


@dataclass(frozen=True)
class CoverageScore:
    score: float  # 0..1
    snippet_count: int
    total_chars: int
    unique_repos: int
    sufficient: bool
    reason: str


def assess_coverage(
    ctx: GroundedContext,
    route: RouteDecision,
    *,
    min_snippets: int = 3,
    min_chars: int = 800,
) -> CoverageScore:
    """Score whether prefetched snippets are enough for single-shot synthesis."""
    snippets = ctx.snippets or []
    n = len(snippets)
    chars = sum(len(s.content or "") for s in snippets)
    repos = {s.repo for s in snippets if s.repo}

    if n == 0:
        return CoverageScore(
            score=0.0,
            snippet_count=0,
            total_chars=0,
            unique_repos=0,
            sufficient=False,
            reason="no snippets retrieved",
        )

    # Weighted components
    snippet_part = min(1.0, n / max(min_snippets, 1))
    char_part = min(1.0, chars / max(min_chars, 1))
    repo_part = min(1.0, len(repos) / 2.0) if route.tier in (TaskTier.COMPLEX, TaskTier.TRACE) else 1.0
    serena_bonus = 0.1 if ctx.metrics.serena_hits > 0 else 0.0

    score = min(1.0, 0.45 * snippet_part + 0.35 * char_part + 0.15 * repo_part + serena_bonus)

    # Tier-specific thresholds
    if route.tier == TaskTier.ENUMERATION:
        threshold = 0.35
        sufficient = n >= 2 and chars >= 400
    elif route.tier == TaskTier.SIMPLE:
        threshold = 0.40
        sufficient = n >= 2 and score >= threshold
    elif route.tier == TaskTier.TRACE:
        threshold = 0.55
        sufficient = n >= min_snippets and score >= threshold
    else:  # COMPLEX / decompose
        threshold = 0.50
        sufficient = n >= min_snippets and score >= threshold

    if not sufficient:
        reason = f"coverage {score:.2f} < tier threshold {threshold:.2f} ({n} snippets, {chars} chars)"
    else:
        reason = f"coverage ok ({score:.2f}, {n} snippets, {chars} chars, {len(repos)} repos)"

    return CoverageScore(
        score=score,
        snippet_count=n,
        total_chars=chars,
        unique_repos=len(repos),
        sufficient=sufficient,
        reason=reason,
    )


def snippet_paths(snippets: list[GroundingSnippet]) -> set[str]:
    """Normalized repo/path keys for overlap checks."""
    out: set[str] = set()
    for s in snippets:
        if s.path:
            key = f"{s.repo}/{s.path}" if s.repo else s.path
            out.add(key.replace("\\", "/"))
    return out
