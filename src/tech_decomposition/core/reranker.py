"""Cross-encoder reranking for grounding snippets (optional, local)."""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from functools import lru_cache

from .grounding import GroundingSnippet


@dataclass(frozen=True)
class RerankResult:
    snippets: list[GroundingSnippet]
    duration_ms: int
    model: str
    candidates: int


@lru_cache(maxsize=2)
def _cross_encoder(model_name: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


def _rerank_sync(
    query: str,
    snippets: list[GroundingSnippet],
    *,
    model: str,
    top_k: int,
) -> RerankResult:
    t0 = time.monotonic()
    if not snippets:
        return RerankResult(snippets=[], duration_ms=0, model=model, candidates=0)
    capped = snippets[: max(top_k * 3, top_k)]
    pairs = [(query, (s.content or "")[:2000]) for s in capped]
    scores = _cross_encoder(model).predict(pairs)
    ranked = sorted(
        zip(scores, capped, strict=True),
        key=lambda row: float(row[0]),
        reverse=True,
    )
    out = [s for _, s in ranked[:top_k]]
    return RerankResult(
        snippets=out,
        duration_ms=int((time.monotonic() - t0) * 1000),
        model=model,
        candidates=len(capped),
    )


async def rerank_snippets(
    query: str,
    snippets: list[GroundingSnippet],
    *,
    model: str,
    top_k: int,
) -> RerankResult:
    """Score snippets with a local cross-encoder; keep top_k by relevance."""
    model = (model or "").strip()
    if not model or not snippets:
        return RerankResult(
            snippets=snippets[:top_k],
            duration_ms=0,
            model=model,
            candidates=len(snippets),
        )
    if len(snippets) <= top_k:
        return RerankResult(
            snippets=snippets,
            duration_ms=0,
            model=model,
            candidates=len(snippets),
        )
    return await asyncio.to_thread(
        _rerank_sync, query, snippets, model=model, top_k=top_k,
    )
