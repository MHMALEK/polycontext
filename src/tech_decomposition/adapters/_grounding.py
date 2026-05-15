"""Cross-repo retrieval prelude shared across CLI-backed adapters.

Background: ``cline``/``cursor-agent``/``opencode`` all start in a single
working directory and don't know about the other repos under
``settings.repos_root``. Asking them about cross-repo behaviour gets
shallow answers because they only see one tree.

This module gives any adapter a cheap way to (a) pick relevant files from
**every** configured repo and (b) inject them as a "look here first"
preamble in the prompt. The agent then opens those files with its own
tools as needed. Originally extracted from a Aider-specific adapter so
cline/opencode/cursor can share the pattern without coupling.

The retrieval is best-effort and free of LLM calls — we hand-roll the
keyword extraction (CamelCase, snake_case, dotted paths) so a grounded
``ask`` doesn't cost an extra Gemini Flash hop just to enrich the query.
"""
from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import Iterable

from ..config import Settings
from ..models import EnrichedQuery
from ..retrievers.ripgrep import RipgrepRetriever
from ..retrievers.sourcebot import SourcebotRetriever

log = logging.getLogger(__name__)


DEFAULT_MAX_FILES = 12


# Naive code-keyword extraction. CamelCase, snake_case, dotted paths, and
# bare identifiers (≥3 chars). Same heuristic the deep-decompose engine
# uses for grep tool inputs.
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}(?:\.[A-Za-z_][A-Za-z0-9_]+)*")
_STOPWORDS = {
    "the", "this", "that", "with", "from", "into", "have", "has", "are",
    "and", "for", "what", "where", "when", "which", "how", "why", "who",
    "page", "code", "file", "files", "function", "method", "class",
    "validation", "validator", "validate",  # too generic for our codebase
}


def build_keyword_query(text: str, settings: Settings) -> EnrichedQuery:
    """Hand-roll an ``EnrichedQuery`` without spending an LLM call.

    Good enough for retriever input; the LLM-based enricher remains
    available via the legacy ``/ask`` pipeline for callers who want it.
    """
    candidates: list[str] = []
    for m in _TOKEN_RE.finditer(text):
        tok = m.group(0)
        if tok.lower() in _STOPWORDS:
            continue
        candidates.append(tok)
    seen: set[str] = set()
    keywords: list[str] = []
    for c in candidates:
        if c.lower() in seen:
            continue
        seen.add(c.lower())
        keywords.append(c)
        if len(keywords) >= 12:
            break
    return EnrichedQuery(
        summary=text[:200],
        intent="investigation",
        entities=[],
        code_keywords=keywords,
        suspected_repos=list(settings.repos),
        search_queries=keywords[:6],
        open_questions=[],
        confidence="medium",
    )


async def retrieve_context_paths(
    text: str,
    settings: Settings,
    *,
    max_files: int = DEFAULT_MAX_FILES,
) -> list[Path]:
    """Fan out across repos × (Sourcebot, ripgrep), return best-scoring file paths.

    Returns absolute paths under ``settings.repos_root``, sorted highest
    score first, deduped, capped at ``max_files``. Best-effort: a failing
    retriever logs a warning and contributes nothing rather than raising.
    """
    query = build_keyword_query(text, settings)
    sourcebot = SourcebotRetriever(settings)
    ripgrep = RipgrepRetriever(settings)

    async def _try(retriever, repo: str):
        try:
            return await retriever.retrieve(repo, query)
        except Exception as e:  # noqa: BLE001 — best-effort
            log.warning("retriever %s failed on %s: %s", retriever.name, repo, e)
            return None

    coros = []
    for repo in settings.repos:
        coros.append(_try(sourcebot, repo))
        coros.append(_try(ripgrep, repo))
    results = await asyncio.gather(*coros)

    path_score: dict[Path, float] = {}
    for ctx in results:
        if not ctx or not ctx.snippets:
            continue
        for s in ctx.snippets:
            abs_path = (settings.repo_path(s.repo) / s.path).resolve()
            if not abs_path.exists():
                continue
            if abs_path not in path_score or s.score > path_score[abs_path]:
                path_score[abs_path] = s.score

    ordered = sorted(path_score.items(), key=lambda kv: -kv[1])
    return [p for p, _ in ordered[:max_files]]


def format_grounding_block(paths: Iterable[Path], repos_root: Path) -> str:
    """Render a "look here first" preamble to inject into the prompt.

    Empty input returns an empty string so callers can unconditionally
    prepend it.
    """
    rels = []
    for p in paths:
        try:
            rels.append(str(p.relative_to(repos_root)))
        except ValueError:
            rels.append(str(p))
    if not rels:
        return ""
    lines = [
        "Relevant files retrieved across configured repos (open them first if helpful):",
    ]
    lines.extend(f"- {r}" for r in rels)
    return "\n".join(lines) + "\n\n"
