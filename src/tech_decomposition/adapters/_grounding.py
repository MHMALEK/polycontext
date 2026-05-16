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

import httpx

from ..config import Settings
from ..models import EnrichedQuery, Snippet
from ..retrievers.sourcebot import SourcebotRetriever
from ..enrichers._cheap_impl import enrich_query
from ._repo_map import generate_repo_map
from ._local_index import LocalSemanticIndex

log = logging.getLogger(__name__)
_RERANKER_CACHE: dict[str, object] = {}


DEFAULT_MAX_FILES = 12


# Naive code-keyword extraction. CamelCase, snake_case, dotted paths, and
# bare identifiers (≥3 chars). Same heuristic the deep-decompose engine
# uses for grep tool inputs.
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}(?:\.[A-Za-z_][A-Za-z0-9_]+)*")
_STOPWORDS = {
    "the", "this", "that", "with", "from", "into", "have", "has", "are",
    "and", "for", "what", "where", "when", "which", "how", "why", "who",
    "does", "work", "works", "across", "flow", "flows",
    "page", "code", "file", "files", "function", "method", "class",
    "trace", "exact", "sources", "query", "endpoint", "path", "paths",
    "filtering", "rules", "shown", "show", "users", "cite", "line", "ranges",
    "call", "uncertainty", "explicitly", "differ", "between", "please", "out", "any",
    "validation", "validator", "validate",  # too generic for our codebase
}
_NOISE_PATH_PARTS = {
    "tests", "__tests__", "test", "fixtures", "mocks", "mock",
    "locales", "i18n", "translations", ".specify", "docs", ".github",
}
_NOISE_FILE_PATTERNS = (
    ".test.", "_test.", "test_", ".spec.",
)
_NOISE_EXTS = {".md", ".txt", ".rst"}


def build_keyword_query(text: str, settings: Settings) -> EnrichedQuery:
    """Hand-roll an ``EnrichedQuery`` without spending an LLM call.

    Good enough for retriever input; the LLM-based enricher remains
    available via the legacy ``/ask`` pipeline for callers who want it.
    """
    symbol_like: list[str] = []
    generic: list[str] = []
    for m in _TOKEN_RE.finditer(text):
        tok = m.group(0)
        if tok.lower() in _STOPWORDS:
            continue
        if _is_symbol_like(tok):
            symbol_like.append(tok)
        else:
            generic.append(tok)
    candidates = symbol_like + generic
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


def _is_symbol_like(tok: str) -> bool:
    """Prioritize code-ish tokens over plain English words."""
    return (
        "." in tok
        or "_" in tok
        or "-" in tok
        or any(c.isupper() for c in tok[1:])
    )


def _is_noise_path(path: str) -> bool:
    p = path.lower()
    if Path(p).suffix in _NOISE_EXTS:
        return True
    parts = set(Path(p).parts)
    if parts & _NOISE_PATH_PARTS:
        return True
    return any(pat in p for pat in _NOISE_FILE_PATTERNS)


def _path_quality_bonus(path: str) -> float:
    """Boost likely runtime code, penalize docs/assets/noise."""
    p = path.lower()
    bonus = 0.0
    if _is_noise_path(p):
        return -2.5
    if "/src/" in p:
        bonus += 0.5
    if "/api/" in p or "controller" in p or "service" in p:
        bonus += 0.3
    if p.endswith(".json") or "/public/" in p:
        bonus -= 0.8
    return bonus


def _source_quality_bonus(source: str) -> float:
    """Relative trust/quality signal by retriever source."""
    if source == "sourcebot":
        return 1.0
    if source == "serena":
        return 0.8
    return 0.0


def _focus_terms(text: str) -> list[str]:
    """Extract high-signal query terms for path/content matching."""
    out: list[str] = []
    seen: set[str] = set()
    for m in _TOKEN_RE.finditer(text):
        tok = m.group(0)
        low = tok.lower()
        if low in _STOPWORDS or len(low) < 4:
            continue
        if low in seen:
            continue
        seen.add(low)
        out.append(low)
        if len(out) >= 12:
            break
    return out


def _focus_match_bonus(snippet: Snippet, focus_terms: list[str]) -> float:
    if not focus_terms:
        return 0.0
    path_low = snippet.path.lower()
    content_low = snippet.content.lower()
    score = 0.0
    for t in focus_terms:
        if t in path_low:
            score += 0.5
        elif t in content_low:
            score += 0.15
    return min(score, 2.0)


async def retrieve_context_paths(
    text: str,
    settings: Settings,
    *,
    repos: list[str] | None = None,
    max_files: int = DEFAULT_MAX_FILES,
) -> list[Path]:
    """Return absolute file paths for the best grounding snippets.

    Convenience wrapper over ``retrieve_context_snippets`` that keeps the old
    path-only API for existing callers and tests.
    """
    snippets = await retrieve_context_snippets(
        text, settings, repos=repos, max_files=max_files,
    )
    return [(settings.repo_path(s.repo) / s.path).resolve() for s in snippets]


async def retrieve_context_snippets(
    text: str,
    settings: Settings,
    *,
    repos: list[str] | None = None,
    max_files: int = DEFAULT_MAX_FILES,
) -> list[Snippet]:
    """Sourcebot-first retrieval for pre-adapter grounding.

    Minimal-by-design path:
      1) Build a compact keyword query
      2) Query Sourcebot API per repo
      3) Dedupe by file + line
      4) Keep top-k simple relevance order
    """
    active_repos = [r for r in (repos or settings.repos) if r in settings.repos]
    if not active_repos:
        active_repos = list(settings.repos)
    if not active_repos:
        return []

    query = build_keyword_query(text, settings)
    focus_terms = _focus_terms(text)
    
    search_query_text = text
    if getattr(settings, "grounding_use_llm_enricher", True) and settings.gemini_api_key:
        try:
            res = await enrich_query(text, active_repos, settings)
            query = res.output
            focus_terms = query.code_keywords
            search_query_text = " ".join(query.search_queries)
        except Exception as e:
            log.warning("LLM enrichment failed, falling back to heuristic: %s", e)

    dedup: dict[tuple[str, str, int], Snippet] = {}

    # 1. Local Semantic Search (ChromaDB + AST)
    try:
        local_idx = LocalSemanticIndex()
        if local_idx.collection.count() == 0:
            local_idx.index_repositories(settings.repos_root, active_repos)
        
        local_results = local_idx.search(search_query_text, n_results=max_files)
        for r in local_results:
            meta = r['metadata']
            parts = meta['file'].split('/')
            repo_name = parts[0]
            file_path = "/".join(parts[1:]) if len(parts) > 1 else meta['file']
            
            s = Snippet(
                repo=repo_name,
                path=file_path,
                line_start=meta['start_line'] + 1,
                line_end=meta['end_line'] + 1,
                content=r['content'],
                score=0.9, # Base high score for semantic match
                source="local_chromadb"
            )
            key = (s.repo, s.path, s.line_start)
            dedup[key] = s
    except Exception as e:
        log.warning("Local semantic index search failed: %s", e)

    # 2. Sourcebot Search (if available)
    sourcebot = SourcebotRetriever(settings)
    require_sourcebot = bool(getattr(settings, "grounding_require_sourcebot", False))
    if require_sourcebot:
        if not sourcebot.enabled:
            raise RuntimeError(
                "grounding_require_sourcebot=true but SOURCEBOT_URL/API_KEY is not configured"
            )
        await _assert_sourcebot_available(settings, active_repos)
    
    if sourcebot.enabled:
        async def _try(repo: str):
            try:
                return await sourcebot.retrieve(repo, query)
            except Exception as e:  # noqa: BLE001 — best-effort
                log.warning("sourcebot failed on %s: %s", repo, e)
                return None

        results = await asyncio.gather(*[_try(repo) for repo in active_repos])

        for ctx in results:
            if not ctx or not ctx.snippets:
                continue
            for s in ctx.snippets:
                abs_path = (settings.repo_path(s.repo) / s.path).resolve()
                if not abs_path.exists():
                    continue
                if _is_noise_path(s.path):
                    continue
                score = s.score + _path_quality_bonus(s.path) + _focus_match_bonus(s, focus_terms)
                key = (s.repo, s.path, s.line_start)
                cand = s.model_copy(update={"score": score, "source": "sourcebot"})
                
                # Expand context directly from the local disk if available
                try:
                    disk_lines = abs_path.read_text(encoding="utf-8").splitlines()
                    # Expand by 5 lines in both directions for better context
                    exp_start = max(0, cand.line_start - 6)
                    exp_end = min(len(disk_lines), cand.line_end + 5)
                    cand = cand.model_copy(update={
                        "content": "\n".join(disk_lines[exp_start:exp_end]),
                        "line_start": exp_start + 1,
                        "line_end": exp_end,
                    })
                except Exception:
                    pass # fallback to raw sourcebot chunk if disk read fails

                if key not in dedup or cand.score > dedup[key].score:
                    dedup[key] = cand

    if not dedup:
        return []

    ranked = sorted(dedup.values(), key=lambda s: (-s.score, s.repo, s.path, s.line_start))
    
    # Rerank a slightly wider pool of candidates using the cross-encoder (if configured)
    candidates_for_reranking = ranked[:max_files * 2]
    reranked = _rerank_snippets(text, candidates_for_reranking, settings)
    
    return reranked[:max_files]


async def _assert_sourcebot_available(settings: Settings, repos: list[str]) -> None:
    """Fail-fast availability probe for Sourcebot when fail-closed is enabled."""
    if not settings.sourcebot_url or not settings.sourcebot_api_key:
        raise RuntimeError("sourcebot is not configured")
    probe_repo = repos[0] if repos else (settings.repos[0] if settings.repos else "")
    repo_hint = settings.gitlab_projects.get(probe_repo, probe_repo) if probe_repo else ""
    body = {
        "query": f"repo:{repo_hint}" if repo_hint else "repo:",
        "matches": 1,
        "contextLines": 0,
        "isRegexEnabled": False,
        "isCaseSensitivityEnabled": False,
    }
    headers = {
        "Authorization": f"Bearer {settings.sourcebot_api_key}",
        "Content-Type": "application/json",
    }
    url = settings.sourcebot_url.rstrip("/") + "/api/search"
    timeout = max(2.0, float(settings.sourcebot_timeout_seconds))
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, headers=headers, json=body)
    except httpx.HTTPError as e:
        raise RuntimeError(f"sourcebot unavailable: {e}") from e
    if resp.status_code != 200:
        raise RuntimeError(
            f"sourcebot unavailable: status={resp.status_code} body={resp.text[:200]}"
        )


def _rerank_snippets(text: str, snippets: list[Snippet], settings: Settings) -> list[Snippet]:
    """Optionally rerank candidates using an open-source cross-encoder.

    Enabled by setting ``GROUNDING_RERANKER_MODEL`` (e.g.
    ``BAAI/bge-reranker-base``) and installing FlagEmbedding.
    """
    model_name = (settings.grounding_reranker_model or "").strip()
    if not model_name or len(snippets) <= 1:
        return snippets

    max_candidates = max(4, int(settings.grounding_reranker_max_candidates))
    candidates = snippets[:max_candidates]
    try:
        from FlagEmbedding import FlagReranker  # type: ignore
    except Exception as e:  # noqa: BLE001
        log.warning("grounding reranker requested but FlagEmbedding unavailable: %s", e)
        return snippets

    reranker = _RERANKER_CACHE.get(model_name)
    if reranker is None:
        try:
            reranker = FlagReranker(model_name, use_fp16=False)
        except Exception as e:  # noqa: BLE001
            log.warning("failed to initialize reranker model %s: %s", model_name, e)
            return snippets
        _RERANKER_CACHE[model_name] = reranker

    try:
        pairs = []
        for s in candidates:
            excerpt = " ".join(s.content.split())
            if len(excerpt) > 1500:
                excerpt = excerpt[:1500]
            pairs.append([text, excerpt])
        scores = reranker.compute_score(pairs)
        if isinstance(scores, (int, float)):
            scores = [float(scores)]
        if not isinstance(scores, list) or len(scores) != len(candidates):
            return snippets
    except Exception as e:  # noqa: BLE001
        log.warning("grounding rerank scoring failed: %s", e)
        return snippets

    rescored: list[tuple[float, Snippet]] = []
    for idx, s in enumerate(candidates):
        try:
            score = float(scores[idx])
        except Exception:
            score = s.score
        # Preserve original retrieval score as a small tie-breaker.
        combined = score + (0.01 * s.score)
        rescored.append((combined, s))
    rescored.sort(key=lambda x: -x[0])
    reranked = [s for _, s in rescored]
    if len(snippets) > len(candidates):
        reranked.extend(snippets[len(candidates):])
    return reranked




def format_grounding_block(paths: Iterable[Path] | Iterable[Snippet], repos_root: Path) -> str:
    """Render a compact "look here first" preamble to inject into prompts.

    Empty input returns an empty string so callers can unconditionally
    prepend it.
    """
    items = list(paths)
    if not items:
        return ""

    # Support both legacy input (Path list) and richer input (Snippet list).
    if isinstance(items[0], Snippet):
        snippets: list[Snippet] = items  # type: ignore[assignment]
        active_repos = list(set(s.repo for s in snippets))
        repo_map = generate_repo_map(repos_root, active_repos)
        
        lines = [
            f"{repo_map}\n",
            "Grounded context retrieved across configured repos (use this evidence first):",
        ]
        for s in snippets:
            loc = f"{s.repo}/{s.path}:L{s.line_start}-L{s.line_end}"
            excerpt = s.content.strip()
            if len(excerpt) > 2000:
                excerpt = excerpt[:1997] + "..."
            lines.append(f"- {loc} [{s.source}]\n```\n{excerpt}\n```")
        lines.append(
            "If evidence is insufficient, say what is missing instead of speculating."
        )
        return "\n".join(lines) + "\n\n"

    rels = []
    for p in items:
        try:
            rels.append(str(p.relative_to(repos_root)))
        except ValueError:
            rels.append(str(p))
    lines = [
        "Relevant files retrieved across configured repos (open them first if helpful):",
    ]
    lines.extend(f"- {r}" for r in rels)
    return "\n".join(lines) + "\n\n"
