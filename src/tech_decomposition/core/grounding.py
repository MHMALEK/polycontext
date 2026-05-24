"""Grounded retrieval — opt-in pre-fetch of code snippets via Sourcebot search.

Single source: Sourcebot's ``/api/search`` (a zoekt-style code search engine).
Returns typed snippets, a ready-to-prepend grounding block, and its own
metrics so the caller can see what grounding cost in latency / snippet
count / chars separately from the adapter's own LLM call.

Two query-planning paths:

1. **LLM classifier (preferred)** — a Gemini Flash pass decides whether the
   question would benefit from pre-fetched code at all, and if yes, extracts
   1-6 code-relevant search terms tuned for zoekt. Handles multi-word
   domain phrases ("Data Sharing", "Farm Name", "supplier portal") that
   regex misses. Cost: ~$0.0003, ~500ms.

2. **Regex fallback** — used when ``GEMINI_API_KEY`` isn't set. Pulls
   CamelCase, UPPER_SNAKE, snake_case, dotted paths, quoted strings, and
   distinctive lowercase tokens. Works for symbol-heavy queries; fails on
   pure English questions.

Callers:
- ``POST /v1/grounding/retrieve`` — see grounding output alone (and its cost).
- ``POST /v1/adapters/{name}/ask`` with ``grounded=true`` — prepend the block
  to the user query before calling the adapter.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json as _json
import os
import re
import time
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field

from ..clients.sourcebot import _x_sourcebot_api_key_value, effective_sourcebot_repos_for_ask
from ..config import Settings
from .serena_client import SerenaError, SerenaMcpClient


class GroundingSnippet(BaseModel):
    """One code chunk surfaced by grounded retrieval."""

    repo: str = ""
    path: str = ""
    start_line: int | None = None
    end_line: int | None = None
    content: str = ""
    url: str | None = None
    language: str | None = None


class GroundingMetrics(BaseModel):
    """Telemetry for one grounding call, kept separate from the adapter's LLM metrics."""

    duration_ms: int = 0
    snippet_count: int = 0
    total_chars: int = 0
    sources: list[str] = Field(default_factory=list)
    sourcebot_files_seen: int = 0
    serena_hits: int = 0
    serena_ms: int = 0
    serena_error: str | None = None
    error: str | None = None
    extracted_terms: list[str] = Field(
        default_factory=list,
        description="Code-relevant tokens pulled from the NL query and sent to search.",
    )
    search_query: str = Field(
        default="",
        description="The literal query string sent to Sourcebot — useful for debugging.",
    )
    extractor: str = Field(
        default="regex",
        description='Which extractor produced the terms: "llm" or "regex".',
    )
    classifier_decision: str = Field(
        default="",
        description='LLM classifier verdict: "needs_grounding" / "skip" / "" if not used.',
    )
    classifier_reason: str = Field(
        default="",
        description="Brief reasoning from the classifier — visible in the UI/logs.",
    )
    classifier_ms: int = 0
    rerank_ms: int = 0
    rerank_model: str = ""
    rerank_candidates: int = 0
    per_repo_cap_applied: int = 0
    per_repo_cap_dropped: int = 0
    expanded_snippets: int = 0
    expansion_added_chars: int = 0
    serena_symbol_lookups: int = 0
    serena_symbol_hits: int = 0
    serena_symbol_ms: int = 0


class GroundingDecision(BaseModel):
    """Output of the LLM-based pre-grounding step.

    The classifier decides BOTH:
      1. whether this question would benefit from pre-fetched code at all
         (skips conceptual / architecture / opinion questions), and
      2. if yes, which 1-6 code-relevant terms to search for.
    """

    needs_grounding: bool = Field(
        description=(
            "True if pre-fetching code snippets would likely help answer this "
            "code-Q&A question. False for conceptual, opinion, or "
            "architecture-level questions where the LLM's general reasoning "
            "is more useful than retrieved code."
        ),
    )
    search_terms: list[str] = Field(
        default_factory=list,
        description=(
            "1-6 short, code-relevant search terms tuned for zoekt-style "
            "literal matching. Prefer: class names, function names, file "
            "stems, UPPER_SNAKE constants, domain nouns (e.g. 'Farm Name', "
            "'Data Sharing'). AVOID generic English (the, and, what, etc.). "
            "Multi-word phrases are OK and will be searched as-is. Empty "
            "list when needs_grounding=false."
        ),
    )
    reasoning: str = Field(
        default="",
        description="One short sentence (under 25 words) explaining the decision.",
    )


class GroundedContext(BaseModel):
    """What grounded retrieval returns. ``grounding_block`` is prompt-ready markdown."""

    snippets: list[GroundingSnippet] = Field(default_factory=list)
    grounding_block: str = ""
    metrics: GroundingMetrics = Field(default_factory=GroundingMetrics)


# ---------------------------------------------------------------------------
# Keyword extraction — turn an English question into Sourcebot search terms
# ---------------------------------------------------------------------------

_RX_CAMEL = re.compile(r"\b[A-Z][a-zA-Z0-9]*(?:[A-Z][a-zA-Z0-9]+)+\b")
_RX_UPPER_SNAKE = re.compile(r"\b[A-Z][A-Z0-9_]{2,}[A-Z0-9]\b")
_RX_SNAKE = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+){1,}\b")
_RX_DOTTED = re.compile(r"\b[a-zA-Z][a-zA-Z0-9_]*\.[a-zA-Z][a-zA-Z0-9_.]+\b")
_RX_QUOTED = re.compile(r'["\']([^"\']{2,40})["\']')
_RX_LOWER_WORD = re.compile(r"\b[a-z][a-z0-9]{3,}\b")
# Title Case multi-word: "Data Sharing", "Farm Name", "Reference ID", "Data Uploader".
# Sentinel users phrase things this way constantly; CamelCase regex misses these
# because each word is a separate capitalized token.
_RX_TITLE_PHRASE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-zA-Z0-9]+){1,3}\b")

# Tokens that look like words but never help a code search.
_STOPWORDS = {
    # short / function words (catch in leading-strip of Title Case phrases too:
    # "On Data Sharing" → "Data Sharing")
    "a", "an", "as", "at", "be", "by", "do", "if", "in", "is", "it", "of",
    "on", "or", "to", "we", "you",
    # generic English filler
    "about", "across", "after", "again", "all", "also", "and", "any",
    "are", "around", "based", "between", "both", "but", "can", "come",
    "comes", "could", "current", "currently", "describe", "describes",
    "different", "does", "doing", "each", "every", "exactly", "explain",
    "explains", "for", "from", "give", "gives", "going", "good", "has",
    "have", "help", "here", "how", "into", "just", "know", "knows",
    "like", "look", "make", "many", "more", "most",
    "much", "must", "need", "new", "now", "off", "one",
    "only", "our", "out", "over", "platform", "please", "really", "same",
    "see", "should", "show", "shows", "some", "such", "tell", "tells",
    "than", "that", "the", "their", "them", "then", "there", "these",
    "they", "this", "those", "through", "use", "used", "uses", "using",
    "very", "want", "wants", "was", "way", "ways", "were", "what",
    "when", "where", "whether", "which", "while", "will", "with",
    "work", "works", "would", "your",
    # Vague filler nouns
    "thing", "things", "time", "times", "stuff",
    # NOTE: deliberately keeping "name", "names", "list", "lists", "type"
    # OUT — they're meaningful suffixes in code-domain phrases like
    # "Farm Name", "Reference Type", "Supplier List". The leading-strip
    # logic in _extract_search_terms() handles the case where a question
    # starts with "List the ..." by removing only LEADING stopwords.
}


def _extract_search_terms(query: str, *, max_terms: int = 8) -> list[str]:
    """Pull code-relevant tokens from a natural-language question.

    Priority order (high → low):
      1. Quoted strings — user is signaling exact text.
      2. UPPER_SNAKE constants, CamelCase classes, snake_case identifiers,
         dotted paths (``user_model.py``, ``a.b.c``).
      3. Title Case multi-word phrases ("Data Sharing", "Farm Name").
      4. Distinctive lowercase words (≥4 chars, not stopwords).

    Returns up to ``max_terms`` terms with duplicates collapsed
    case-insensitively. Empty list when nothing useful can be extracted.
    """
    seen: set[str] = set()
    terms: list[str] = []

    def add(token: str) -> None:
        t = token.strip()
        # Trim leading stopword tokens from multi-word phrases:
        # "On Data Sharing" → "Data Sharing". DO NOT trim trailing — words
        # like "Name" / "Type" / "List" are in the English-stopword set but
        # are meaningful suffixes in code-domain phrases ("Farm Name",
        # "User Type").
        if " " in t:
            parts = t.split()
            while parts and parts[0].lower() in _STOPWORDS:
                parts.pop(0)
            t = " ".join(parts)
        if not t or len(t) < 2:
            return
        low = t.lower()
        if low in _STOPWORDS or low in seen:
            return
        seen.add(low)
        terms.append(t)

    for m in _RX_QUOTED.finditer(query):
        add(m.group(1))
    for rx in (_RX_UPPER_SNAKE, _RX_CAMEL, _RX_DOTTED, _RX_SNAKE, _RX_TITLE_PHRASE):
        for m in rx.finditer(query):
            add(m.group(0))
        if len(terms) >= max_terms:
            return terms[:max_terms]
    # Lowercase fallback only if we don't have many symbol-like terms yet.
    if len(terms) < 3:
        for m in _RX_LOWER_WORD.finditer(query):
            if len(terms) >= max_terms:
                break
            add(m.group(0))
    return terms[:max_terms]


def _has_strong_terms(terms: list[str]) -> bool:
    """True if the regex produced at least one symbol-class term, OR ≥3 distinct
    terms total. Used to gate the LLM classifier — if regex already has a
    strong set, we skip the 5-8s Flash call.

    Symbol-class = contains an uppercase letter, underscore, dot, or space
    (i.e. came from one of the higher-priority regexes, not the lowercase
    fallback).
    """
    if not terms:
        return False
    has_symbol = any(re.search(r"[A-Z_.\s]", t) for t in terms)
    return has_symbol or len(terms) >= 3


def _generate_code_variants(phrase: str) -> list[str]:
    """For a multi-word phrase, generate code-style compound variants.

    Sourcebot's search is case-insensitive on individual tokens but treats
    multi-word phrases (``"Data Sharing"``) as a phrase search. Files that
    use the compound form (``DataSharing``, ``data_sharing``) won't match.
    Emitting variants explicitly widens recall when the question phrases
    a concept differently from how the code names it.

    Returns PascalCase / camelCase / snake_case / UPPER_SNAKE_CASE forms,
    deduped. Single-word inputs return ``[]`` (already covered by literal
    matching).
    """
    parts = [p for p in phrase.split() if p]
    if len(parts) < 2:
        return []
    lower = [p.lower() for p in parts]
    pascal = "".join(w.capitalize() for w in lower)
    camel = lower[0] + "".join(w.capitalize() for w in lower[1:])
    snake = "_".join(lower)
    upper = snake.upper()
    seen: set[str] = set()
    out: list[str] = []
    for v in (pascal, camel, snake, upper):
        if v.lower() not in seen:
            seen.add(v.lower())
            out.append(v)
    return out


def _adjacent_lowercase_pair_variants(terms: list[str]) -> list[str]:
    """For pairs of adjacent simple-lowercase terms in ``terms``, generate
    compound variants. Catches the "user roles" → ``UserRoles`` /
    ``user_roles`` case where the question used English but the codebase
    names things with CamelCase or snake_case.

    Only pairs that are BOTH simple lowercase (no spaces, no underscores,
    no uppercase) qualify — multi-word phrases already get variants
    directly via ``_generate_code_variants``.
    """
    def _is_simple_lower(t: str) -> bool:
        return bool(t) and t == t.lower() and " " not in t and "_" not in t and "." not in t

    out: list[str] = []
    seen: set[str] = set()
    for i in range(len(terms) - 1):
        a, b = terms[i], terms[i + 1]
        if not (_is_simple_lower(a) and _is_simple_lower(b)):
            continue
        for v in _generate_code_variants(f"{a} {b}"):
            if v.lower() not in seen:
                seen.add(v.lower())
                out.append(v)
    return out


def _expand_with_variants(terms: list[str], *, max_total: int = 14) -> list[str]:
    """Append code-style variants for each multi-word term + adjacent-pair
    variants for simple lowercase tokens.

    Original term order is preserved (variants follow their source phrase
    in the output, then pair-variants at the end). Capped at ``max_total``
    to avoid blowing up the per-term parallel-search fan-out.
    """
    seen: set[str] = set()
    out: list[str] = []
    for t in terms:
        low = t.lower()
        if low not in seen:
            seen.add(low)
            out.append(t)
        for v in _generate_code_variants(t):
            if v.lower() not in seen:
                seen.add(v.lower())
                out.append(v)
        if len(out) >= max_total:
            return out[:max_total]
    # Adjacent-pair variants for lowercase singletons ("user roles" → UserRoles).
    for v in _adjacent_lowercase_pair_variants(terms):
        if v.lower() not in seen and len(out) < max_total:
            seen.add(v.lower())
            out.append(v)
    return out[:max_total]


def _quote_if_needed(t: str) -> str:
    return f'"{t}"' if (" " in t or any(c in t for c in '()')) else t


async def _merge_per_term_files(
    client: httpx.AsyncClient,
    search_fn,
    terms: list[str],
    *,
    top_k: int,
) -> tuple[list[dict[str, Any]], int]:
    """Run per-term (with variants) searches in parallel; merge unique files."""
    t_fallback = time.monotonic()
    expanded = _expand_with_variants(terms)
    # Apply the same noise filters as the AND-mode query — the per-term
    # fallback is where most noise sneaks in because each lone term matches
    # broadly. Without filters, "user" alone returns 100s of test files.
    quoted_terms = [f"{_quote_if_needed(t)} {_ZOEKT_NOISE_FILTERS}" for t in expanded]
    per_term = await asyncio.gather(
        *[search_fn(client, q) for q in quoted_terms],
        return_exceptions=False,
    )
    extra_ms = int((time.monotonic() - t_fallback) * 1000)

    def _file_id(f: dict[str, Any]) -> str:
        return f"{f.get('repository','')}::{_file_path_text(f.get('fileName'))}"

    seen: set[str] = set()
    merged: list[dict[str, Any]] = []
    for _status, found, _ms in per_term:
        for f in found:
            fid = _file_id(f)
            if fid in seen:
                continue
            seen.add(fid)
            merged.append(f)
            if len(merged) >= top_k:
                break
        if len(merged) >= top_k:
            break
    return merged, extra_ms


@dataclass
class _SourcebotFetchResult:
    snippets: list[GroundingSnippet]
    files: list[dict[str, Any]]
    duration_ms: int
    final_query: str
    error: str | None = None
    early_context: GroundedContext | None = None


async def _sourcebot_fetch(
    *,
    url: str,
    headers: dict[str, str],
    terms: list[str],
    top_k: int,
    context_lines: int,
    broad: bool,
    extractor: str,
    classifier_decision: str,
    classifier_reason: str,
    classifier_ms: int,
) -> _SourcebotFetchResult:
    async def _search(client: httpx.AsyncClient, q: str) -> tuple[int, list[dict[str, Any]], int]:
        t0 = time.monotonic()
        try:
            r = await client.post(
                url,
                json={"query": q, "matches": top_k, "contextLines": context_lines},
                headers=headers,
            )
        except httpx.HTTPError:
            return 0, [], int((time.monotonic() - t0) * 1000)
        ms = int((time.monotonic() - t0) * 1000)
        if r.status_code != 200:
            return r.status_code, [], ms
        body = r.json() if r.content else {}
        return r.status_code, (body.get("files") or []), ms

    def _file_id(f: dict[str, Any]) -> str:
        return f"{f.get('repository','')}::{_file_path_text(f.get('fileName'))}"

    and_query = _build_sourcebot_query(terms, mode="and")
    final_query = and_query
    total_ms = 0
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
        status, files, dur = await _search(client, and_query)
        total_ms += dur
        if status != 200:
            return _SourcebotFetchResult(
                snippets=[],
                files=[],
                duration_ms=total_ms,
                final_query=and_query,
                error=f"HTTP {status}",
                early_context=GroundedContext(metrics=GroundingMetrics(
                    duration_ms=total_ms,
                    sources=["sourcebot"],
                    error=f"HTTP {status}",
                    extracted_terms=terms,
                    search_query=and_query,
                    extractor=extractor,
                    classifier_decision=classifier_decision,
                    classifier_reason=classifier_reason,
                    classifier_ms=classifier_ms,
                )),
            )

        if not files and len(terms) > 1:
            merged, extra = await _merge_per_term_files(client, _search, terms, top_k=top_k)
            total_ms += extra
            files = merged
            final_query = f"per-term fallback ({len(terms)} terms)"
        elif broad and terms:
            and_files = list(files)
            merged, extra = await _merge_per_term_files(client, _search, terms, top_k=top_k)
            total_ms += extra
            seen: set[str] = set()
            files = []
            for f in and_files + merged:
                fid = _file_id(f)
                if fid in seen:
                    continue
                seen.add(fid)
                files.append(f)
                if len(files) >= top_k:
                    break
            final_query = f"broad: AND + per-term ({len(terms)} terms)"

    return _SourcebotFetchResult(
        snippets=_snippets_from_sourcebot_files(files),
        files=files,
        duration_ms=total_ms,
        final_query=final_query,
    )


async def _serena_fetch(
    *,
    terms: list[str],
    settings: Settings,
    cap: int,
) -> tuple[list[GroundingSnippet], int, str | None]:
    t0 = time.monotonic()
    try:
        snippets = await _serena_search_terms(terms=terms, settings=settings, cap=cap)
        return snippets, int((time.monotonic() - t0) * 1000), None
    except SerenaError as e:
        return [], int((time.monotonic() - t0) * 1000), str(e)[:300]
    except Exception as e:
        return [], int((time.monotonic() - t0) * 1000), f"{type(e).__name__}: {e}"[:300]


def _snippet_dedup_key(snippet: GroundingSnippet) -> str:
    return f"{snippet.repo}::{snippet.path}"


def normalize_repo_name(raw: str, *, known_repos: list[str]) -> str:
    """Map Sourcebot's URL-style repo ('gitlab.com/<org>/<group>/<name>') to
    the trailing segment that matches one of ``known_repos`` on disk.

    Sourcebot indexes by the full GitLab path; but our local clones, the
    Serena project layout, and the ``expected_files`` recall labels all use
    just the leaf repo name (``traceability``, ``frontend``). Without this
    normalization snippet.repo doesn't match anything downstream — window
    expansion can't find the file on disk and the recall metric can't
    substring-match against the labels.

    Strategy: prefer the longest known_repos entry that ``raw`` ends with;
    fall back to ``raw``'s last path segment so unconfigured repos still
    render sanely.
    """
    if not raw:
        return ""
    cleaned = raw.replace("\\", "/").strip("/")
    for r in sorted(known_repos or [], key=len, reverse=True):
        if not r:
            continue
        if cleaned == r or cleaned.endswith(f"/{r}"):
            return r
    return cleaned.rsplit("/", 1)[-1]


def normalize_sourcebot_snippets(
    snippets: list[GroundingSnippet],
    *,
    known_repos: list[str],
) -> list[GroundingSnippet]:
    """Rewrite ``snippet.repo`` on Sourcebot results to match the local repo
    layout. Returns a new list; inputs are not mutated."""
    out: list[GroundingSnippet] = []
    for s in snippets:
        if not s.repo:
            out.append(s)
            continue
        norm = normalize_repo_name(s.repo, known_repos=known_repos)
        if norm == s.repo:
            out.append(s)
        else:
            out.append(s.model_copy(update={"repo": norm}))
    return out


def _merge_snippets(
    primary: list[GroundingSnippet],
    secondary: list[GroundingSnippet],
) -> list[GroundingSnippet]:
    seen = {_snippet_dedup_key(s) for s in primary}
    out = list(primary)
    for s in secondary:
        key = _snippet_dedup_key(s)
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def apply_per_repo_cap(
    snippets: list[GroundingSnippet],
    *,
    cap: int,
) -> list[GroundingSnippet]:
    """Trim ``snippets`` so each ``repo`` contributes at most ``cap`` entries.

    Order-preserving: keeps the first ``cap`` snippets per repo as ranked by
    the upstream caller. Snippets with empty ``repo`` are kept un-capped (rare,
    and the bound is unknowable). ``cap <= 0`` disables the cap.
    """
    if cap <= 0:
        return snippets
    counts: dict[str, int] = {}
    out: list[GroundingSnippet] = []
    for s in snippets:
        key = s.repo or ""
        if key and counts.get(key, 0) >= cap:
            continue
        counts[key] = counts.get(key, 0) + 1
        out.append(s)
    return out


def expand_snippets_from_disk(
    snippets: list[GroundingSnippet],
    *,
    repos_root: Any,
    context_lines: int,
    max_lines: int,
) -> list[GroundingSnippet]:
    """Re-read each snippet's source file and widen the window by ``context_lines``
    above and below, capped at ``max_lines`` total.

    Skips snippets without ``start_line`` or whose source can't be opened. Any
    I/O error on a single snippet falls back to the original — the caller
    never raises. Returns a new list; inputs are not mutated.
    """
    if context_lines <= 0:
        return snippets
    from pathlib import Path

    root = Path(str(repos_root))
    out: list[GroundingSnippet] = []
    for s in snippets:
        if not s.start_line or not s.path:
            out.append(s)
            continue
        rel = s.path.lstrip("/")
        candidate = (root / s.repo / rel) if s.repo else (root / rel)
        try:
            if not candidate.is_file():
                out.append(s)
                continue
            text = candidate.read_text(errors="replace")
        except OSError:
            out.append(s)
            continue
        lines = text.splitlines()
        n = len(lines)
        if n == 0:
            out.append(s)
            continue
        hit_start = max(1, s.start_line)
        hit_end = s.end_line if (s.end_line and s.end_line >= hit_start) else hit_start
        hit_end = min(hit_end, n)
        new_start = max(1, hit_start - context_lines)
        new_end = min(n, hit_end + context_lines)
        if max_lines > 0:
            span = new_end - new_start + 1
            if span > max_lines:
                # Shrink each wing toward the hit until we fit ``max_lines``.
                # The hit interval [hit_start, hit_end] is preserved verbatim;
                # if it alone exceeds max_lines we don't truncate it.
                excess = span - max_lines
                top_wing = hit_start - new_start
                bot_wing = new_end - hit_end
                trim_top = min(top_wing, excess // 2)
                trim_bot = min(bot_wing, excess - trim_top)
                leftover = excess - trim_top - trim_bot
                if leftover > 0 and top_wing > trim_top:
                    trim_top += min(top_wing - trim_top, leftover)
                new_start += trim_top
                new_end -= trim_bot
        content = "\n".join(lines[new_start - 1:new_end])
        out.append(s.model_copy(update={
            "content": content,
            "start_line": new_start,
            "end_line": new_end,
        }))
    return out


def _repo_and_path_from_serena(path: str, settings: Settings) -> tuple[str, str]:
    """Map Serena workspace-relative paths to (repo, path-within-repo)."""
    p = path.lstrip("/").replace("\\", "/")
    known = list(settings.repos or [])
    for repo in sorted(known, key=len, reverse=True):
        prefix = f"{repo}/"
        if p.startswith(prefix):
            return repo, p[len(prefix):]
        if p == repo:
            return repo, ""
    parts = p.split("/", 1)
    if parts and (settings.repos_root / parts[0]).is_dir():
        return parts[0], parts[1] if len(parts) > 1 else ""
    return "", p


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

_CLASSIFIER_SYSTEM = """\
You are a pre-grounding planner for a code-Q&A agent. Decide:

1. Would pre-fetching code snippets help answer this question?
2. If yes, what 1-6 short search terms should we send to a zoekt-style
   code-search index?

YES (needs_grounding=true) — questions that map to specific code:
- factual lookups: roles, constants, field definitions
- validation rules: "what's the regex for X?", "is Y allowed?"
- behavior: "what happens when X?", "how does Y flow?"
- enumeration: "list all endpoints that...", "what are the types of..."
- file/symbol-named questions: "what does UserModel do?"

NO (needs_grounding=false) — questions where retrieval adds noise:
- opinion / suggestion: "what would you recommend?", "should we...?"
- pure conceptual: "what is OAuth?", "explain JWT in general"
- meta questions about the codebase as a whole, with no specific anchor

SEARCH TERMS:
- 1-6 terms, each 1-3 words.
- Prefer: class names, function names, file stems, UPPER_SNAKE constants,
  multi-word domain nouns (e.g. "Farm Name", "Data Sharing", "supplier portal").
- AVOID generic English (the, and, what, current, describe, list).
- Multi-word terms are OK — the search engine handles them as phrases.
- Skip terms that would match millions of files ("user", "id", "data" alone).

REASONING: one short sentence on why you chose this decision.
"""


# Module-level Agent cache. Schema generation + model lookup take ~7s on cold
# build for pydantic-ai's GeminiModel; cache by (api_key_fingerprint, model)
# so the second call onward pays only the network round-trip (~500ms).
_AGENT_CACHE: dict[tuple[str, str], Any] = {}


def _build_classifier_agent(settings: Settings) -> Any | None:
    """Build (or fetch from cache) the pydantic-ai classifier agent."""
    if not settings.gemini_api_key:
        return None
    try:
        from pydantic_ai import Agent
        from pydantic_ai.models.gemini import GeminiModel
        from pydantic_ai.settings import ModelSettings
    except ImportError:
        return None

    os.environ.setdefault("GEMINI_API_KEY", settings.gemini_api_key)
    model_name = (settings.enrich_model or "gemini-2.5-flash").strip()
    key = (settings.gemini_api_key[-8:], model_name)
    cached = _AGENT_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        agent = Agent(
            model=GeminiModel(model_name),
            output_type=GroundingDecision,
            system_prompt=_CLASSIFIER_SYSTEM,
            model_settings=ModelSettings(temperature=0.0),
        )
        _AGENT_CACHE[key] = agent
        return agent
    except Exception:
        return None


_RX_JUNK_TERM = re.compile(
    r"[=:;{}]"                       # code-statement punctuation
    r"|^\s*(class|def|import|from)\s"  # Python statements
    r"|\\n|^\s*\d+:"                 # multi-line or line-prefixed grep output
)
_RX_TEST_TERM = re.compile(
    r"\b(test_|Test[A-Z]|conftest|pytest|fixture|mock_|Mock[A-Z])",
)


def _sanitize_llm_terms(raw_terms: list[str]) -> list[str]:
    """Filter junk out of LLM-produced search terms.

    The classifier sometimes emits code statements ("class Foo(StrEnum):"),
    test-fixture symbols ("test_client_data_uploader"), template strings
    ("{TEST_RATE_LIMIT}/minute"), or runaway phrases (>40 chars). Sourcebot
    treats each term as a phrase OR-search, so a handful of noise terms
    dominates the retrieved candidate pool and drowns the real answer files.
    """
    out: list[str] = []
    for t in raw_terms or []:
        t = (t or "").strip()
        if not t:
            continue
        if len(t) > 40:
            continue
        if "\n" in t or "\\n" in t:
            continue
        if _RX_JUNK_TERM.search(t):
            continue
        if _RX_TEST_TERM.search(t):
            continue
        out.append(t)
    return out


async def _classify_with_llm(query: str, settings: Settings) -> GroundingDecision | None:
    """Run the Gemini Flash classifier. Returns None on any failure (caller
    falls back to regex extractor). Output terms are passed through
    ``_sanitize_llm_terms`` so downstream search isn't drowned by noise."""
    agent = _build_classifier_agent(settings)
    if agent is None:
        return None
    try:
        result = await agent.run(query.strip())
        decision = result.output
        if decision is not None and decision.search_terms:
            decision = decision.model_copy(update={
                "search_terms": _sanitize_llm_terms(decision.search_terms),
            })
        return decision
    except Exception:
        return None


# Zoekt-style negative filters appended to every Sourcebot search to suppress
# the noise that dominates our pre-rerank pool. From bakeoff observation:
# .specify/ docs (Tract's planning markdown), tests, mocks, and translation
# JSON were taking 8 of every 10 retrieved slots — burying the real code.
# These filters apply to every query (AND-mode and per-term fallback).
_ZOEKT_NOISE_FILTERS = (
    "-lang:markdown "
    "-f:test "          # tests/, *_test.py, *.test.tsx — anything with "test" in path
    "-f:specify "       # .specify/ — Tract's planning markdown directory
    "-f:fixtures "
    "-f:mocks "
    "-f:locales "       # frontend/public/locales/ — translation JSON
    "-f:openapi"        # generated OpenAPI dumps that mention every domain word
)


def _build_sourcebot_query(terms: list[str], *, mode: str = "and") -> str:
    """Compose a zoekt-style query, with noise filters appended.

    ``mode='and'`` joins terms with spaces (implicit AND — strictest match,
    surfaces the most relevant file when terms co-occur).
    ``mode='or'``  joins with ``OR`` — broader recall, used as a fallback
    when the AND form returned zero files.

    Repo scoping is intentionally omitted: in practice multiple ``repo:``
    filters are AND-ed together (no file is in two repos) so the search
    returns 0. The user's Sourcebot instance only indexes their repos
    anyway, so scoping rarely earns its keep.
    """
    if not terms:
        return ""
    parts = [_quote_if_needed(t) for t in terms]
    if mode == "or":
        body = " OR ".join(parts)
    else:
        body = " ".join(parts)
    return f"{body} {_ZOEKT_NOISE_FILTERS}"


RetrievalMode = Literal["default", "broad"]


async def retrieve_grounded_context(
    *,
    query: str,
    settings: Settings,
    repos: list[str] | None = None,
    top_k: int = 8,
    context_lines: int = 3,
    retrieval_mode: RetrievalMode = "default",
    extra_terms: list[str] | None = None,
) -> GroundedContext:
    """Call Sourcebot's ``/api/search`` and return typed snippets + a grounding block.

    Empty / error responses never raise — they return an empty context with the
    failure noted in ``metrics.error`` so callers can degrade gracefully.
    """
    if not (settings.sourcebot_url and settings.sourcebot_api_key):
        return GroundedContext(metrics=GroundingMetrics(
            sources=["sourcebot:unconfigured"],
            error="SOURCEBOT_URL or SOURCEBOT_API_KEY not set",
        ))

    # 1. Plan: regex first (instant, free), LLM classifier only as fallback
    #    when regex returns weak terms. The LLM adds 5-8s per call so we only
    #    pay for it when the question clearly needs help — usually pure
    #    English questions with no symbols or Title Case phrases the regex
    #    can grab.
    classifier_ms = 0
    classifier_decision = ""
    classifier_reason = ""

    regex_terms = _extract_search_terms(query)
    extractor = "regex"
    terms = regex_terms

    broad = retrieval_mode == "broad"
    # The LLM hallucinates symbols that don't exist ("WCAG", "L126") — those
    # are *harmless* for symbol lookup (Serena.find_symbol returns empty for
    # non-existent symbols) but they'd corrupt keyword search if merged into
    # ``terms``. Hence we split: regex terms go to Sourcebot keyword search;
    # symbol-shaped LLM suggestions go to Serena.find_symbol where bad guesses
    # are silently dropped. We get the upside (real symbol guesses like
    # ``JwtService``, ``UserModel``) without the downside.
    llm_symbol_suggestions: list[str] = []
    # Run the LLM whenever we'd benefit: broad mode (extra coverage from
    # symbol hints) OR when regex itself is weak (need help producing any
    # plan). Skip otherwise to save the ~500ms Flash call.
    if broad or not _has_strong_terms(regex_terms):
        cls_t0 = time.monotonic()
        decision = await _classify_with_llm(query, settings)
        classifier_ms = int((time.monotonic() - cls_t0) * 1000) if decision is not None else 0
        if decision is not None:
            classifier_decision = "needs_grounding" if decision.needs_grounding else "skip"
            classifier_reason = decision.reasoning
            if not decision.needs_grounding and not extra_terms and retrieval_mode != "broad":
                return GroundedContext(metrics=GroundingMetrics(
                    sources=["sourcebot"],
                    error="classifier: question does not need grounding",
                    extractor="llm",
                    classifier_decision=classifier_decision,
                    classifier_reason=classifier_reason,
                    classifier_ms=classifier_ms,
                ))
            # Set aside LLM symbol suggestions for the symbol-graph step.
            # Do NOT merge into keyword terms — that's what created the
            # hallucination noise in past runs.
            llm_symbol_suggestions = [
                t for t in decision.search_terms if _RX_SYMBOL_LIKE.match(t.strip())
            ]
            if regex_terms:
                extractor = "regex+llm_symbols"
            elif decision.search_terms:
                # Fall back to LLM keyword terms only if regex was empty —
                # better a noisy plan than no plan.
                seen_low: set[str] = set()
                merged: list[str] = []
                for t in decision.search_terms:
                    low = t.strip().lower()
                    if low and low not in seen_low:
                        seen_low.add(low)
                        merged.append(t.strip())
                terms = merged
                extractor = "llm"

    if extra_terms:
        seen_low = {t.lower() for t in terms}
        for t in extra_terms:
            tt = t.strip()
            if tt and tt.lower() not in seen_low:
                seen_low.add(tt.lower())
                terms.append(tt)
        if extra_terms:
            extractor = f"{extractor}+boost"

    # Hard cap on the term list before variant-expansion. Each term becomes its
    # own Sourcebot OR-search in the per-term fallback path; a dozen+ terms
    # produces a low-precision union dominated by markdown/spec/test matches
    # that out-rank the real answer files. 8 is empirically the sweet spot.
    MAX_TERMS_TOTAL = 8
    if len(terms) > MAX_TERMS_TOTAL:
        terms = terms[:MAX_TERMS_TOTAL]

    if not terms:
        return GroundedContext(metrics=GroundingMetrics(
            sources=["sourcebot"],
            error="no search terms produced (classifier and regex both empty)",
            extractor=extractor,
            classifier_decision=classifier_decision,
            classifier_reason=classifier_reason,
            classifier_ms=classifier_ms,
        ))

    url = settings.sourcebot_url.rstrip("/") + "/api/search"
    headers = {
        "X-Sourcebot-Api-Key": _x_sourcebot_api_key_value(settings.sourcebot_api_key),
        "Content-Type": "application/json",
    }
    _ = effective_sourcebot_repos_for_ask(settings, repos)

    serena_enabled = bool((settings.serena_url or "").strip())

    sb_task = asyncio.create_task(
        _sourcebot_fetch(
            url=url,
            headers=headers,
            terms=terms,
            top_k=top_k,
            context_lines=context_lines,
            broad=broad,
            extractor=extractor,
            classifier_decision=classifier_decision,
            classifier_reason=classifier_reason,
            classifier_ms=classifier_ms,
        ),
    )
    serena_task = (
        asyncio.create_task(
            _serena_fetch(
                terms=terms,
                settings=settings,
                cap=top_k,
            ),
        )
        if serena_enabled
        else None
    )

    sb = await sb_task
    if sb.error and sb.early_context is not None:
        if serena_task is not None:
            serena_task.cancel()
        return sb.early_context

    serena_snippets: list[GroundingSnippet] = []
    serena_ms = 0
    serena_error: str | None = None
    if serena_task is not None:
        serena_snippets, serena_ms, serena_error = await serena_task

    # Sourcebot returns 'gitlab.com/<org>/.../traceability' as the repo name;
    # rewrite to the local repo leaf so downstream consumers (window expansion,
    # recall scoring, per-repo cap) see consistent identifiers.
    sb_snippets_norm = normalize_sourcebot_snippets(
        sb.snippets, known_repos=list(settings.repos or []),
    )

    # Symbol-graph expansion via Serena's LSP find_symbol: for symbol-shaped
    # query terms (UserRoles, NODE_NAME_PATTERN, RolesChecker, get_identity),
    # add the file containing each symbol's *definition*. Text search often
    # misses these — the file may not contain the user's vocabulary even
    # though it carries the answer.
    sym_snippets: list[GroundingSnippet] = []
    sym_lookups = 0
    sym_hits = 0
    sym_ms = 0
    if serena_enabled:
        # Pool: query-derived symbol-shaped terms + LLM symbol suggestions.
        # Cap at 6 total; LLM suggestions go first because they're the most
        # likely to find a definition the user couldn't name from their
        # query alone (e.g. "JWT" the user types vs. "get_identity" the
        # LLM guesses).
        symbol_pool = list(llm_symbol_suggestions) + _symbol_candidates(terms, cap=6)
        symbol_candidates = _symbol_candidates(symbol_pool, cap=6)
        if symbol_candidates:
            sym_lookups = len(symbol_candidates)
            sym_snippets, sym_ms = await _serena_symbol_lookup(
                symbols=symbol_candidates, settings=settings,
            )
            sym_hits = len(sym_snippets)

    snippets = _merge_snippets(sb_snippets_norm, serena_snippets)
    sources = ["sourcebot"]
    serena_hits = len(serena_snippets)
    if serena_hits:
        sources.append("serena")
    if sym_snippets:
        snippets = _merge_snippets(snippets, sym_snippets)
        sources.append("serena_lsp")

    total_ms = sb.duration_ms

    # Per-repo cap BEFORE rerank: keeps cross-repo diversity in the candidate
    # pool so a single noisy repo can't dominate the top-k after reranking.
    per_repo_cap_applied = 0
    per_repo_cap_dropped = 0
    per_repo_cap = int(getattr(settings, "grounding_per_repo_cap", 0) or 0)
    if per_repo_cap > 0 and snippets:
        before = len(snippets)
        snippets = apply_per_repo_cap(snippets, cap=per_repo_cap)
        per_repo_cap_applied = per_repo_cap
        per_repo_cap_dropped = before - len(snippets)

    rerank_ms = 0
    rerank_model = ""
    rerank_candidates = len(snippets)
    rerank_name = (settings.grounding_reranker_model or "").strip()
    if rerank_name and snippets:
        from .reranker import rerank_snippets

        cap = min(len(snippets), max(top_k, settings.grounding_reranker_max_candidates))
        rr = await rerank_snippets(
            query,
            snippets,
            model=rerank_name,
            top_k=min(top_k, cap),
        )
        snippets = rr.snippets
        rerank_ms = rr.duration_ms
        rerank_model = rr.model
        rerank_candidates = rr.candidates

    # Window-expand the final selection — done AFTER rerank so we only pay file
    # I/O on snippets we'll actually send to the model.
    expanded_count = 0
    expansion_added_chars = 0
    expand_lines = int(getattr(settings, "grounding_expand_context_lines", 0) or 0)
    if expand_lines > 0 and snippets:
        before_chars = sum(len(s.content or "") for s in snippets)
        expanded = expand_snippets_from_disk(
            snippets,
            repos_root=settings.repos_root,
            context_lines=expand_lines,
            max_lines=int(getattr(settings, "grounding_expand_max_lines", 0) or 0),
        )
        after_chars = sum(len(s.content or "") for s in expanded)
        expanded_count = sum(
            1 for o, e in zip(snippets, expanded, strict=True)
            if (o.content or "") != (e.content or "")
        )
        expansion_added_chars = max(0, after_chars - before_chars)
        snippets = expanded

    return GroundedContext(
        snippets=snippets,
        grounding_block=format_grounding_block(snippets),
        metrics=GroundingMetrics(
            duration_ms=total_ms + serena_ms,
            snippet_count=len(snippets),
            total_chars=sum(len(s.content) for s in snippets),
            sources=sources,
            sourcebot_files_seen=len(sb.files),
            serena_hits=serena_hits,
            serena_ms=serena_ms,
            serena_error=serena_error,
            extracted_terms=terms,
            search_query=sb.final_query,
            extractor=extractor,
            classifier_decision=classifier_decision,
            classifier_reason=classifier_reason,
            classifier_ms=classifier_ms,
            rerank_ms=rerank_ms,
            rerank_model=rerank_model,
            rerank_candidates=rerank_candidates,
            per_repo_cap_applied=per_repo_cap_applied,
            per_repo_cap_dropped=per_repo_cap_dropped,
            expanded_snippets=expanded_count,
            expansion_added_chars=expansion_added_chars,
            serena_symbol_lookups=sym_lookups,
            serena_symbol_hits=sym_hits,
            serena_symbol_ms=sym_ms,
        ),
    )


def _already_seen_ids(snippets: list[GroundingSnippet]) -> set[str]:
    """File-level dedup keys (repo + path) for snippets already in the result."""
    return {_snippet_dedup_key(s) for s in snippets if s.path}


# Symbol-shaped terms: PascalCase / UPPER_SNAKE / dotted / snake_case_with_2plus_segments.
# Plain English words are excluded — they aren't valid LSP name_path patterns.
_RX_SYMBOL_LIKE = re.compile(
    r"^("
    r"[A-Z][a-zA-Z0-9]*[a-z][A-Z][a-zA-Z0-9]*"          # PascalCase ≥ 2 segments (UserRoles)
    r"|[A-Z][A-Z0-9_]{2,}[A-Z0-9]"                       # UPPER_SNAKE
    r"|[a-z][a-z0-9]*(?:_[a-z0-9]+){1,}"                 # snake_case_with_2_plus
    r"|[a-zA-Z][a-zA-Z0-9_]*\.[a-zA-Z][a-zA-Z0-9_.]+"    # dotted (foo.bar)
    r")$",
)


def _symbol_candidates(terms: list[str], *, cap: int = 6) -> list[str]:
    """Pick terms shaped like code symbols, in priority order, capped."""
    seen: set[str] = set()
    out: list[str] = []
    for t in terms:
        if not _RX_SYMBOL_LIKE.match(t):
            continue
        low = t.lower()
        if low in seen:
            continue
        seen.add(low)
        out.append(t)
        if len(out) >= cap:
            break
    return out


async def _serena_symbol_lookup(
    *,
    symbols: list[str],
    settings: Settings,
    cap_per_symbol: int = 3,
    total_cap: int = 10,
) -> tuple[list[GroundingSnippet], int]:
    """Use Serena's LSP find_symbol to surface definition files for symbol-
    shaped query terms. Returns (snippets, duration_ms).

    For each symbol we get the relative_path of its definition site —
    *exactly* the file the model needs. Text search can miss this when the
    user's vocabulary doesn't appear in the code (e.g. asking "what regex
    validates farm name" surfaces ``NODE_NAME_PATTERN`` only because the
    regex extracted "NODE_NAME_PATTERN"; LSP finds its definition file
    deterministically).
    """
    if not symbols:
        return [], 0
    t0 = time.monotonic()
    client = SerenaMcpClient(
        url=settings.serena_url,
        api_key=settings.serena_api_key,
        timeout=settings.serena_timeout_seconds,
    )
    project = str(settings.repos_root)
    calls: list[tuple[str, dict[str, Any]]] = [
        ("activate_project", {"project": project}),
    ]
    for sym in symbols:
        calls.append((
            "find_symbol",
            {
                "name_path_pattern": sym,
                # Substring matching lets "UserRole" match the real "UserRoles"
                # symbol; without this Serena requires exact name_path equality
                # and most query-derived terms miss by a character.
                "substring_matching": True,
                "include_body": False,
                "depth": 0,
                # LSP SymbolKinds: 5=Class, 6=Method, 11=Interface, 12=Function,
                # 14=Constant. Excludes Variable (13), Property (7), String,
                # Number, etc. — those hit on incidental occurrences of the
                # name in unrelated code rather than real definitions.
                "include_kinds": [5, 6, 11, 12, 14],
            },
        ))

    try:
        raw_results = await client.batch_calls(calls)
    except Exception:
        return [], int((time.monotonic() - t0) * 1000)

    seen_paths: set[str] = set()
    out: list[GroundingSnippet] = []
    for sym, raw in zip(symbols, raw_results[1:], strict=True):
        if len(out) >= total_cap:
            break
        try:
            matches = _json.loads(raw or "[]")
        except Exception:
            continue
        if not isinstance(matches, list):
            continue
        per_symbol = 0
        # Real LSP kinds (Class/Function/Method/Constant/Variable) point at
        # actual definitions in code; "Package" is Serena's filesystem-walk
        # fallback (directory paths) which never carries the answer body.
        # Sort so real kinds come first; keep the rest only if budget allows.
        sorted_matches = sorted(
            (m for m in matches if isinstance(m, dict)),
            key=lambda m: 0 if m.get("kind") in {"Class", "Function", "Method", "Constant", "Variable"} else 1,
        )
        for m in sorted_matches:
            rel = m.get("relative_path")
            if not isinstance(rel, str) or not rel:
                continue
            rel = rel.replace("\\", "/").lstrip("/")
            if rel in seen_paths:
                continue
            seen_paths.add(rel)
            # Pull repo/path apart against settings.repos so window expansion
            # can find the file on disk.
            repo, rel_path = _repo_and_path_from_serena(rel, settings)
            # Start_line=1; the post-rerank expand_snippets_from_disk widens
            # to ±N lines so the model sees enough of the definition.
            out.append(GroundingSnippet(
                repo=repo,
                path=rel_path or rel,
                start_line=1,
                end_line=None,
                content=f"<symbol {sym} defined here — body via window expansion>",
                language=_language_from_path(rel),
            ))
            per_symbol += 1
            if per_symbol >= cap_per_symbol or len(out) >= total_cap:
                break
    return out, int((time.monotonic() - t0) * 1000)


async def _serena_search_terms(
    *,
    terms: list[str],
    settings: Settings,
    cap: int,
) -> list[GroundingSnippet]:
    """Call Serena.search_for_pattern once per term in one MCP session."""
    if cap <= 0 or not terms:
        return []
    client = SerenaMcpClient(
        url=settings.serena_url,
        api_key=settings.serena_api_key,
        timeout=settings.serena_timeout_seconds,
    )
    project = str(settings.repos_root)
    calls: list[tuple[str, dict[str, Any]]] = [
        ("activate_project", {"project": project}),
    ]
    for t in terms:
        calls.append(
            (
                "search_for_pattern",
                {
                    "substring_pattern": t,
                    "context_lines_before": 1,
                    "context_lines_after": 6,
                },
            ),
        )
    raw_results = await client.batch_calls(calls)
    out: list[GroundingSnippet] = []
    for term, raw in zip(terms, raw_results[1:], strict=True):
        if not raw or raw.strip().startswith("Error:"):
            continue
        for snippet in _parse_serena_search_result(raw, term=term, settings=settings):
            out.append(snippet)
            if len(out) >= cap:
                return out
    return out


def _parse_serena_search_result(
    raw: str, *, term: str, settings: Settings,
) -> list[GroundingSnippet]:
    """Serena.search_for_pattern returns a JSON-stringified
    ``{path: [block, ...]}`` mapping. Convert to GroundingSnippets.

    Each block is a text run like ``"  >  N:line\\n... M:context_line\\n..."``.
    We capture the first hit-line per file (good enough for grounding signal —
    multiple chunks per file inflate the prompt without adding signal)."""
    try:
        parsed = _json.loads(raw)
    except Exception:
        return []
    if not isinstance(parsed, dict):
        return []
    out: list[GroundingSnippet] = []
    for path, blocks in parsed.items():
        if not isinstance(path, str) or not isinstance(blocks, list) or not blocks:
            continue
        first = blocks[0]
        if not isinstance(first, str):
            continue
        start_line = _serena_first_lineno(first)
        repo, rel_path = _repo_and_path_from_serena(path, settings)
        out.append(GroundingSnippet(
            repo=repo,
            path=rel_path or path,
            start_line=start_line,
            end_line=None,
            content=first,
            language=_language_from_path(path),
        ))
    return out


_RX_SERENA_LINENO = re.compile(r"^\s*>?\s*(\d+):", re.MULTILINE)


def _serena_first_lineno(block: str) -> int | None:
    m = _RX_SERENA_LINENO.search(block)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _language_from_path(path: str) -> str | None:
    p = path.lower()
    if p.endswith(".py"): return "python"
    if p.endswith((".ts", ".tsx")): return "typescript"
    if p.endswith((".js", ".jsx")): return "javascript"
    if p.endswith(".go"): return "go"
    if p.endswith(".rs"): return "rust"
    return None


def _file_path_text(file_name: Any) -> str:
    """Sourcebot's ``fileName`` is either a string or ``{text, matchRanges}``."""
    if isinstance(file_name, str):
        return file_name
    if isinstance(file_name, dict):
        text = file_name.get("text")
        if isinstance(text, str):
            return text
    return ""


def _chunk_start_line(chunk: dict[str, Any]) -> int | None:
    """Sourcebot's ``contentStart`` is ``{byteOffset, lineNumber, column}``."""
    cs = chunk.get("contentStart") or {}
    if isinstance(cs, dict):
        ln = cs.get("lineNumber")
        if isinstance(ln, int):
            return ln
    return None


def _snippets_from_sourcebot_files(files: list[dict[str, Any]]) -> list[GroundingSnippet]:
    """Flatten Sourcebot's ``files[].chunks[]`` into a flat snippet list.

    Keep only the FIRST chunk per file. Sourcebot returns up to N chunks per
    file matched, and multiple chunks of the same file occupy multiple
    snippet slots — which (a) starves the candidate pool of cross-repo
    diversity and (b) is redundant once window expansion widens the kept
    chunk by ±30 lines (the additional chunks likely fall inside that
    window anyway). Mirrors the Serena parser's "first hit per file" rule.
    """
    out: list[GroundingSnippet] = []
    for f in files:
        repo = f.get("repository") or ""
        path = _file_path_text(f.get("fileName"))
        url = f.get("webUrl")
        lang = f.get("language")
        chunks = f.get("chunks") or []
        if not chunks:
            continue
        ch = chunks[0]
        content = ch.get("content") or ""
        start = _chunk_start_line(ch)
        end = start + len(content.splitlines()) - 1 if start else None
        out.append(GroundingSnippet(
            repo=repo, path=path, start_line=start, end_line=end,
            content=content, url=url, language=lang,
        ))
    return out


def format_grounding_block(snippets: list[GroundingSnippet]) -> str:
    """Render snippets as a markdown block to prepend to an adapter prompt."""
    if not snippets:
        return ""
    lines = ["## Relevant code (grounded retrieval)", ""]
    for s in snippets:
        label = f"{s.repo}/{s.path}" if s.repo else s.path
        if s.start_line:
            label += f":L{s.start_line}"
            if s.end_line and s.end_line != s.start_line:
                label += f"-L{s.end_line}"
        lines.append(f"### `{label}`")
        if s.url:
            lines.append(f"[view in Sourcebot]({s.url})")
        lines.append("```" + (s.language or ""))
        lines.append(s.content.rstrip())
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


def build_grounded_prompt(*, grounding_block: str, query: str) -> str:
    """Concatenate the grounding block + a separator + the user's question."""
    if not grounding_block.strip():
        return query
    return (
        f"{grounding_block.rstrip()}\n\n"
        f"---\n\n"
        f"Use the code snippets above as primary evidence. Answer the question:\n\n"
        f"{query}"
    )
