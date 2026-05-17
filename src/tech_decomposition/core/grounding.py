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
import os
import re
import time
from typing import Any

import httpx
from pydantic import BaseModel, Field

from ..clients.sourcebot import _x_sourcebot_api_key_value, effective_sourcebot_repos_for_ask
from ..config import Settings


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


# ---------------------------------------------------------------------------
# LLM classifier — Gemini Flash decides "needs grounding?" + extracts terms
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


async def _classify_with_llm(query: str, settings: Settings) -> GroundingDecision | None:
    """Run the Gemini Flash classifier. Returns None on any failure (caller
    falls back to regex extractor)."""
    agent = _build_classifier_agent(settings)
    if agent is None:
        return None
    try:
        result = await agent.run(query.strip())
        return result.output
    except Exception:
        return None


def _build_sourcebot_query(terms: list[str], *, mode: str = "and") -> str:
    """Compose a zoekt-style query.

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
        return " OR ".join(parts)
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def retrieve_grounded_context(
    *,
    query: str,
    settings: Settings,
    repos: list[str] | None = None,
    top_k: int = 8,
    context_lines: int = 3,
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

    if not _has_strong_terms(regex_terms):
        cls_t0 = time.monotonic()
        decision = await _classify_with_llm(query, settings)
        classifier_ms = int((time.monotonic() - cls_t0) * 1000) if decision is not None else 0
        if decision is not None:
            classifier_decision = "needs_grounding" if decision.needs_grounding else "skip"
            classifier_reason = decision.reasoning
            if not decision.needs_grounding:
                return GroundedContext(metrics=GroundingMetrics(
                    sources=["sourcebot"],
                    error="classifier: question does not need grounding",
                    extractor="llm",
                    classifier_decision=classifier_decision,
                    classifier_reason=classifier_reason,
                    classifier_ms=classifier_ms,
                ))
            # Merge LLM + regex terms (LLM first for priority, dedupe case-insensitive).
            seen_low: set[str] = set()
            merged: list[str] = []
            for t in list(decision.search_terms) + regex_terms:
                low = t.strip().lower()
                if low and low not in seen_low:
                    seen_low.add(low)
                    merged.append(t.strip())
            terms = merged
            extractor = "llm+regex"

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
    # effective_sourcebot_repos_for_ask is called for side-effect-free reference
    # in the metrics; we do NOT pass repo: filters into the query because
    # Sourcebot AND-joins multiple repo: clauses (returning 0).
    _ = effective_sourcebot_repos_for_ask(settings, repos)

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

    # Two-stage strategy:
    #   1. Strict implicit-AND search — surfaces the single most-relevant file
    #      when all terms co-occur. (e.g. "user roles role" → user_model.py)
    #   2. If AND returns 0, run each term SEPARATELY in parallel and merge.
    #      Sourcebot's OR operator silently breaks for mixed quoted/unquoted
    #      multi-term queries (verified empirically); per-term parallel
    #      searches are more robust and naturally bound the latency cost.
    and_query = _build_sourcebot_query(terms, mode="and")
    final_query = and_query
    total_ms = 0
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
        status, files, dur = await _search(client, and_query)
        total_ms += dur
        if status != 200:
            return GroundedContext(metrics=GroundingMetrics(
                duration_ms=total_ms,
                sources=["sourcebot"],
                error=f"HTTP {status}",
                extracted_terms=terms,
                search_query=and_query,
                extractor=extractor,
                classifier_decision=classifier_decision,
                classifier_reason=classifier_reason,
                classifier_ms=classifier_ms,
            ))

        if not files and len(terms) > 1:
            t_fallback = time.monotonic()
            # Expand each multi-word term to its code-style variants. A
            # question saying "Data Sharing" then yields DataSharing,
            # dataSharing, data_sharing, DATA_SHARING for the search —
            # files that use any of those naming conventions get found.
            expanded = _expand_with_variants(terms)
            quoted_terms = [_quote_if_needed(t) for t in expanded]
            per_term = await asyncio.gather(
                *[_search(client, q) for q in quoted_terms],
                return_exceptions=False,
            )
            total_ms += int((time.monotonic() - t_fallback) * 1000)
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
            files = merged
            final_query = f"per-term ({len(quoted_terms)} variants): " + " | ".join(quoted_terms)

    snippets = _snippets_from_sourcebot_files(files)
    return GroundedContext(
        snippets=snippets,
        grounding_block=format_grounding_block(snippets),
        metrics=GroundingMetrics(
            duration_ms=total_ms,
            snippet_count=len(snippets),
            total_chars=sum(len(s.content) for s in snippets),
            sources=["sourcebot"],
            sourcebot_files_seen=len(files),
            extracted_terms=terms,
            search_query=final_query,
            extractor=extractor,
            classifier_decision=classifier_decision,
            classifier_reason=classifier_reason,
            classifier_ms=classifier_ms,
        ),
    )


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
    """Flatten Sourcebot's ``files[].chunks[]`` into a flat snippet list."""
    out: list[GroundingSnippet] = []
    for f in files:
        repo = f.get("repository") or ""
        path = _file_path_text(f.get("fileName"))
        url = f.get("webUrl")
        lang = f.get("language")
        for ch in (f.get("chunks") or []):
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
