"""Grounded retrieval — opt-in pre-fetch of code snippets via Sourcebot search.

Single source for v1: Sourcebot's ``/api/search`` (a zoekt-style code search
engine). Returns typed snippets, a ready-to-prepend grounding block, and its
own metrics so the caller can see what grounding cost in latency / snippet
count / chars separately from the adapter's own LLM call.

Sourcebot's search is literal/regex — it does NOT do semantic search. Passing
it an English sentence finds nothing. So we extract code-relevant keywords
from the user's natural-language query first (CamelCase, UPPER_SNAKE,
snake_case, dotted paths, quoted strings, distinctive lowercase tokens) and
OR them in a zoekt query. The list of extracted terms is reported in
``metrics.extracted_terms`` so the caller can see what we actually searched
for.

Callers:
- ``POST /v1/grounding/retrieve`` — see grounding output alone (and its cost).
- ``POST /v1/adapters/{name}/ask`` with ``grounded=true`` — prepend the block
  to the user query before calling the adapter.
"""
from __future__ import annotations

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

# Tokens that look like words but never help a code search.
_STOPWORDS = {
    "about", "across", "after", "again", "all", "also", "and", "any",
    "are", "around", "based", "between", "both", "but", "can", "come",
    "comes", "could", "current", "currently", "describe", "describes",
    "different", "does", "doing", "each", "every", "exactly", "explain",
    "explains", "for", "from", "give", "gives", "going", "good", "has",
    "have", "help", "here", "how", "into", "just", "know", "knows",
    "like", "list", "lists", "look", "make", "many", "more", "most",
    "much", "must", "name", "names", "need", "new", "now", "off", "one",
    "only", "our", "out", "over", "platform", "please", "really", "same",
    "see", "should", "show", "shows", "some", "such", "tell", "tells",
    "than", "that", "the", "their", "them", "then", "there", "these",
    "they", "this", "those", "through", "use", "used", "uses", "using",
    "very", "want", "wants", "was", "way", "ways", "were", "what",
    "when", "where", "whether", "which", "while", "will", "with",
    "work", "works", "would", "you", "your",
    # Vague filler nouns
    "thing", "things", "time", "times", "stuff",
}


def _extract_search_terms(query: str, *, max_terms: int = 8) -> list[str]:
    """Pull code-relevant tokens from a natural-language question.

    Priority order (high → low):
      1. Quoted strings — user is signaling exact text.
      2. UPPER_SNAKE constants, CamelCase classes, snake_case identifiers,
         dotted paths (``user_model.py``, ``a.b.c``).
      3. Distinctive lowercase words (≥4 chars, not stopwords).

    Returns up to ``max_terms`` terms with duplicates collapsed
    case-insensitively. Empty list when nothing useful can be extracted.
    """
    seen: set[str] = set()
    terms: list[str] = []

    def add(token: str) -> None:
        t = token.strip()
        if not t:
            return
        low = t.lower()
        if low in _STOPWORDS or low in seen:
            return
        seen.add(low)
        terms.append(t)

    for m in _RX_QUOTED.finditer(query):
        add(m.group(1))
    for rx in (_RX_UPPER_SNAKE, _RX_CAMEL, _RX_DOTTED, _RX_SNAKE):
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


def _quote_if_needed(t: str) -> str:
    return f'"{t}"' if (" " in t or any(c in t for c in '()')) else t


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

    terms = _extract_search_terms(query)
    if not terms:
        return GroundedContext(metrics=GroundingMetrics(
            sources=["sourcebot"],
            error="no code-like keywords extracted from query",
            extracted_terms=[],
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

    async def _search(q: str) -> tuple[int, list[dict[str, Any]], int]:
        t0 = time.monotonic()
        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
            r = await client.post(
                url,
                json={"query": q, "matches": top_k, "contextLines": context_lines},
                headers=headers,
            )
        ms = int((time.monotonic() - t0) * 1000)
        if r.status_code != 200:
            return r.status_code, [], ms
        body = r.json() if r.content else {}
        return r.status_code, (body.get("files") or []), ms

    # 1st try: strict implicit-AND search — surfaces the most relevant file.
    and_query = _build_sourcebot_query(terms, mode="and")
    try:
        status, files, dur_and = await _search(and_query)
    except httpx.HTTPError as e:
        return GroundedContext(metrics=GroundingMetrics(
            sources=["sourcebot"],
            error=f"network: {type(e).__name__}: {e}",
            extracted_terms=terms,
            search_query=and_query,
        ))
    if status != 200:
        return GroundedContext(metrics=GroundingMetrics(
            duration_ms=dur_and,
            sources=["sourcebot"],
            error=f"HTTP {status}",
            extracted_terms=terms,
            search_query=and_query,
        ))

    final_query = and_query
    total_ms = dur_and

    # Fallback: if AND found nothing, try OR alternation for broader recall.
    if not files and len(terms) > 1:
        or_query = _build_sourcebot_query(terms, mode="or")
        try:
            status, files, dur_or = await _search(or_query)
        except httpx.HTTPError:
            files, dur_or = [], 0
        if status == 200:
            final_query = or_query
            total_ms += dur_or

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
