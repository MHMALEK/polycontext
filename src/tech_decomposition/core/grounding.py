"""Grounded retrieval — opt-in pre-fetch of code snippets via Sourcebot search.

Single source for v1: Sourcebot's ``/api/search``. Returns typed snippets, a
ready-to-prepend grounding block, and its own metrics so the caller can see
what grounding cost in latency / snippet count / chars separately from the
adapter's own LLM call.

Callers:
- ``POST /v1/grounding/retrieve`` — see grounding output alone (and its cost).
- ``POST /v1/adapters/{name}/ask`` with ``grounded=true`` — prepend the block
  to the user query before calling the adapter.
"""
from __future__ import annotations

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


class GroundedContext(BaseModel):
    """What grounded retrieval returns. ``grounding_block`` is prompt-ready markdown."""

    snippets: list[GroundingSnippet] = Field(default_factory=list)
    grounding_block: str = ""
    metrics: GroundingMetrics = Field(default_factory=GroundingMetrics)


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

    # Scope the search to specific repos by embedding `repo:` filters in the query.
    sb_repos = effective_sourcebot_repos_for_ask(settings, repos)
    q = query.strip()
    if sb_repos:
        scope = " or ".join(f"repo:{r}" for r in sb_repos)
        q = f"{q} ({scope})"

    url = settings.sourcebot_url.rstrip("/") + "/api/search"
    headers = {
        "X-Sourcebot-Api-Key": _x_sourcebot_api_key_value(settings.sourcebot_api_key),
        "Content-Type": "application/json",
    }
    payload = {"query": q, "matches": top_k, "contextLines": context_lines}

    t0 = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
            r = await client.post(url, json=payload, headers=headers)
    except httpx.HTTPError as e:
        return GroundedContext(metrics=GroundingMetrics(
            duration_ms=int((time.monotonic() - t0) * 1000),
            sources=["sourcebot"],
            error=f"network: {type(e).__name__}: {e}",
        ))

    duration_ms = int((time.monotonic() - t0) * 1000)
    if r.status_code != 200:
        return GroundedContext(metrics=GroundingMetrics(
            duration_ms=duration_ms,
            sources=["sourcebot"],
            error=f"HTTP {r.status_code}: {r.text[:200]}",
        ))

    body = r.json() if r.content else {}
    files = body.get("files") or []
    snippets = _snippets_from_sourcebot_files(files)
    return GroundedContext(
        snippets=snippets,
        grounding_block=format_grounding_block(snippets),
        metrics=GroundingMetrics(
            duration_ms=duration_ms,
            snippet_count=len(snippets),
            total_chars=sum(len(s.content) for s in snippets),
            sources=["sourcebot"],
            sourcebot_files_seen=len(files),
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
