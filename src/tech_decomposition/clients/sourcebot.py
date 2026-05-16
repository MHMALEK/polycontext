"""Code Q&A via Sourcebot — primary path is ``POST /api/chat/blocking``.

This is the same endpoint Sourcebot's UI consumes, so the answer we get out is
the same the user would see in the chat UI — no SSE parsing, no Postgres
hydration, no narration scrubbing.

If that endpoint is missing on a given Sourcebot build we fall back to MCP
``ask_codebase`` over ``/api/mcp``. Set ``SOURCEBOT_DISABLE_MCP_FALLBACK=1``
(or pass ``disable_mcp_fallback=True``) to fail fast instead of falling back.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import httpx
from pydantic import BaseModel, Field

from ..config import Settings

log = logging.getLogger(__name__)


class Citation(BaseModel):
    repo: str
    path: str
    start_line: int | None = None
    end_line: int | None = None
    revision: str | None = None
    url: str | None = None


class AskMetadata(BaseModel):
    total_tokens: int | None = None
    total_input_tokens: int | None = None
    total_output_tokens: int | None = None
    total_response_time_ms: int | None = None
    model_name: str | None = None
    trace_id: str | None = None
    chat_id: str | None = None
    chat_url: str | None = None
    sources_seen: list[str] = Field(default_factory=list)
    transport: Literal["chat", "mcp", "local"] = "chat"


class AskResult(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    metadata: AskMetadata | None = None
    wall_seconds: float | None = None


class SourcebotAskError(Exception):
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


_MCP_ANSWER_STYLE_SUFFIX = (
    "\n\n"
    "Answer requirements:\n"
    "- Return only the final answer, no internal process narration.\n"
    "- Be concise and direct.\n"
    "- Include concrete file citations with paths and line ranges when available.\n"
    "- Prefer implementation files under src/; avoid specs/docs unless directly needed.\n"
)


def _x_sourcebot_api_key_value(api_key: str) -> str:
    """Match Sentinel: header value is ``sourcebot-…`` unless already prefixed."""
    return api_key if api_key.startswith("sourcebot-") else f"sourcebot-{api_key}"


def _default_sourcebot_repos(settings: Settings) -> list[str] | None:
    """Map ``GITLAB_PROJECTS`` to fully qualified Sourcebot repo names."""
    repos: list[str] = []
    for path in settings.gitlab_projects.values():
        p = str(path).strip().lstrip("/")
        if p:
            repos.append(f"gitlab.com/{p}")
    deduped = [r for i, r in enumerate(repos) if r and r not in repos[:i]]
    return deduped or None


def effective_sourcebot_repos_for_ask(
    settings: Settings,
    repos: list[str] | None,
) -> list[str] | None:
    """Resolve short repo keys (e.g. ``frontend``) to Sourcebot repo filters."""
    if not repos:
        return _default_sourcebot_repos(settings)
    out: list[str] = []
    for name in repos:
        path = settings.gitlab_projects.get(name)
        if path:
            p = str(path).strip().lstrip("/")
            out.append(f"gitlab.com/{p}")
        else:
            n = name.strip()
            if n.startswith("gitlab.com/"):
                out.append(n)
            else:
                out.append(f"gitlab.com/{n.lstrip('/')}")
    deduped = [r for i, r in enumerate(out) if r and r not in out[:i]]
    return deduped or None


async def _ask_via_chat_blocking(
    question: str,
    *,
    settings: Settings,
    repos: list[str] | None,
    max_steps: int | None,
    timeout_seconds: float,
) -> AskResult:
    """Primary path: POST /api/chat/blocking. Returns the UI-equivalent answer."""
    base = settings.sourcebot_url.rstrip("/")
    url = f"{base}/api/chat/blocking"
    headers = {
        "X-Sourcebot-Api-Key": _x_sourcebot_api_key_value(settings.sourcebot_api_key),
        "Content-Type": "application/json",
    }
    payload: dict = {"query": question.strip()}
    if repos:
        payload["repos"] = repos
    if max_steps is not None:
        if not (1 <= max_steps <= 50):
            raise SourcebotAskError("maxSteps must be between 1 and 50", status_code=400)
        payload["maxSteps"] = max_steps

    t0 = time.monotonic()
    timeout = httpx.Timeout(timeout_seconds, connect=15.0)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        resp = await client.post(url, json=payload, headers=headers)

    wall = round(time.monotonic() - t0, 2)
    if resp.status_code != 200:
        detail = resp.text[:500] if resp.text else f"HTTP {resp.status_code}"
        raise SourcebotAskError(detail, status_code=resp.status_code)

    body = resp.json()
    answer = (body.get("answer") or "").strip()
    if not answer:
        raise SourcebotAskError("empty answer from /api/chat/blocking")

    lm = body.get("languageModel") or {}
    return AskResult(
        answer=answer,
        citations=[],  # Inline in the answer markdown; no structured extraction.
        metadata=AskMetadata(
            model_name=lm.get("model") or lm.get("displayName"),
            chat_id=body.get("chatId"),
            chat_url=body.get("chatUrl"),
            transport="chat",
        ),
        wall_seconds=wall,
    )


def _extract_text(content: object) -> str:
    text_parts: list[str] = []
    for item in (content or []):
        t = getattr(item, "text", None)
        if t:
            text_parts.append(t)
    return "\n".join(text_parts).strip()


async def _ask_via_mcp_ask_codebase(
    question: str,
    *,
    settings: Settings,
    repos: list[str] | None,
    timeout_seconds: float,
) -> AskResult:
    """Fallback only — used when /api/chat/blocking is unavailable.

    MCP returns the agent's full streamed transcript including process
    narration. We return it as-is; if the primary path is missing on your
    Sourcebot build, that's the bigger problem to solve.
    """
    try:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client
    except ImportError as e:
        raise SourcebotAskError(f"mcp library not available: {e}")

    base = settings.sourcebot_url.rstrip("/")
    endpoint = f"{base}/api/mcp"
    bearer = _x_sourcebot_api_key_value(settings.sourcebot_api_key)
    headers = {"Authorization": f"Bearer {bearer}"}
    effective_repos = repos or _default_sourcebot_repos(settings)

    t0 = time.monotonic()
    async with streamablehttp_client(endpoint, headers=headers) as (read, write, *_):
        async with ClientSession(read, write) as session:
            await session.initialize()
            args: dict = {"query": question.strip() + _MCP_ANSWER_STYLE_SUFFIX}
            if effective_repos:
                args["repos"] = effective_repos
            result = await session.call_tool("ask_codebase", args)

    raw = _extract_text(result.content).strip()
    if not raw or raw.lower().startswith("failed to ask"):
        raise SourcebotAskError(raw or "MCP ask_codebase returned empty content")

    wall = round(time.monotonic() - t0, 2)
    return AskResult(
        answer=raw,
        citations=[],  # Inline in the answer markdown; no structured extraction.
        metadata=AskMetadata(transport="mcp"),
        wall_seconds=wall,
    )


async def ask_sourcebot(
    question: str,
    *,
    settings: Settings,
    repos: list[str] | None = None,
    max_steps: int | None = None,
    timeout_seconds: float = 300.0,
    disable_mcp_fallback: bool | None = None,
) -> AskResult:
    """Ask Sourcebot.

    Primary: ``POST /api/chat/blocking`` — same response the UI shows.
    Fallback: MCP ``ask_codebase`` (only on 404 from primary, unless disabled).
    """
    if not (settings.sourcebot_url and settings.sourcebot_api_key):
        raise SourcebotAskError(
            "SOURCEBOT_URL and SOURCEBOT_API_KEY must be set to use Sourcebot ask"
        )

    no_fallback = (
        settings.sourcebot_disable_mcp_fallback
        if disable_mcp_fallback is None
        else disable_mcp_fallback
    )

    effective_repos = effective_sourcebot_repos_for_ask(settings, repos)

    try:
        return await _ask_via_chat_blocking(
            question,
            settings=settings,
            repos=effective_repos,
            max_steps=max_steps,
            timeout_seconds=timeout_seconds,
        )
    except httpx.ConnectError as e:
        raise SourcebotAskError(f"cannot reach Sourcebot: {e}") from e
    except httpx.TimeoutException as e:
        raise SourcebotAskError(f"Sourcebot /api/chat/blocking timed out: {e}") from e
    except SourcebotAskError as e:
        if no_fallback or e.status_code != 404:
            raise
        log.info("Sourcebot /api/chat/blocking returned 404; falling back to MCP ask_codebase")

    try:
        return await _ask_via_mcp_ask_codebase(
            question, settings=settings, repos=effective_repos, timeout_seconds=timeout_seconds,
        )
    except Exception as fallback_err:
        raise SourcebotAskError(f"MCP fallback failed: {fallback_err}") from fallback_err


def render_ask_markdown(question: str, result: AskResult) -> str:
    lines: list[str] = []
    lines.append(f"# Q: {question.strip().splitlines()[0][:200]}")
    lines.append("")
    if result.wall_seconds is not None:
        tr = result.metadata.transport if result.metadata else "chat"
        via = {
            "chat": "Sourcebot /api/chat/blocking",
            "mcp": "Sourcebot MCP ask_codebase",
            "local": "local agent",
        }[tr]
        bits = [f"via: {via}", f"wall: {result.wall_seconds}s"]
        if result.metadata and result.metadata.model_name:
            bits.append(f"model: {result.metadata.model_name}")
        if result.metadata and result.metadata.chat_url:
            bits.append(f"chat: {result.metadata.chat_url}")
        lines.append(f"_{', '.join(bits)}_")
        lines.append("")
    lines.append("## Answer")
    lines.append(result.answer.strip() or "_(empty answer)_")
    lines.append("")
    if result.citations:
        lines.append("## Citations")
        for c in result.citations:
            label = f"{c.repo}/{c.path}" if c.repo else c.path
            suf = ""
            if c.start_line is not None and c.end_line is not None and c.end_line != c.start_line:
                suf = f":L{c.start_line}-L{c.end_line}"
            elif c.start_line is not None:
                suf = f":L{c.start_line}"
            if c.url:
                lines.append(f"- [{label}{suf}]({c.url})")
            else:
                lines.append(f"- `{label}{suf}`")
    return "\n".join(lines).rstrip() + "\n"


def write_ask_markdown(question: str, result: AskResult, settings: Settings) -> Path:
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    slug = "".join(ch for ch in question.strip()[:40] if ch.isalnum() or ch in "-_")[:40] or "ask"
    out = settings.output_dir / f"{stamp}-ask-{slug}.md"
    out.write_text(render_ask_markdown(question, result))
    return out
