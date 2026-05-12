"""Code Q&A via Sourcebot's built-in MCP server (`/api/mcp` endpoint).

Sourcebot v4.17+ ships an MCP server over Streamable HTTP at `/api/mcp`.
Among the tools it exposes is `ask_codebase` — Sourcebot's internal AI
agent that autonomously searches and analyzes the codebase to answer
natural-language questions.

This module connects to Sourcebot's MCP and forwards questions to
`ask_codebase`. It is a thin wrapper — Sourcebot does all the reasoning
internally.

DEPLOYMENT STATE (as of writing):
- `/api/mcp` works against our OSS Sourcebot. `list_language_models`
  returns the two Gemini models we registered in `config/sourcebot/config.json`.
- `ask_codebase` matches requested models by provider + model + displayName.
  We therefore call `list_language_models` first and pass the exact returned
  object back into `ask_codebase`.

LEGACY ENDPOINTS WE TRIED (all 404 on our deployment, kept here so the
next debugger doesn't repeat the journey):
- POST /api/ask        — sentinel uses this. Doesn't exist in 4.17.1.
- POST /api/chat       — exists, schema validates, returns silent 404 inside.
- POST /api/ask_codebase — doesn't exist (the MCP tool is named the same
  but is NOT exposed as a REST endpoint).

WORKAROUND: when this module's `ask_sourcebot()` returns a failure, the
CLI auto-falls-back to `local_ask.py`, which runs the same agentic loop
locally with Pydantic AI + Gemini. Same toolset, costs more, but always
works."""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from .config import Settings

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
    sources_seen: list[str] = Field(default_factory=list)


class AskResult(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    metadata: AskMetadata | None = None
    wall_seconds: float | None = None


class SourcebotAskError(Exception):
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


async def ask_sourcebot(
    question: str,
    *,
    settings: Settings,
    repos: list[str] | None = None,
    max_steps: int | None = None,
    timeout_seconds: float = 300.0,
) -> AskResult:
    """Forward `question` to Sourcebot's MCP `ask_codebase` tool. Returns
    the synthesized answer. Raises SourcebotAskError on any failure so the
    CLI's --ask-via=auto can fall back to the local agent.

    `max_steps` is ignored (the underlying tool doesn't take a step cap).
    Kept in the signature for API stability with the old REST-based
    implementation."""
    if not (settings.sourcebot_url and settings.sourcebot_api_key):
        raise SourcebotAskError(
            "SOURCEBOT_URL and SOURCEBOT_API_KEY must be set to use Sourcebot ask"
        )

    try:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client
    except ImportError as e:
        raise SourcebotAskError(f"mcp library not available: {e}")

    base = settings.sourcebot_url.rstrip("/")
    endpoint = f"{base}/api/mcp"
    key = settings.sourcebot_api_key
    bearer = key if key.startswith("sourcebot-") else f"sourcebot-{key}"
    headers = {"Authorization": f"Bearer {bearer}"}

    # Pydantic AI's `decompose_model` is the same Gemini we registered in
    # Sourcebot's config.json. Use it here so the user can swap one place.
    model_name = settings.decompose_model
    # Strip the "gemini/" pydantic-ai prefix if present — Sourcebot wants
    # bare model name.
    if model_name.startswith("gemini/"):
        model_name = model_name[len("gemini/"):]

    t0 = time.monotonic()
    try:
        async with streamablehttp_client(endpoint, headers=headers) as (read, write, *_):
            async with ClientSession(read, write) as session:
                await session.initialize()
                models_result = await session.call_tool("list_language_models", {})
                language_model = _select_language_model(
                    _extract_text(models_result.content),
                    provider="google-generative-ai",
                    model=model_name,
                )
                if language_model is None:
                    raise SourcebotAskError(
                        f"Sourcebot model google-generative-ai/{model_name} is not configured"
                    )

                args: dict = {
                    "query": question.strip(),
                    # Sourcebot's matcher includes displayName in the model key.
                    # Pass back the exact object returned by list_language_models.
                    "languageModel": language_model,
                }
                if repos:
                    args["repos"] = repos

                result = await session.call_tool("ask_codebase", args)
    except Exception as e:
        if isinstance(e, SourcebotAskError):
            raise
        raise SourcebotAskError(f"sourcebot MCP error: {e}")

    raw = _extract_text(result.content)

    if not raw or raw.lower().startswith("failed to ask"):
        # Surface Sourcebot's error verbatim so the next debugger sees it.
        raise SourcebotAskError(raw or "sourcebot ask_codebase returned empty content")

    return AskResult(
        answer=raw,
        citations=_extract_citations_from_text(raw),
        metadata=AskMetadata(model_name=model_name),
        wall_seconds=round(time.monotonic() - t0, 2),
    )


def _extract_text(content: object) -> str:
    text_parts: list[str] = []
    for item in (content or []):
        t = getattr(item, "text", None)
        if t:
            text_parts.append(t)
    return "\n".join(text_parts).strip()


def _select_language_model(
    models_json: str,
    *,
    provider: str,
    model: str,
) -> dict | None:
    """Return Sourcebot's exact model info object for ask_codebase.

    Sourcebot currently keys configured models as
    provider-model-displayName. If we send only provider/model while the
    config includes displayName, ask_codebase reports the model as missing.
    """
    try:
        models = json.loads(models_json)
    except json.JSONDecodeError:
        log.warning("sourcebot list_language_models returned non-JSON: %s", models_json[:200])
        return None

    if not isinstance(models, list):
        return None

    for item in models:
        if not isinstance(item, dict):
            continue
        if item.get("provider") == provider and item.get("model") == model:
            return {
                k: v
                for k, v in item.items()
                if k in {"provider", "model", "displayName"} and v is not None
            }
    return None


def _extract_citations_from_text(text: str) -> list[Citation]:
    """Best-effort: pull `repo/path:line` references out of the answer.
    Sourcebot's ask_codebase response embeds links inline rather than as a
    structured citations list."""
    import re
    citations: list[Citation] = []
    # Pattern: [`...path...:Lstart-Lend`](url) or path:Lnum:colN — keep it simple.
    pattern = re.compile(r"\b([a-z0-9_-]+(?:/[A-Za-z0-9_.\-]+)+\.(?:py|ts|tsx|js|jsx|json|md|yml|yaml))(?::L?(\d+))?")
    seen: set[tuple[str, int | None]] = set()
    for m in pattern.finditer(text):
        path = m.group(1)
        line = int(m.group(2)) if m.group(2) else None
        if (path, line) in seen:
            continue
        seen.add((path, line))
        # Try to split <repo>/<rel> when prefix matches a known shape.
        parts = path.split("/", 1)
        repo = parts[0] if len(parts) == 2 else ""
        rel = parts[1] if len(parts) == 2 else path
        citations.append(Citation(repo=repo, path=rel, start_line=line))
    return citations


def render_ask_markdown(question: str, result: AskResult) -> str:
    lines: list[str] = []
    lines.append(f"# Q: {question.strip().splitlines()[0][:200]}")
    lines.append("")
    if result.wall_seconds is not None:
        bits = [f"wall: {result.wall_seconds}s"]
        if result.metadata and result.metadata.model_name:
            bits.append(f"model: {result.metadata.model_name}")
        lines.append(f"_via Sourcebot ask_codebase — {', '.join(bits)}_")
        lines.append("")
    lines.append("## Answer")
    lines.append(result.answer.strip() or "_(empty answer)_")
    lines.append("")
    if result.citations:
        lines.append("## Citations (parsed from answer)")
        for c in result.citations:
            label = f"{c.repo}/{c.path}" if c.repo else c.path
            suffix = f":L{c.start_line}" if c.start_line else ""
            lines.append(f"- `{label}{suffix}`")
    return "\n".join(lines).rstrip() + "\n"


def write_ask_markdown(question: str, result: AskResult, settings: Settings) -> Path:
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    slug = "".join(ch for ch in question.strip()[:40] if ch.isalnum() or ch in "-_")[:40] or "ask"
    out = settings.output_dir / f"{stamp}-ask-{slug}.md"
    out.write_text(render_ask_markdown(question, result))
    return out
