"""Code Q&A via Sourcebot — Sentinel-compatible ``POST /api/ask`` (SSE).

Aligns with `tract-projects/sentinel` `SourcebotCodeIntelligenceAdapter`:

- Endpoint: ``POST {SOURCEBOT_URL}/api/ask``
- Headers: ``X-Sourcebot-Api-Key: sourcebot-{key}``, ``Accept: text/event-stream``
- JSON body: camelCase (`question`, optional `repos`, `maxSteps`, `languageModel`, …)

If that endpoint returns **404** (some OSS builds omit it), we normally fall back to MCP
``ask_codebase`` over ``/api/mcp``. Set ``SOURCEBOT_DISABLE_MCP_FALLBACK=1`` to use only
the chat-style SSE endpoint (no MCP).

Optional model override env vars mirror Sentinel (`SOURCEBOT_LANGUAGE_MODEL_*`)."""
from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import httpx

from pydantic import BaseModel, Field

from .config import Settings
from .sourcebot_answer_hydrate import (
    hydrate_sourcebot_tool_response,
    split_mcp_formatted_answer_and_footer,
)

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
    transport: Literal["sse", "mcp", "local"] = "sse"


class AskResult(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    metadata: AskMetadata | None = None
    wall_seconds: float | None = None


class SourcebotAskError(Exception):
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class _PolishedAnswer(BaseModel):
    answer: str = Field(description="Clean final answer only. Keep citations and links intact.")


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


def _normalize_model_name(name: str) -> str:
    if name.startswith("gemini/"):
        return name[len("gemini/"):]
    return name


def _language_model_payload(settings: Settings) -> dict[str, str] | None:
    """Build ``languageModel`` for /api/ask (camelCase nested object)."""
    prov = settings.sourcebot_language_model_provider.strip()
    mname = settings.sourcebot_language_model_name.strip()
    disp = settings.sourcebot_language_model_display_name.strip()
    if prov and mname:
        out: dict[str, str] = {"provider": prov, "model": mname}
        if disp:
            out["displayName"] = disp
        return out
    # Default: Gemini model from decomposition settings + provider used in Sourcebot config.
    dm = _normalize_model_name(settings.decompose_model.strip())
    if not dm:
        return None
    return {"provider": "google-generative-ai", "model": dm}


def _build_sse_request_payload(
    question: str,
    settings: Settings,
    repos: list[str] | None,
    max_steps: int | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"question": question.strip()}
    if repos:
        payload["repos"] = repos
    if max_steps is not None:
        if not (1 <= max_steps <= 50):
            raise SourcebotAskError("maxSteps must be between 1 and 50", status_code=400)
        payload["maxSteps"] = max_steps
    lm = _language_model_payload(settings)
    if lm:
        payload["languageModel"] = lm
    return payload


def _sse_parse_citation(d: dict) -> Citation | None:
    try:
        return Citation(
            repo=d.get("repo", "") or "",
            path=d.get("path", "") or "",
            start_line=d.get("startLine"),
            end_line=d.get("endLine"),
            revision=d.get("revision"),
            url=d.get("url"),
        )
    except Exception:
        return None


async def _ask_via_sse_api_ask(
    question: str,
    *,
    settings: Settings,
    repos: list[str] | None,
    max_steps: int | None,
    timeout_seconds: float,
) -> AskResult:
    base = settings.sourcebot_url.rstrip("/")
    url = f"{base}/api/ask"
    key = settings.sourcebot_api_key
    headers = {
        "X-Sourcebot-Api-Key": _x_sourcebot_api_key_value(key),
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    payload = _build_sse_request_payload(question, settings, repos, max_steps)
    model_for_meta = (_language_model_payload(settings) or {}).get("model")

    t0 = time.monotonic()
    answer_parts: list[str] = []
    citations: list[Citation] = []
    seen_cit: set[tuple[str, str, int | None, int | None]] = set()
    metadata_raw: dict = {}

    timeout = httpx.Timeout(timeout_seconds, connect=15.0)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        async with client.stream("POST", url, json=payload, headers=headers) as resp:
            if resp.status_code != 200:
                detail = ""
                try:
                    detail = (await resp.aread()).decode("utf-8", errors="replace")[:500]
                except Exception:
                    pass
                msg = detail or f"HTTP {resp.status_code}"
                raise SourcebotAskError(msg, status_code=resp.status_code)

            buffer = ""
            async for chunk in resp.aiter_bytes():
                if not chunk:
                    continue
                buffer += chunk.decode("utf-8", errors="replace")
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    if line in ("[DONE]", "data: [DONE]"):
                        break
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            event_data = json.loads(data_str)
                        except json.JSONDecodeError:
                            if data_str.strip() != "[DONE]":
                                log.debug("skipping bad SSE JSON: %s", data_str[:80])
                            continue
                        et = event_data.get("type")
                        if et == "text-delta":
                            td = event_data.get("delta") or event_data.get("textDelta", "")
                            if td:
                                answer_parts.append(td)
                        elif et == "citation":
                            c = _sse_parse_citation(event_data)
                            if c:
                                ck = (c.repo, c.path, c.start_line, c.end_line)
                                if ck not in seen_cit:
                                    seen_cit.add(ck)
                                    citations.append(c)
                        elif et == "message-metadata":
                            mm = event_data.get("messageMetadata") or {}
                            if isinstance(mm, dict):
                                metadata_raw = mm
                                for cd in mm.get("citations", []) or []:
                                    if isinstance(cd, dict):
                                        c = _sse_parse_citation(cd)
                                        if c:
                                            ck = (c.repo, c.path, c.start_line, c.end_line)
                                            if ck not in seen_cit:
                                                seen_cit.add(ck)
                                                citations.append(c)
                        elif et in ("finish", "start"):
                            pass

    answer = "".join(answer_parts).strip()
    if not answer:
        raise SourcebotAskError("empty answer from Sourcebot SSE /api/ask")

    meta = _metadata_from_message_metadata(metadata_raw, model_for_meta=model_for_meta, transport="sse")
    wall = round(time.monotonic() - t0, 2)
    return AskResult(answer=answer, citations=citations, metadata=meta, wall_seconds=wall)


def _metadata_from_message_metadata(
    metadata_raw: dict,
    *,
    model_for_meta: str | None,
    transport: Literal["sse", "mcp", "local"],
) -> AskMetadata:
    sources_seen_strs: list[str] = []
    for src in metadata_raw.get("sourcesSeen", []) or []:
        if isinstance(src, str):
            sources_seen_strs.append(src)
        elif isinstance(src, dict):
            repo = src.get("repo", "")
            path = src.get("path", "")
            if repo and path:
                sources_seen_strs.append(f"{repo}/{path}")
            elif repo:
                sources_seen_strs.append(str(repo))

    mn = metadata_raw.get("modelName") or model_for_meta
    return AskMetadata(
        total_tokens=metadata_raw.get("totalTokens"),
        total_input_tokens=metadata_raw.get("totalInputTokens"),
        total_output_tokens=metadata_raw.get("totalOutputTokens"),
        total_response_time_ms=metadata_raw.get("totalResponseTimeMs"),
        model_name=mn,
        trace_id=metadata_raw.get("traceId"),
        sources_seen=sources_seen_strs,
        transport=transport,
    )


def _maybe_fallback_to_mcp(err: SourcebotAskError) -> bool:
    """Fall back only when Route /api/ask is absent (OSS) or unreachable as 404."""
    return err.status_code == 404


def _extract_text(content: object) -> str:
    text_parts: list[str] = []
    for item in (content or []):
        t = getattr(item, "text", None)
        if t:
            text_parts.append(t)
    return "\n".join(text_parts).strip()


def _build_mcp_query(question: str) -> str:
    q = question.strip()
    if not q:
        return q
    return q + _MCP_ANSWER_STYLE_SUFFIX


def _default_sourcebot_repos(settings: Settings) -> list[str] | None:
    repos: list[str] = []
    # Prefer fully qualified repo names from config, e.g. gitlab.com/org/project.
    for path in settings.gitlab_projects.values():
        p = str(path).strip().lstrip("/")
        if p:
            repos.append(f"gitlab.com/{p}")
    deduped = [r for i, r in enumerate(repos) if r and r not in repos[:i]]
    return deduped or None


def _strip_process_narration(text: str) -> str:
    """Best-effort cleanup when MCP returns process/meta narration."""
    text = re.sub(
        r"(?i)\bI can now answer(?: the user's question)?\.?",
        "",
        text,
    )
    text = re.sub(
        r"(?i)\s*I have sufficient information[^.]*\.?",
        "",
        text,
    )
    text = re.sub(
        r"(?i)\bI(?:'ve| have) (?:analyzed|located|confirmed|reviewed|checked)\b[^.]*\.\s*",
        "",
        text,
    )
    lines: list[str] = []
    for line in text.splitlines():
        s = line.strip()
        if re.match(r"(?i)^i (have|will|can now) ", s):
            continue
        if re.match(r"(?i)^based on my analysis", s):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _prefer_final_answer_block(text: str) -> str:
    core, footer = split_mcp_formatted_answer_and_footer(text)
    m = list(
        re.finditer(
            r"(?im)^(?:final answer:|here's the final answer:)",
            core,
        )
    )
    if m:
        core = core[m[-1].start():].strip()
    if footer:
        if footer.lstrip().startswith("---"):
            return f"{core}\n\n{footer}"
        return f"{core}\n\n---\n{footer}"
    return core


async def _maybe_polish_with_pydantic_ai(
    text: str,
    *,
    question: str,
    settings: Settings,
) -> str:
    """Optional LLM cleanup pass for MCP output noise."""
    if not settings.ask_cleanup_with_pydantic_ai:
        return text
    if not settings.gemini_api_key.strip():
        return text

    try:
        from pydantic_ai import Agent
        from pydantic_ai.models.gemini import GeminiModel
        from pydantic_ai.settings import ModelSettings
    except Exception as e:  # pragma: no cover - optional dependency runtime
        log.warning("pydantic-ai cleanup unavailable: %s", e)
        return text

    os.environ["GEMINI_API_KEY"] = settings.gemini_api_key
    model_name = settings.ask_cleanup_model.strip() or settings.enrich_model
    agent = Agent(
        model=GeminiModel(model_name),
        output_type=_PolishedAnswer,
        model_settings=ModelSettings(temperature=0.0),
        system_prompt=(
            "Rewrite raw Q&A output into a clean final answer.\n"
            "Rules:\n"
            "- Remove process narration and duplicate paragraphs.\n"
            "- Keep factual content, markdown links, and file citations.\n"
            "- Do not invent new claims.\n"
            "- Return only the final answer text."
        ),
    )

    prompt = (
        "Question:\n"
        f"{question.strip()}\n\n"
        "Raw answer to clean:\n"
        f"{text.strip()}"
    )
    try:
        out = await agent.run(prompt)
    except Exception as e:  # pragma: no cover - defensive fallback
        log.warning("pydantic-ai cleanup failed: %s", e)
        return text

    cleaned = (out.output.answer or "").strip()
    return cleaned or text


def _select_language_model(models_json: str, *, provider: str, model: str) -> dict | None:
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


async def _ask_via_mcp_ask_codebase(
    question: str,
    *,
    settings: Settings,
    repos: list[str] | None,
    timeout_seconds: float,
) -> AskResult:
    try:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client
    except ImportError as e:
        raise SourcebotAskError(f"mcp library not available: {e}")

    base = settings.sourcebot_url.rstrip("/")
    endpoint = f"{base}/api/mcp"
    bearer = _x_sourcebot_api_key_value(settings.sourcebot_api_key)
    headers = {"Authorization": f"Bearer {bearer}"}

    model_name = _normalize_model_name(settings.decompose_model)
    effective_repos = repos or _default_sourcebot_repos(settings)

    t0 = time.monotonic()
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

            args: dict = {"query": _build_mcp_query(question), "languageModel": language_model}
            if effective_repos:
                args["repos"] = effective_repos

            result = await session.call_tool("ask_codebase", args)

    raw = _extract_text(result.content)
    if not raw or raw.lower().startswith("failed to ask"):
        raise SourcebotAskError(raw or "sourcebot ask_codebase returned empty content")

    if settings.sourcebot_database_url.strip():
        browse_origin = (settings.sourcebot_browse_origin or settings.sourcebot_url).strip()
        if browse_origin:
            raw = hydrate_sourcebot_tool_response(
                raw,
                database_url=settings.sourcebot_database_url,
                browse_origin=browse_origin,
            )
    raw = _prefer_final_answer_block(raw)
    raw = _strip_process_narration(raw)
    raw = await _maybe_polish_with_pydantic_ai(raw, question=question, settings=settings)

    wall = round(time.monotonic() - t0, 2)
    return AskResult(
        answer=raw,
        citations=_extract_citations_from_text(raw),
        metadata=AskMetadata(model_name=model_name, transport="mcp"),
        wall_seconds=wall,
    )


def _extract_citations_from_text(text: str) -> list[Citation]:
    """Best-effort path heuristics for MCP plaintext answers."""
    import re

    citations: list[Citation] = []
    pattern = re.compile(
        r"\b([a-z0-9_-]+(?:/[A-Za-z0-9_.\-]+)+\.(?:py|ts|tsx|js|jsx|json|md|yml|yaml))(?::L?(\d+))?"
    )
    seen: set[tuple[str, int | None]] = set()
    for m in pattern.finditer(text):
        path = m.group(1)
        line = int(m.group(2)) if m.group(2) else None
        if (path, line) in seen:
            continue
        seen.add((path, line))
        parts = path.split("/", 1)
        repo = parts[0] if len(parts) == 2 else ""
        rel = parts[1] if len(parts) == 2 else path
        citations.append(Citation(repo=repo, path=rel, start_line=line))
    return citations


async def ask_sourcebot(
    question: str,
    *,
    settings: Settings,
    repos: list[str] | None = None,
    max_steps: int | None = None,
    timeout_seconds: float = 300.0,
    disable_mcp_fallback: bool | None = None,
) -> AskResult:
    """Ask Sourcebot: prefer Sentinel-style ``/api/ask`` SSE; fall back to MCP on 404 unless disabled."""
    if not (settings.sourcebot_url and settings.sourcebot_api_key):
        raise SourcebotAskError(
            "SOURCEBOT_URL and SOURCEBOT_API_KEY must be set to use Sourcebot ask"
        )

    no_mcp = (
        settings.sourcebot_disable_mcp_fallback
        if disable_mcp_fallback is None
        else disable_mcp_fallback
    )

    try:
        return await _ask_via_sse_api_ask(
            question, settings=settings, repos=repos,
            max_steps=max_steps, timeout_seconds=timeout_seconds,
        )
    except httpx.ConnectError as e:
        raise SourcebotAskError(f"cannot reach Sourcebot: {e}") from e
    except httpx.TimeoutException as e:
        raise SourcebotAskError(f"Sourcebot /api/ask timed out: {e}") from e
    except SourcebotAskError as e:
        if no_mcp or not _maybe_fallback_to_mcp(e):
            raise
        log.info(
            "Sourcebot /api/ask unavailable (%s); falling back to MCP ask_codebase",
            e.status_code or e.args[0] if e.args else "?",
        )

    try:
        return await _ask_via_mcp_ask_codebase(
            question, settings=settings, repos=repos, timeout_seconds=timeout_seconds,
        )
    except Exception as fallback_err:
        raise SourcebotAskError(f"MCP fallback failed: {fallback_err}") from fallback_err


def render_ask_markdown(question: str, result: AskResult) -> str:
    lines: list[str] = []
    lines.append(f"# Q: {question.strip().splitlines()[0][:200]}")
    lines.append("")
    if result.wall_seconds is not None:
        bits = [f"wall: {result.wall_seconds}s"]
        if result.metadata and result.metadata.model_name:
            bits.append(f"model: {result.metadata.model_name}")
        tr = result.metadata.transport if result.metadata else "sse"
        via = {
            "sse": "Sourcebot SSE /api/ask",
            "mcp": "Sourcebot MCP ask_codebase",
            "local": "local agent",
        }[tr]
        bits.insert(0, f"via: {via}")
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
