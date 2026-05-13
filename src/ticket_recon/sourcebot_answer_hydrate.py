"""Hydrate MCP `ask_codebase` answers from Sourcebot Postgres `Chat.messages`.

Sourcebot bundles the same extraction bug as MCP: `askCodebase` picks the first
assistant message (`find`), while the Ask UI renders the entire thread—the final
assistant turn usually mirrors what users read. We read persisted messages and
reuse the intended UI-facing extraction (<!--answer--> sentinel + `@file:{}`
linkification, parity with packages/web/src/features/chat/utils.ts)."""

from __future__ import annotations

import json
import logging
import re
from pathlib import PurePosixPath
from urllib.parse import quote, urlencode, urlparse

log = logging.getLogger(__name__)

ANSWER_TAG = "<!--answer-->"
# @see Sourcebot FILE_REFERENCE_REGEX
_FILE_REF_RX = re.compile(
    r"@file:\{([^:}]+)::([^:}]+)(?::(\d+)(?:-(\d+))?)?\}",
    re.MULTILINE,
)
_SESSION_MARKER = "**View full research session:**"


def split_mcp_formatted_answer_and_footer(tool_text: str) -> tuple[str, str]:
    """Split MCP body into `{portable_answer_block}` and trailing session footer."""
    idx = tool_text.find("\n---\n" + _SESSION_MARKER)
    if idx == -1:
        idx = tool_text.find("\r\n---\r\n" + _SESSION_MARKER)
    if idx == -1:
        return tool_text.strip(), ""
    core = tool_text[:idx].strip()
    footer = tool_text[idx:].lstrip("\n").strip()
    return core, footer


def extract_chat_id_from_footer(tool_text: str) -> str | None:
    m = re.search(
        rf"{re.escape(_SESSION_MARKER)}\s*(\S+)",
        tool_text,
        flags=re.MULTILINE,
    )
    if not m:
        return None
    url = m.group(1).strip()
    path = urlparse(url).path or url
    if "/chat/" not in path:
        return None
    tail = path.rstrip("/").rsplit("/", 1)[-1]
    if len(tail) >= 18 and tail[0].isalpha():
        return tail
    return None


def _normalize_repo_relative_path(path: str) -> str:
    return path.replace("\\", "/").lstrip("/")


def _browse_blob_url_path(
    repo_name: str,
    *,
    file_path: str,
    start_line: int | None,
    end_line: int | None,
    revision: str = "HEAD",
) -> str:
    """Equivalent to Next.js `getBrowsePath` for a blob (+ optional highlight)."""
    qs: dict[str, str] = {}
    if start_line is not None:
        lo = start_line
        hi = end_line if end_line is not None else start_line
        qs["highlightRange"] = f"{lo},{hi}"
    normalized = _normalize_repo_relative_path(file_path)
    tail = quote(normalized, safe="")
    out = f"/browse/{repo_name}@{revision}/-/blob/{tail}"
    if qs:
        out += "?" + urlencode(qs)
    return out


def _convert_file_refs_to_portable_links(text: str, browse_origin: str) -> str:
    browse_origin = browse_origin.rstrip("/")

    def repl(m: re.Match[str]) -> str:
        repo, fname = m.group(1), m.group(2)
        sl, el = m.group(3), m.group(4)
        fname = fname.replace("\\", "/")
        display = PurePosixPath(fname).name

        hl_start = int(sl) if sl else None
        hl_end = int(el) if el else None
        label = display
        if hl_start is not None:
            if hl_end is not None and hl_end != hl_start:
                label += f":{hl_start}-{hl_end}"
            else:
                label += f":{hl_start}"
        bp = _browse_blob_url_path(
            repo_name=repo,
            file_path=fname,
            start_line=hl_start,
            end_line=hl_end if hl_end is not None else hl_start,
            revision="HEAD",
        )
        return f"[{label}]({browse_origin}{bp})"

    return _FILE_REF_RX.sub(repl, text)


def _last_text_ui_part(messages: dict) -> str | None:
    parts = messages.get("parts")
    if not isinstance(parts, list):
        return None
    for p in reversed(parts):
        if not isinstance(p, dict):
            continue
        if p.get("type") == "text" and isinstance(p.get("text"), str):
            return p["text"]
    return None


def answer_text_from_saved_messages(messages_json: object, browse_origin: str) -> str | None:
    """Match Sourcebot UI answer extraction applied to persisted JSON messages."""
    if isinstance(messages_json, str):
        try:
            messages_json = json.loads(messages_json)
        except json.JSONDecodeError:
            return None
    if not isinstance(messages_json, list):
        return None
    assistants = [m for m in messages_json if isinstance(m, dict) and m.get("role") == "assistant"]
    if not assistants:
        return None
    last_assistant = assistants[-1]
    raw = _last_text_ui_part(last_assistant)
    if raw is None:
        return None
    if ANSWER_TAG in raw:
        raw = raw.split(ANSWER_TAG, 1)[1]
    stripped = raw.replace(ANSWER_TAG, "").strip()
    return _convert_file_refs_to_portable_links(stripped, browse_origin).strip()


def fetch_answer_via_chat_row(
    database_url: str,
    *,
    chat_id: str,
    browse_origin: str,
) -> str | None:
    try:
        import psycopg
    except ImportError:
        log.warning("psycopg not installed; cannot hydrate Ask answer from Postgres")
        return None
    try:
        with psycopg.connect(database_url, connect_timeout=8) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    'SELECT "messages"::text FROM "Chat" WHERE id = %s LIMIT 1',
                    (chat_id,),
                )
                row = cur.fetchone()
                if row is None or row[0] is None:
                    return None
                return answer_text_from_saved_messages(row[0], browse_origin=browse_origin)
    except Exception as e:
        log.warning("SOURCEBOT_DATABASE_URL hydration failed for chat %s: %s", chat_id, e)
        return None


def hydrate_sourcebot_tool_response(
    mcp_tool_text: str,
    *,
    database_url: str,
    browse_origin: str,
) -> str:
    """Return tool text body, rewritten with DB-backed answer when possible."""
    if not database_url.strip():
        return mcp_tool_text
    core, footer = split_mcp_formatted_answer_and_footer(mcp_tool_text)
    chat_id = extract_chat_id_from_footer(mcp_tool_text) or extract_chat_id_from_footer(footer + "\n" + footer)
    if footer and not chat_id:
        chat_id = extract_chat_id_from_footer("\n".join(("", "---", footer)))
    if not chat_id:
        return mcp_tool_text
    better = fetch_answer_via_chat_row(
        database_url, chat_id=chat_id, browse_origin=browse_origin
    )
    if not better:
        return mcp_tool_text
    if footer:
        return f"{better}\n\n---\n{footer}"
    return better
