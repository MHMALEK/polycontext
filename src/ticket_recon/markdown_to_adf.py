"""Tiny markdown→ADF converter, tailored to the markdown this app emits.

ADF spec: https://developer.atlassian.com/cloud/jira/platform/apis/document/structure/

We support exactly the markdown shapes our renderer produces:
- ATX headings (# ## ###)
- Paragraphs
- Unordered lists (- item) including task lists (- [ ] item)
- Inline links [text](url)
- Inline code `code`
- Horizontal rules (---)

Anything we don't recognize is emitted as a plain paragraph. Good enough for
Jira comments — they look readable, links work, headings format properly.
"""
from __future__ import annotations

import re

_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")
_CODE_RE = re.compile(r"`([^`]+)`")


def _inline_marks(text: str) -> list[dict]:
    """Split a line of inline markdown into ADF inline nodes (text/link/code)."""
    nodes: list[dict] = []
    i = 0
    n = len(text)
    while i < n:
        m_link = _LINK_RE.search(text, i)
        m_code = _CODE_RE.search(text, i)
        # Find whichever comes first.
        candidates = [m for m in (m_link, m_code) if m]
        if not candidates:
            nodes.append({"type": "text", "text": text[i:]})
            break
        m_next = min(candidates, key=lambda m: m.start())
        if m_next.start() > i:
            nodes.append({"type": "text", "text": text[i:m_next.start()]})
        if m_next is m_link:
            label, url = m_next.group(1), m_next.group(2)
            nodes.append({
                "type": "text",
                "text": label,
                "marks": [{"type": "link", "attrs": {"href": url}}],
            })
        else:
            code = m_next.group(1)
            nodes.append({
                "type": "text",
                "text": code,
                "marks": [{"type": "code"}],
            })
        i = m_next.end()
    return [n for n in nodes if n.get("text")]


def _paragraph(text: str) -> dict:
    return {"type": "paragraph", "content": _inline_marks(text)}


def _heading(level: int, text: str) -> dict:
    return {
        "type": "heading",
        "attrs": {"level": max(1, min(6, level))},
        "content": _inline_marks(text),
    }


def _list_item(text: str) -> dict:
    # Task-list style "[ ] x" rendered as a plain bullet (Jira ADF taskItem
    # requires a taskList parent + localId; not worth the complexity here).
    text = re.sub(r"^\[[ x]\]\s*", "", text)
    return {
        "type": "listItem",
        "content": [{"type": "paragraph", "content": _inline_marks(text)}],
    }


def markdown_to_adf(md: str) -> dict:
    lines = md.splitlines()
    content: list[dict] = []

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()

        if not line.strip():
            i += 1
            continue

        if line.strip() == "---":
            content.append({"type": "rule"})
            i += 1
            continue

        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            content.append(_heading(len(m.group(1)), m.group(2)))
            i += 1
            continue

        if line.lstrip().startswith("- "):
            items: list[dict] = []
            while i < len(lines) and lines[i].lstrip().startswith("- "):
                items.append(_list_item(lines[i].lstrip()[2:]))
                i += 1
            content.append({"type": "bulletList", "content": items})
            continue

        # Paragraph: collect adjacent non-empty, non-list, non-heading lines.
        buf = [line]
        i += 1
        while i < len(lines):
            nxt = lines[i].rstrip()
            if not nxt.strip():
                break
            if nxt.lstrip().startswith("- ") or re.match(r"^#{1,6}\s", nxt) or nxt.strip() == "---":
                break
            buf.append(nxt)
            i += 1
        content.append(_paragraph(" ".join(buf)))

    return {"type": "doc", "version": 1, "content": content}
