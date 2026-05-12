from __future__ import annotations

import re
from urllib.parse import urlparse

import httpx

from .config import Settings
from .models import Ticket

_KEY_RE = re.compile(r"([A-Z][A-Z0-9]+-\d+)")


def parse_key_from_url(url: str) -> str | None:
    m = _KEY_RE.search(url)
    return m.group(1) if m else None


def _adf_to_text(node) -> str:
    """Atlassian Document Format → plain text. Lossy but good enough for prompting."""
    if node is None:
        return ""
    if isinstance(node, str):
        return node
    if isinstance(node, list):
        return "".join(_adf_to_text(n) for n in node)
    if isinstance(node, dict):
        t = node.get("type")
        if t == "text":
            return node.get("text", "")
        if t == "hardBreak":
            return "\n"
        parts = _adf_to_text(node.get("content"))
        if t in {"paragraph", "heading", "bulletList", "orderedList", "listItem", "codeBlock"}:
            return parts + "\n"
        return parts
    return ""


async def fetch_ticket(*, settings: Settings, key: str | None, url: str | None) -> Ticket:
    if not key and url:
        key = parse_key_from_url(url)
    if not key:
        raise ValueError("Could not determine Jira ticket key from input.")
    if not (settings.jira_base_url and settings.jira_email and settings.jira_api_token):
        raise RuntimeError(
            "Jira creds not configured. Set JIRA_BASE_URL, JIRA_EMAIL, JIRA_API_TOKEN — "
            "or submit ticket_text instead."
        )

    base = settings.jira_base_url.rstrip("/")
    api = f"{base}/rest/api/3/issue/{key}"
    auth = (settings.jira_email, settings.jira_api_token)

    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.get(api, auth=auth, headers={"Accept": "application/json"})
        r.raise_for_status()
        data = r.json()

    fields = data.get("fields", {})
    title = fields.get("summary", "") or ""
    desc = fields.get("description")
    body = _adf_to_text(desc).strip() if isinstance(desc, dict) else (desc or "")
    labels = list(fields.get("labels") or [])
    components = [c.get("name", "") for c in (fields.get("components") or []) if c.get("name")]

    return Ticket(
        key=key,
        url=url or f"{base}/browse/{key}",
        title=title,
        body=body,
        labels=labels,
        components=components,
    )


def ticket_from_text(text: str, key: str | None = None, url: str | None = None) -> Ticket:
    """For tests / when Jira creds are not configured."""
    head, _, rest = text.partition("\n")
    title = head.strip() or "(untitled)"
    body = rest.strip()
    return Ticket(key=key, url=url, title=title, body=body)


def is_jira_url(value: str) -> bool:
    try:
        u = urlparse(value)
        return bool(u.scheme and u.netloc and "/browse/" in u.path)
    except Exception:
        return False


async def post_comment(*, settings: Settings, key: str, adf_body: dict) -> str:
    """Post a comment on a Jira issue. Returns the created comment ID."""
    if not (settings.jira_base_url and settings.jira_email and settings.jira_api_token):
        raise RuntimeError("Jira creds not configured; cannot post comment.")
    base = settings.jira_base_url.rstrip("/")
    url = f"{base}/rest/api/3/issue/{key}/comment"
    auth = (settings.jira_email, settings.jira_api_token)
    payload = {"body": adf_body}
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(
            url, auth=auth,
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            json=payload,
        )
        r.raise_for_status()
    return str(r.json().get("id", ""))
