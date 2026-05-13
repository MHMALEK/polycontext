"""JiraTicketSource: load a Jira ticket by key or URL via the existing client."""
from __future__ import annotations

from typing import Any

from ..core.context import RunContext
from ..core.protocols import LoadedInput
from ..clients.jira import fetch_ticket


class JiraTicketSource:
    """Accepts a ref string (treated as key or URL) or a dict with `key`/`url`."""

    name = "jira"

    async def load(self, ref: str | dict[str, Any], ctx: RunContext) -> LoadedInput:
        key: str | None = None
        url: str | None = None
        if isinstance(ref, dict):
            key = ref.get("key")
            url = ref.get("url")
        else:
            s = str(ref).strip()
            if s.startswith("http://") or s.startswith("https://"):
                url = s
            else:
                key = s
        ticket = await fetch_ticket(settings=ctx.settings, key=key, url=url)
        return LoadedInput(
            kind="jira_ticket",
            title=ticket.title,
            body=ticket.body,
            metadata={
                "key": ticket.key,
                "url": ticket.url,
                "labels": list(ticket.labels),
                "components": list(ticket.components),
                "ticket": ticket.model_dump(mode="json"),
            },
        )
