"""TextFileSource: load a ticket-shaped text file from disk."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..core.context import RunContext
from ..core.protocols import LoadedInput
from ..clients.jira import ticket_from_text


class TextFileSource:
    """Reads `ref` as a filesystem path; the file contents become title+body."""

    name = "text_file"

    async def load(self, ref: str | dict[str, Any], ctx: RunContext) -> LoadedInput:
        if isinstance(ref, dict):
            path = ref.get("path")
            key = ref.get("key")
            url = ref.get("url")
        else:
            path, key, url = str(ref), None, None
        if not path:
            raise ValueError("TextFileSource: missing path")
        text = Path(path).read_text()
        ticket = ticket_from_text(text, key=key, url=url)
        return LoadedInput(
            kind="text_file",
            title=ticket.title,
            body=ticket.body,
            metadata={
                "key": ticket.key,
                "url": ticket.url,
                "ticket": ticket.model_dump(mode="json"),
                "source_path": path,
            },
        )
