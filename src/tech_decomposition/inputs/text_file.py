"""TextFileSource: load a query from a text file from disk."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..core.context import RunContext
from ..core.protocols import LoadedInput


class TextFileSource:
    """Reads `ref` as a filesystem path; the file contents become the query."""

    name = "text_file"

    async def load(self, ref: str | dict[str, Any], ctx: RunContext) -> LoadedInput:
        if isinstance(ref, dict):
            path = ref.get("path")
        else:
            path = str(ref)
        if not path:
            raise ValueError("TextFileSource: missing path")
        text = Path(path).read_text()
        return LoadedInput(
            kind="text_file",
            title=str(path),
            body=text,
            metadata={
                "source_path": path,
            },
        )
