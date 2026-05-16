"""FileSink: writes the rendered body to ``outputs/{subdir}/`` with an
extension matching the format. Replaces the old MarkdownFileSink.

The format is set at construction time (so the sink knows what extension to
use) and matches the ``prefers`` attribute, which lets the Pipeline pick this
sink's preferred Rendered out of multiple candidates.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from ..core.context import RunContext
from ..core.protocols import Rendered, SinkResult


_EXT_BY_FORMAT: dict[str, str] = {
    "markdown": "md",
    "html": "html",
    "text": "txt",
}


def _slugify(text: str, n: int = 40) -> str:
    cleaned = "".join(ch for ch in text.strip()[:n] if ch.isalnum() or ch in "-_")
    return cleaned[:n] or "out"


class FileSink:
    name = "file"

    def __init__(
        self,
        *,
        subdir: str = "answers",
        format: Literal["markdown", "html", "text"] = "markdown",
        slug_source: str = "",
    ):
        self.subdir = subdir
        self.format = format
        self.prefers = format
        self.slug_source = slug_source

    async def deliver(self, rendered: Rendered, ctx: RunContext) -> SinkResult:
        out_dir = ctx.settings.output_dir / self.subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        slug = _slugify(self.slug_source or rendered.metadata.get("engine") or "out")
        ext = _EXT_BY_FORMAT.get(rendered.format, "txt")
        out_path = out_dir / f"{stamp}-{slug}.{ext}"
        out_path.write_text(rendered.body)
        return SinkResult(sink=self.name, location=str(out_path), extra={"format": rendered.format})


# Backwards-compat alias — was the old name; some imports may still use it.
MarkdownFileSink = FileSink
