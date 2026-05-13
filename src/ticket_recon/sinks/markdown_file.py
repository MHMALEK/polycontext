"""MarkdownFileSink: writes the rendered body to ``outputs/`` with a timestamped slug."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from ..core.context import RunContext
from ..core.protocols import Rendered, SinkResult


def _slugify(text: str, n: int = 40) -> str:
    cleaned = "".join(ch for ch in text.strip()[:n] if ch.isalnum() or ch in "-_")
    return cleaned[:n] or "out"


class MarkdownFileSink:
    name = "markdown_file"
    prefers = "markdown"

    def __init__(self, *, subdir: str = "answers", slug_source: str = ""):
        self.subdir = subdir
        self.slug_source = slug_source

    async def deliver(self, rendered: Rendered, ctx: RunContext) -> SinkResult:
        out_dir: Path = ctx.settings.output_dir / self.subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        slug = _slugify(self.slug_source or rendered.metadata.get("engine") or "out")
        out_path = out_dir / f"{stamp}-{slug}.md"
        out_path.write_text(rendered.body)
        return SinkResult(sink=self.name, location=str(out_path))
