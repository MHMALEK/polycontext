"""CLISink: prints the rendered body to the terminal via rich."""
from __future__ import annotations

from rich.console import Console

from ..core.context import RunContext
from ..core.protocols import Rendered, SinkResult


class CLISink:
    name = "cli"
    prefers = "markdown"

    def __init__(self, *, console: Console | None = None):
        self.console = console or Console()

    async def deliver(self, rendered: Rendered, ctx: RunContext) -> SinkResult:
        self.console.print(rendered.body)
        return SinkResult(sink=self.name)
