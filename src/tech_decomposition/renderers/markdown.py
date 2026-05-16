"""MarkdownRenderer: emits the engine's answer markdown unchanged.

The adapter (Sourcebot, Claude Code SDK, etc.) already produces a complete
markdown answer with inline citation links. We don't add a metadata header
(the CLI prints that as a status line separately) or a separate citations
section (citations are already inline in the answer body).
"""
from __future__ import annotations

from ..core.context import RunContext
from ..core.protocols import EngineResult, Rendered


class MarkdownRenderer:
    name = "markdown"

    def render(self, result: EngineResult, ctx: RunContext) -> Rendered:
        body = result.answer_markdown.strip() or "_(empty answer)_"
        return Rendered(format="markdown", body=body + "\n", metadata={"engine": result.engine})
