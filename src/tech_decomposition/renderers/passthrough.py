"""PassthroughMarkdownRenderer: emits ``EngineResult.answer_markdown`` as-is.

Used when the engine has already produced fully rendered markdown
(decompose engines do their own ``render_markdown`` because the prose layout
needs the retrieved-context block + GitLab permalinks).
"""
from __future__ import annotations

from ..core.context import RunContext
from ..core.protocols import EngineResult, Rendered


class PassthroughMarkdownRenderer:
    name = "passthrough_markdown"

    def render(self, result: EngineResult, ctx: RunContext) -> Rendered:
        return Rendered(
            format="markdown",
            body=result.answer_markdown,
            metadata={"engine": result.engine, **result.extra},
        )
