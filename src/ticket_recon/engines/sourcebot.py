"""SourcebotEngine: wraps the existing ``ask_sourcebot`` call.

Thin adapter — pipes EnrichedQuestion in, EngineResult out. The actual
HTTP logic lives in ``ticket_recon.ask`` so the engine here stays a
shim that converts shapes.
"""
from __future__ import annotations

from ..clients.sourcebot import ask_sourcebot
from ..core.context import RunContext
from ..core.protocols import EngineResult, EnrichedQuestion


class SourcebotEngine:
    name = "sourcebot"

    def __init__(self, *, max_steps: int | None = None, disable_mcp_fallback: bool | None = None):
        self.max_steps = max_steps
        self.disable_mcp_fallback = disable_mcp_fallback

    async def run(self, q: EnrichedQuestion, ctx: RunContext) -> EngineResult:
        repos = q.suspected_repos or None
        result = await ask_sourcebot(
            q.question,
            settings=ctx.settings,
            repos=repos,
            max_steps=self.max_steps,
            disable_mcp_fallback=self.disable_mcp_fallback,
        )
        meta = result.metadata
        return EngineResult(
            engine=self.name,
            answer_markdown=result.answer,
            citations=[c.model_dump(mode="json") for c in result.citations],
            model=(meta.model_name if meta else None),
            transport=(meta.transport if meta else None),
            wall_seconds=result.wall_seconds,
            input_tokens=(meta.total_input_tokens if meta else None),
            output_tokens=(meta.total_output_tokens if meta else None),
            extra={
                "chat_id": meta.chat_id if meta else None,
                "chat_url": meta.chat_url if meta else None,
            },
        )
