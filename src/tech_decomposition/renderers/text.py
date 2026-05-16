"""TextRenderer: strip markdown formatting for plain-text output.

Renders code blocks as-is, replaces inline code with backticks (kept as
plain backticks), strips emphasis markers (``*`` / ``_``) and link syntax
(keeps the link label, drops the URL). Useful for piping into terminals,
plain-text channels, or `--format text` CLI usage.
"""
from __future__ import annotations

import re

from ..core.context import RunContext
from ..core.protocols import EngineResult, Rendered


_LINK_RX = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_BOLD_RX = re.compile(r"\*\*([^*]+)\*\*")
_ITALIC_RX = re.compile(r"(?<![*_])[*_]([^*_]+)[*_](?![*_])")
_HEADER_RX = re.compile(r"^#{1,6}\s+(.+)$", re.MULTILINE)


def _strip_markdown(md: str) -> str:
    # Headers — drop the leading # markers, keep the text.
    md = _HEADER_RX.sub(r"\1", md)
    # Links: [label](url) → label (url) so URLs are still visible.
    md = _LINK_RX.sub(r"\1 (\2)", md)
    # Bold + italic markers.
    md = _BOLD_RX.sub(r"\1", md)
    md = _ITALIC_RX.sub(r"\1", md)
    return md


class TextRenderer:
    name = "text"

    def render(self, result: EngineResult, ctx: RunContext) -> Rendered:
        meta_bits = []
        if result.engine:
            meta_bits.append(f"engine: {result.engine}")
        if result.model:
            meta_bits.append(f"model: {result.model}")
        if result.wall_seconds is not None:
            meta_bits.append(f"wall: {result.wall_seconds}s")
        if result.cost_usd is not None:
            meta_bits.append(f"cost: ${result.cost_usd:.4f}")
        header = " · ".join(meta_bits)
        body = _strip_markdown(result.answer_markdown).strip()
        out = (header + "\n\n" + body) if header else body
        return Rendered(format="text", body=out + "\n", metadata={"engine": result.engine, **result.extra})
