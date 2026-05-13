"""MarkdownRenderer: renders an EngineResult to user-facing markdown."""
from __future__ import annotations

from ..core.context import RunContext
from ..core.protocols import EngineResult, Rendered


class MarkdownRenderer:
    name = "markdown"

    def render(self, result: EngineResult, ctx: RunContext) -> Rendered:
        lines: list[str] = []
        head_bits: list[str] = []
        if result.transport:
            head_bits.append(f"via: `{result.transport}`")
        if result.model:
            head_bits.append(f"model: `{result.model}`")
        if result.wall_seconds is not None:
            head_bits.append(f"wall: {result.wall_seconds}s")
        if result.cost_usd is not None:
            head_bits.append(f"cost: ${result.cost_usd:.4f}")
        if head_bits:
            lines.append("_" + " · ".join(head_bits) + "_")
            lines.append("")
        lines.append(result.answer_markdown.strip() or "_(empty answer)_")
        if result.citations:
            lines.append("")
            lines.append("## Citations")
            for c in result.citations:
                repo = c.get("repo") or ""
                path = c.get("path") or ""
                lo = c.get("start_line")
                hi = c.get("end_line")
                suf = ""
                if lo and hi and hi != lo:
                    suf = f":L{lo}-L{hi}"
                elif lo:
                    suf = f":L{lo}"
                label = f"{repo}/{path}" if repo else path
                url = c.get("url")
                if url:
                    lines.append(f"- [{label}{suf}]({url})")
                else:
                    lines.append(f"- `{label}{suf}`")
        body = "\n".join(lines).rstrip() + "\n"
        return Rendered(format="markdown", body=body, metadata={"engine": result.engine})
