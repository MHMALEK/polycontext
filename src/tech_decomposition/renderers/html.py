"""HtmlRenderer: render the engine's markdown answer to HTML.

Uses the standard ``markdown`` library with the ``fenced_code``, ``tables``,
and ``codehilite``-friendly extensions. Wraps the result in a minimal HTML
shell so the output is renderable on its own.
"""
from __future__ import annotations

import html

import markdown as _md

from ..core.context import RunContext
from ..core.protocols import EngineResult, Rendered

_EXTENSIONS = ["fenced_code", "tables", "sane_lists", "nl2br"]

_SHELL = """\
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            max-width: 880px; margin: 2rem auto; padding: 0 1rem; line-height: 1.55;
            color: #1f2328; }}
    pre, code {{ font-family: SFMono-Regular, Menlo, Consolas, monospace; }}
    pre {{ background: #f6f8fa; padding: 12px 16px; border-radius: 6px; overflow-x: auto; }}
    code {{ background: rgba(175,184,193,.2); padding: 0.1em 0.3em; border-radius: 3px; }}
    pre code {{ background: none; padding: 0; }}
    a {{ color: #0969da; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    table {{ border-collapse: collapse; margin: 1em 0; }}
    th, td {{ border: 1px solid #d0d7de; padding: 6px 12px; }}
    blockquote {{ border-left: 4px solid #d0d7de; margin: 1em 0; padding: 0 1em; color: #57606a; }}
    .meta {{ color: #57606a; font-size: 0.9em; margin-bottom: 1em; }}
  </style>
</head>
<body>
  <div class="meta">{meta}</div>
  {body}
</body>
</html>
"""


class HtmlRenderer:
    name = "html"

    def render(self, result: EngineResult, ctx: RunContext) -> Rendered:
        body_html = _md.markdown(result.answer_markdown, extensions=_EXTENSIONS)
        meta_bits = []
        if result.engine:
            meta_bits.append(f"engine: <code>{html.escape(result.engine)}</code>")
        if result.model:
            meta_bits.append(f"model: <code>{html.escape(result.model)}</code>")
        if result.wall_seconds is not None:
            meta_bits.append(f"wall: {result.wall_seconds}s")
        if result.cost_usd is not None:
            meta_bits.append(f"cost: ${result.cost_usd:.4f}")
        title = (result.answer_markdown.splitlines()[0] if result.answer_markdown else "answer")[:80]
        page = _SHELL.format(
            title=html.escape(title),
            meta=" · ".join(meta_bits) or "&nbsp;",
            body=body_html,
        )
        return Rendered(format="html", body=page, metadata={"engine": result.engine, **result.extra})
