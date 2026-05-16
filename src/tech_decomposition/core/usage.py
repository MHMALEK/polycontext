"""Token-usage helpers shared by Engine implementations.

Pydantic AI's ``RunResult.usage()`` exposes token counts under names that
have shifted across versions; this helper papers over both. Pair with
``core.llm_registry.estimate_cost_usd`` for cost.
"""
from __future__ import annotations

from typing import Any


def usage_from_result(result: Any) -> tuple[int | None, int | None]:
    """Pull ``(input_tokens, output_tokens)`` out of a pydantic-ai run result.

    Falls back to ``(None, None)`` if usage isn't available — never raises.
    """
    try:
        u = result.usage()
    except Exception:
        return (None, None)
    if u is None:
        return (None, None)
    in_tok = getattr(u, "request_tokens", None) or getattr(u, "input_tokens", None)
    out_tok = getattr(u, "response_tokens", None) or getattr(u, "output_tokens", None)
    return (in_tok, out_tok)
