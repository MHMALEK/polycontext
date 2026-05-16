"""Raw-question input source: ref is the question string itself."""
from __future__ import annotations

from typing import Any

from ..core.context import RunContext
from ..core.protocols import LoadedInput


class RawQuestionSource:
    name = "raw"

    async def load(self, ref: str | dict[str, Any], ctx: RunContext) -> LoadedInput:
        if isinstance(ref, dict):
            question = str(
                ref.get("question") or ref.get("body") or ref.get("query") or ""
            ).strip()
        else:
            question = str(ref).strip()
        if not question:
            raise ValueError("RawQuestionSource: empty question")
        return LoadedInput(kind="question", body=question)
