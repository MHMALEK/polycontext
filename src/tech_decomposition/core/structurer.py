"""PydanticAIStructurer: turn raw engine markdown into a typed ``StructuredAnswer``.

This replaces the regex-based "narration stripper + citation extractor"
chain. Instead of guessing at patterns, we let a cheap Flash-tier model
do one pass: strip pre-answer chatter, keep the substantive answer, lift
every file:line reference into a structured ``Citation``.

Cost: one Flash call (~$0.0003) per ask. Worth it for predictable output
shape and zero-regex maintenance.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings

from ..config import Settings
from .llm_registry import estimate_cost_usd, get_model


class StructuredCitation(BaseModel):
    repo: str = ""
    path: str
    start_line: int | None = None
    end_line: int | None = None
    url: str | None = None


class StructuredAnswer(BaseModel):
    """Cleaned, structured form of an engine's raw markdown answer."""

    answer: str = Field(
        description=(
            "The final answer to the user's question, in Markdown. Strip any "
            "pre-answer narration like 'Okay, I'm ready to answer' or 'I can "
            "now answer the user'. Keep code blocks, file mentions, and links."
        ),
    )
    citations: list[StructuredCitation] = Field(
        default_factory=list,
        description=(
            "Every distinct file referenced in the answer, with line numbers "
            "if mentioned. If the raw answer hyperlinks files (e.g. "
            "`[file.py:23](http://.../browse/...)`), include the URL."
        ),
    )
    confidence: Literal["low", "medium", "high"] = "medium"
    open_questions: list[str] = Field(
        default_factory=list,
        description="Anything the answer flagged as unresolved.",
    )


_SYSTEM = """\
You normalize raw output from a code-search assistant into a clean structured
answer.

Rules:
- The raw text may contain "process narration" like "I can now answer the user",
  "Okay, I have what I need", "Time to answer", "I'm ready to write the answer".
  STRIP these. They're never part of the actual answer.
- If the agent wrote the synthesis multiple times (drafts + restatements), keep
  the most complete version. Do not concatenate redundant rounds.
- Preserve Markdown structure: headings, lists, code blocks, links.
- For each file referenced in the answer, add a Citation. If the raw text
  hyperlinks the file (e.g. `[utils.py:23](http://localhost:3000/browse/.../utils.py?highlightRange=23,23)`),
  include the URL in the citation.
- Do NOT invent claims. If a fact isn't in the raw answer, don't add it.
- If the answer is partial / unverifiable, set confidence="low" and add
  open_questions.
"""


class PydanticAIStructurer:
    """Single Flash-call cleanup of raw markdown → StructuredAnswer."""

    name = "pydantic_ai_structurer"

    def __init__(self, settings: Settings, *, model_spec: str | None = None):
        self.settings = settings
        self.model_spec = model_spec or f"gemini:{settings.enrich_model}"
        self._agent: Agent[None, StructuredAnswer] | None = None

    def _build(self) -> Agent[None, StructuredAnswer]:
        if self._agent is not None:
            return self._agent
        model = get_model(self.model_spec, self.settings)
        self._agent = Agent(
            model=model,
            output_type=StructuredAnswer,
            system_prompt=_SYSTEM,
            model_settings=ModelSettings(temperature=0.0),
        )
        return self._agent

    async def structure(
        self,
        raw_markdown: str,
        question: str,
    ) -> tuple[StructuredAnswer, int | None, int | None, float | None]:
        """Run the structurer; return (answer, in_tok, out_tok, cost_usd)."""
        agent = self._build()
        prompt = (
            f"Question:\n{question.strip()}\n\n"
            f"Raw assistant output to clean and structure:\n{raw_markdown.strip()}"
        )
        result = await agent.run(prompt)
        u = result.usage()
        in_tok = getattr(u, "request_tokens", None) or getattr(u, "input_tokens", None)
        out_tok = getattr(u, "response_tokens", None) or getattr(u, "output_tokens", None)
        cost = estimate_cost_usd(self.model_spec, in_tok or 0, out_tok or 0)
        return result.output, in_tok, out_tok, cost
