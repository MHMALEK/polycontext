"""Final-pass contradiction check: read the source ticket + the proposed
decomposition, return any subtask that contradicts a criterion in the ticket.

Catches the SCRUM-18-class failure where the decomposition proposed deleting
the DAG while the ticket required a rollback path. Costs ~$0.002/ticket
because it's a Flash call on small inputs."""
from __future__ import annotations

import os
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.settings import ModelSettings

from ..config import Settings
from ..models import Decomposition


class Contradiction(BaseModel):
    subtask_index: int = Field(description="1-based index of the offending subtask, or 0 if it's an overall conflict.")
    severity: Literal["high", "medium", "low"]
    ticket_excerpt: str = Field(description="Short verbatim or near-verbatim line from the ticket that this subtask contradicts.")
    issue: str = Field(description="One-sentence explanation of the contradiction.")


class ContradictionReport(BaseModel):
    contradictions: list[Contradiction] = Field(default_factory=list)
    overall_assessment: Literal["clean", "minor_issues", "major_issues"]


_SYSTEM = """\
You are a strict reviewer. You receive a task query and a proposed tech decomposition for it.
Your job is to find any subtask that contradicts a criterion stated in the ticket.

Be conservative — only flag actual contradictions (a subtask says X, the ticket says NOT-X
or requires Y where the subtask makes Y impossible). Do not flag missing coverage or
stylistic differences.

If everything is consistent, return contradictions=[] and overall_assessment="clean".
"""


def _build_agent(settings: Settings) -> Agent[None, ContradictionReport]:
    if settings.gemini_api_key:
        os.environ["GEMINI_API_KEY"] = settings.gemini_api_key
    # Use the cheap model — this is a verification pass, not a generation task.
    model = GeminiModel(settings.enrich_model)
    return Agent(
        model=model,
        output_type=ContradictionReport,
        system_prompt=_SYSTEM,
        model_settings=ModelSettings(temperature=0.0),
    )


def _format_decomp(decomp: Decomposition) -> str:
    lines = [
        f"Overview: {decomp.overview}",
        f"Affected repos: {', '.join(decomp.affected_repos)}",
        f"Risks: {'; '.join(decomp.risks) or '(none)'}",
        f"Open questions: {'; '.join(decomp.open_questions) or '(none)'}",
        "",
        "Subtasks:",
    ]
    for i, st in enumerate(decomp.subtasks, 1):
        lines.append(f"  [{i}] ({st.repo}) {st.title}")
        if st.description:
            lines.append(f"      desc: {st.description.strip()}")
        if st.acceptance_criteria:
            for ac in st.acceptance_criteria:
                lines.append(f"      AC: {ac}")
    return "\n".join(lines)


async def check_contradictions(
    *, ticket: Ticket, decomp: Decomposition, settings: Settings
):
    agent = _build_agent(settings)
    user = (
        f"=== TICKET ===\n"
        f"Title: {ticket.title}\n\n"
        f"Body:\n{ticket.body or '(empty)'}\n\n"
        f"=== PROPOSED DECOMPOSITION ===\n"
        f"{_format_decomp(decomp)}\n"
    )
    return await agent.run(user)
