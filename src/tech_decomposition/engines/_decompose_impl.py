from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.settings import ModelSettings

from ..clients.sourcebot import AskResult, ask_sourcebot
from ..config import Settings
from ..models import Decomposition, EnrichedQuery, Ticket

_DECOMPOSE_GROUNDING_MARKDOWN = """\
## Task: grounded tech decomposition (exploration pass)

Use your codebase tools (search, read files, symbols) to ground everything in real paths and behavior.

## What to produce in this answer

1. **Overview** — What the work entails and how it shows up in the codebases involved.
2. **Concrete map** — Repos, key files (paths relative to repo roots), important symbols or flows, and cross-repo touchpoints.
3. **Risks & edge cases** — Especially anything that affects data integrity, auth, or multi-step UX.
4. **Subtask sketch** — Bullet list of implementation chunks an autonomous agent could take one at a time (title + 1–2 sentences each; name repo + files when known).

## Information gaps (required)

If something critical is still unknown after exploration, add a section with **exactly** this heading on its own line:

### Additional information needed

Then bullet the specifics (API contracts, product rules, env vars, ownership, etc.). If nothing is missing, **omit this section entirely** (do not write the heading).

Cite paths and line ranges whenever you have them. Do not invent file paths or symbol names — say what is missing instead.
"""


_STRUCTURE_SYSTEM = """\
You turn a grounded exploration answer (from a Sourcebot-style codebase session) into a structured tech decomposition.

STRICT GROUNDING:
- Every file path in `subtasks[].files` MUST appear verbatim in the SOURCE ANSWER (or in the TICKET / ENRICHMENT block). If the source only hints at a location, put the gap in `open_questions` instead of guessing a path.
- Do not invent acceptance criteria that contradict the ticket or the source answer.
- Merge bullets from a "### Additional information needed" section in the source into `open_questions` (deduplicate sensibly).

QUALITY:
- `subtasks`: each has a specific `repo`, concrete `files` when evidenced, `acceptance_criteria`, and `estimated_complexity`.
- `overview`: 2–4 sentences.
- `affected_repos`: repo keys that are actually touched in the source answer or enrichment.
- `risks`: include uncertainties called out in the source.
"""


def _format_ticket_and_enrichment(
    *,
    ticket: Ticket,
    query: EnrichedQuery,
    query_text: str,
) -> str:
    return (
        f"=== TICKET ===\n"
        f"Key: {ticket.key or '(none)'}\n"
        f"Title: {ticket.title or '(none)'}\n"
        f"URL: {ticket.url or '(none)'}\n\n"
        f"Body:\n{ticket.body or query_text}\n\n"
        f"=== ENRICHMENT ===\n"
        f"Summary: {query.summary}\n"
        f"Intent: {query.intent} (confidence: {query.confidence})\n"
        f"Suspected repos: {', '.join(query.suspected_repos) or '(unspecified)'}\n"
        f"Entities: {', '.join(query.entities) or '(none)'}\n"
        f"Code keywords: {', '.join(query.code_keywords) or '(none)'}\n"
        f"Search queries: {', '.join(query.search_queries) or '(none)'}\n"
        f"Open questions from enrichment: {'; '.join(query.open_questions) or '(none)'}\n"
    )


def _build_structure_agent(settings: Settings) -> Agent[None, Decomposition]:
    if settings.gemini_api_key:
        os.environ["GEMINI_API_KEY"] = settings.gemini_api_key
    model = GeminiModel(settings.decompose_model)
    return Agent(
        model=model,
        output_type=Decomposition,
        system_prompt=_STRUCTURE_SYSTEM,
        model_settings=ModelSettings(temperature=0.0),
    )


@dataclass
class DecomposePipelineResult:
    """Grounding (Sourcebot) + structured post-process."""

    ask: AskResult
    struct_result: Any


async def run_decompose_pipeline(
    *,
    ticket: Ticket,
    query: EnrichedQuery,
    query_text: str,
    settings: Settings,
    repos: list[str] | None,
    max_sourcebot_steps: int | None = None,
) -> DecomposePipelineResult:
    """Sourcebot exploration → structured Decomposition (second LLM pass)."""
    header = _format_ticket_and_enrichment(ticket=ticket, query=query, query_text=query_text)
    ask_question = f"{header}\n\n{_DECOMPOSE_GROUNDING_MARKDOWN}"

    ask = await ask_sourcebot(
        ask_question,
        settings=settings,
        repos=repos,
        max_steps=max_sourcebot_steps,
        timeout_seconds=settings.sourcebot_timeout_seconds,
    )

    agent = _build_structure_agent(settings)
    user = (
        f"{header}\n"
        f"=== SOURCE ANSWER (grounded exploration; use as primary evidence) ===\n"
        f"{ask.answer.strip()}\n"
    )
    struct_result = await agent.run(user)
    decomp = struct_result.output
    decomp.query = query_text
    decomp.enrichment_model = settings.enrich_model
    decomp.decomposition_model = settings.decompose_model
    return DecomposePipelineResult(ask=ask, struct_result=struct_result)
