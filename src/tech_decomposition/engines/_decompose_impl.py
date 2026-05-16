from __future__ import annotations

import os

from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.settings import ModelSettings

from ..config import Settings
from ..models import Decomposition, EnrichedQuery, RetrievedContext, Snippet


_SYSTEM = """\
You are a senior Tract engineer producing a tech decomposition for a task so that
autonomous coding agents can pick up sub-tasks and act.

GROUNDING — HARD RULES (do not violate, even if it makes the output less complete):
- Every file path you reference MUST appear verbatim in the retrieved snippets. If a needed
  path isn't there, write "requires investigation: <what>" instead of guessing a name.
- Every directory name for a NEW file you propose creating MUST follow a pattern visible in
  the snippets (e.g. if existing folders are <name>/, your new folder uses the same shape).
- Do not invent function, class, or DAG names. If you reference one, it must appear in the
  snippets verbatim.
- Do not invent acceptance criteria that contradict criteria stated in the source ticket.
  If the ticket says X, do not propose the opposite of X.

SCOPE:
- subtasks must each name a SPECIFIC repo and a SPECIFIC list of files (paths relative to that repo).
- Each subtask must be small enough that a coding agent could finish it in one focused pass.
- Use the open_questions list for anything the snippets do not answer; do not guess.
- If retrieval is empty or weak, say so in `risks` and keep subtasks minimal.

HONESTY:
- It is better to say "I don't know — needs investigation" than to invent a plausible-sounding
  name or behavior. An agent acting on invented detail will waste time discovering the truth.
- If a subtask would require a file you don't have evidence of, mark it as
  estimated_complexity="unknown" and put the missing file in open_questions.
"""


def _trim_context(ctx: RetrievedContext, max_chars: int) -> list[tuple[str, Snippet]]:
    """Pick highest-scoring snippets across repos until char budget is hit."""
    flat: list[tuple[str, Snippet]] = [
        (rc.repo, s) for rc in ctx.repos for s in rc.snippets
    ]
    flat.sort(key=lambda pair: -pair[1].score)
    used = 0
    kept: list[tuple[str, Snippet]] = []
    for repo, s in flat:
        size = len(s.content) + len(s.path) + 32
        if used + size > max_chars:
            continue
        kept.append((repo, s))
        used += size
    return kept


def _format_context(picks: list[tuple[str, Snippet]]) -> str:
    parts: list[str] = []
    for repo, s in picks:
        parts.append(
            f"--- {repo}: {s.path}:L{s.line_start}-{s.line_end} (source={s.source}) ---\n"
            f"{s.content}"
        )
    return "\n\n".join(parts) if parts else "(no snippets retrieved)"


def _build_agent(settings: Settings) -> Agent[None, Decomposition]:
    if settings.gemini_api_key:
        os.environ["GEMINI_API_KEY"] = settings.gemini_api_key
    model = GeminiModel(settings.decompose_model)
    return Agent(
        model=model,
        output_type=Decomposition,
        system_prompt=_SYSTEM,
        model_settings=ModelSettings(temperature=0.0),
    )


async def decompose(
    *,
    query_text: str,
    query: EnrichedQuery,
    context: RetrievedContext,
    settings: Settings,
):
    agent = _build_agent(settings)
    picks = _trim_context(context, settings.decompose_max_context_chars)

    user = (
        f"Ticket key: {ticket.key or '(none)'}\n"
        f"Ticket title: {ticket.title}\n"
        f"Ticket URL: {ticket.url or '(none)'}\n\n"
        f"Enriched summary: {query.summary}\n"
        f"Intent: {query.intent}\n"
        f"Suspected repos: {', '.join(query.suspected_repos) or '(unspecified)'}\n"
        f"Entities: {', '.join(query.entities) or '(none)'}\n"
        f"Open questions from enrichment: {'; '.join(query.open_questions) or '(none)'}\n\n"
        f"Retrieved snippets ({len(picks)} of {context.total_snippets}, "
        f"~{sum(len(s.content) for _, s in picks)} chars):\n\n"
        f"{_format_context(picks)}\n"
    )

    result = await agent.run(user)
    decomp = result.output
    decomp.ticket_key = decomp.ticket_key or ticket.key
    decomp.ticket_title = decomp.ticket_title or ticket.title
    decomp.ticket_url = decomp.ticket_url or ticket.url
    decomp.enrichment_model = settings.enrich_model
    decomp.decomposition_model = settings.decompose_model
    return result
