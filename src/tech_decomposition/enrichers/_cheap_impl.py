from __future__ import annotations

import os

from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.settings import ModelSettings

from ..config import Settings
from ..models import EnrichedQuery, Ticket

_SYSTEM = """\
You are a senior Tract engineer triaging a Jira ticket so an autonomous coding agent
can act on it. Your job is to produce a STRUCTURED, KEYWORD-DENSE rewrite that drives
multi-repo code search downstream.

Rules:
- Be specific. Prefer concrete code identifiers (function/class/env-var/table names) over prose.
- If the ticket is too vague to act on, set confidence=low and put the missing info in open_questions.
- suspected_repos must be a subset of the repo list provided in the user message.
- search_queries should be the actual strings you'd type into a code search box — short, distinctive.
- Do not invent symbols that aren't suggested by the ticket text.
"""


_QUERY_SYSTEM = """\
You are a senior Tract engineer analyzing a codebase question. Your job is to produce a STRUCTURED, KEYWORD-DENSE rewrite that drives multi-repo code search downstream.

Rules:
- Be specific. Prefer concrete code identifiers (function/class/env-var/table names) over prose.
- suspected_repos must be a subset of the repo list provided in the user message.
- search_queries should be the actual strings you'd type into a code search box — short, distinctive.
- Do not invent symbols that aren't suggested by the query text.
"""

def _build_agent(settings: Settings) -> Agent[None, EnrichedQuery]:
    if settings.gemini_api_key:
        os.environ["GEMINI_API_KEY"] = settings.gemini_api_key
    model = GeminiModel(settings.enrich_model)
    return Agent(
        model=model,
        output_type=EnrichedQuery,
        system_prompt=_SYSTEM,
        model_settings=ModelSettings(temperature=0.0),
    )


def _build_query_agent(settings: Settings) -> Agent[None, EnrichedQuery]:
    if settings.gemini_api_key:
        os.environ["GEMINI_API_KEY"] = settings.gemini_api_key
    model = GeminiModel(settings.enrich_model)
    return Agent(
        model=model,
        output_type=EnrichedQuery,
        system_prompt=_QUERY_SYSTEM,
        model_settings=ModelSettings(temperature=0.0),
    )


async def enrich_query(query: str, repos: list[str], settings: Settings):
    agent = _build_query_agent(settings)
    user = (
        f"Available repos: {', '.join(repos)}\n\n"
        f"User query:\n{query}"
    )
    return await agent.run(user)


async def enrich_ticket(ticket: Ticket, repos: list[str], settings: Settings):
    agent = _build_agent(settings)
    user = (
        f"Available repos: {', '.join(repos)}\n\n"
        f"Ticket title: {ticket.title}\n"
        f"Labels: {', '.join(ticket.labels) or '(none)'}\n"
        f"Components: {', '.join(ticket.components) or '(none)'}\n\n"
        f"Ticket body:\n{ticket.body or '(empty)'}"
    )
    return await agent.run(user)
