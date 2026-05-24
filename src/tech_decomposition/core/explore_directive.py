"""Shared "must-explore" directive injected INTO the user prompt.

Why this lives next to the user message instead of in the SDK's system
parameter:

We observed both Claude Opus 4.7 and Qwen3-235B reply with "Could you
clarify what you mean by X?" to short queries like "what is roles?" —
zero tool calls, model gave up immediately. The session.prompt's
``system`` parameter wasn't reaching them: a 200-token system prompt
yielded only 6 input tokens in the response, meaning the agent's baked-in
prompt overrode ours entirely.

Putting the directive INTO the user message guarantees it travels with
the query. The model can't ignore message-level instructions the same
way it can ignore a swapped-out system prompt.

Apply this wrapping when:
  - the adapter has tools available (tools_enabled=True) — otherwise the
    directive is moot, the model has nothing to explore with
  - grounding hasn't already prepended snippets (grounded=False) — when
    snippets are present the model already has context and doesn't need
    a "go explore" nudge
"""
from __future__ import annotations


_EXPLORE_DIRECTIVE = (
    "[Workspace Q&A — instructions]\n"
    "You have read-only tools (read, grep, glob, ls) over a multi-repo "
    "monorepo. Before answering this question:\n"
    "\n"
    "1. DO NOT ask the user for clarification. If the question is short or "
    "vague (e.g. 'what is roles', 'how does upload work'), interpret it as "
    "a request to find that concept in the codebase.\n"
    "2. Use ``glob`` or ``grep`` FIRST to locate relevant files. At least "
    "one tool call is required.\n"
    "3. Read 2-3 files to confirm before writing your answer.\n"
    "4. Cite file paths and line numbers.\n"
    "5. Only if you searched thoroughly and found nothing, say so with the "
    "search terms you tried — never punt with 'could you clarify?'.\n"
    "\n"
    "[User question]\n"
)


def wrap_with_explore_directive(query: str, *, tools_on: bool) -> str:
    """Prepend the must-explore directive to ``query`` when tools are on.

    Returns ``query`` unchanged when ``tools_on=False`` — there's no point
    telling a no-tool model to use its tools.
    """
    if not tools_on:
        return query
    return _EXPLORE_DIRECTIVE + query.strip()
