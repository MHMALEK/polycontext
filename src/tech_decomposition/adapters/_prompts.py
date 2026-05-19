"""Shared prompt preambles and ticket formatting for adapter paths."""
from __future__ import annotations

from .base import AdapterDecomposeInput

DECOMPOSE_PREAMBLE = """You are decomposing an engineering ticket into a tech
decomposition. Use file-reading tools to ground every reference; never invent
paths.

Output ONE JSON object — the *outer* `Decomposition` object — and nothing else
(no prose, no fences, no leading or trailing text). The outer object is
REQUIRED even when there is only one subtask. Returning a bare subtask object
(starting with `{{"title": ...}}`) is invalid and will be rejected.

The outer shape MUST be:

{{
  "query": "<echo the user query>",
  "overview": "2-4 sentences framing the work",
  "affected_repos": ["repo-a", "repo-b"],
  "risks": ["..."],
  "open_questions": ["..."],
  "subtasks": [
    {{
      "title": "...",
      "description": "...",
      "repo": "...",
      "files": ["..."],
      "file_links": ["..."],
      "acceptance_criteria": ["..."],
      "estimated_complexity": "small" | "medium" | "large" | "unknown"
    }}
  ],
  "enrichment_model": "",
  "decomposition_model": "{model_tag}"
}}

Rules:
  - Top-level keys: query, overview, affected_repos, risks, open_questions,
    subtasks, enrichment_model, decomposition_model. ALL of these must be
    present; use empty strings or empty arrays when you have nothing to say.
  - `subtasks` is an array. Even one subtask must be wrapped: `"subtasks":
    [ {{...}} ]` — never a bare object at the top level.
  - Output the JSON only. Do not summarize first, do not narrate, do not wrap
    in ```json fences.

Query:
"""


def query_blob(inp: AdapterDecomposeInput) -> str:
    parts: list[str] = []
    if inp.query:
        parts.append(inp.query)
    if inp.repos:
        parts.append(f"Repos to consider: {', '.join(inp.repos)}")
    return "\n\n".join(parts)
