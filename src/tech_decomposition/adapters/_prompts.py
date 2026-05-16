"""Shared prompt preambles and ticket formatting for adapter paths."""
from __future__ import annotations

from .base import AdapterDecomposeInput, AdapterImplementInput

ASK_PREAMBLE = """Answer the following question about the codebase under this working
directory. Ground every claim in real code; do not speculate.

Output format (Markdown):
1) End-to-end flow
2) Repo-by-repo responsibilities
3) Validation, persistence, and async/background processing
4) User-visible statuses/errors and where they come from
5) Gaps/unknowns (if any)

Citation format: `repo/path:Lstart-Lend` inline with each claim.
Return only the final answer. Do not include progress narration
(for example: "Searching...", "I am now...", "Next I will...").

Question:
"""


DECOMPOSE_PREAMBLE = """Decompose this query into a tech decomposition. Use
file-reading tools to ground every reference; never invent paths.

Output ONE JSON object matching this schema and nothing else (no prose, no fences):

{{
  "query": str,
  "overview": str, "affected_repos": [str], "risks": [str], "open_questions": [str],
  "subtasks": [
    {{"title": str, "description": str, "repo": str, "files": [str],
      "file_links": [str], "acceptance_criteria": [str],
      "estimated_complexity": "small"|"medium"|"large"|"unknown"}}
  ],
  "enrichment_model": "", "decomposition_model": "{model_tag}"
}}

Query:
"""


IMPLEMENT_PREAMBLE = """You are implementing a single subtask inside a clean git
worktree at this working directory. Edit only files inside this directory. Do not
commit, push, or open MRs.

Subtask:
"""


def query_blob(inp: AdapterDecomposeInput) -> str:
    parts: list[str] = []
    if inp.query:
        parts.append(inp.query)
    if inp.repos:
        parts.append(f"Repos to consider: {', '.join(inp.repos)}")
    return "\n\n".join(parts)


def subtask_prompt(inp: AdapterImplementInput) -> str:
    if inp.subtask:
        st = inp.subtask
        out = [f"# {st.title}", "", st.description]
        if st.acceptance_criteria:
            out.append("")
            out.append("Acceptance criteria:")
            out.extend(f"- {ac}" for ac in st.acceptance_criteria)
        return "\n".join(out)
    return inp.free_text or "Implement the requested task."
