from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class Ticket(BaseModel):
    key: str | None = None
    title: str | None = None
    url: str | None = None
    body: str | None = None
    labels: list[str] = Field(default_factory=list)
    components: list[str] = Field(default_factory=list)

class EnrichedQuery(BaseModel):
    """Output of the cheap rewrite step. Drives retrieval."""
    summary: str = Field(description="One-paragraph plain-English summary of the query.")
    intent: Literal["bug", "feature", "refactor", "investigation", "chore", "unknown"]
    entities: list[str] = Field(
        default_factory=list,
        description="Domain nouns from the ticket (e.g. 'supplier_node', 'data_sharing').",
    )
    code_keywords: list[str] = Field(
        default_factory=list,
        description="Symbol-like or distinctive code tokens to grep for (e.g. function names, env vars).",
    )
    suspected_repos: list[str] = Field(
        default_factory=list,
        description="Subset of available repos most likely affected.",
    )
    search_queries: list[str] = Field(
        default_factory=list,
        description="2-6 keyword queries to fan out to retrievers.",
    )
    open_questions: list[str] = Field(
        default_factory=list,
        description="Things the ticket does not specify and that an engineer would need to clarify.",
    )
    confidence: Literal["low", "medium", "high"] = "medium"


class Snippet(BaseModel):
    """A single retrieved hit. Repo-relative path; permalinks built later."""
    repo: str
    path: str                      # repo-relative path
    line_start: int
    line_end: int
    content: str
    score: float = 0.0
    source: Literal["ripgrep", "sourcebot", "anchor", "local_chromadb"] = "ripgrep"

    def gitlab_url(self, base_url: str, project_path: str, ref: str) -> str:
        return f"{base_url}/{project_path}/-/blob/{ref}/{self.path}#L{self.line_start}-{self.line_end}"


class RepoContext(BaseModel):
    """Per-repo retrieval result."""
    repo: str
    head_sha: str
    snippets: list[Snippet] = Field(default_factory=list)


class RetrievedContext(BaseModel):
    """Aggregated retrieval result across repos."""
    repos: list[RepoContext] = Field(default_factory=list)
    total_snippets: int = 0
    total_chars: int = 0
    grounding: Literal["local_snippets", "sourcebot_chat"] = "local_snippets"


class Subtask(BaseModel):
    """A single agent-pickup-able unit of work."""
    title: str
    description: str
    repo: str
    files: list[str] = Field(default_factory=list)        # repo-relative
    file_links: list[str] = Field(default_factory=list)   # GitLab permalinks
    acceptance_criteria: list[str] = Field(default_factory=list)
    estimated_complexity: Literal["small", "medium", "large", "unknown"] = "unknown"


class Decomposition(BaseModel):
    """Final structured output."""
    query: str

    overview: str = Field(description="2-4 sentence engineer-readable framing of what needs to happen.")
    affected_repos: list[str]
    risks: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    subtasks: list[Subtask]

    generated_at: datetime = Field(default_factory=datetime.utcnow)
    enrichment_model: str = ""
    decomposition_model: str = ""

    def to_markdown(self) -> str:
        """Render the decomposition as Jira/Linear/GitHub-friendly markdown.

        Used as the ``markdown`` field on ``AdapterDecomposeResult`` so the UI's
        "Raw markdown" panel and any downstream renderer see a clean,
        structured document — not the raw upstream LLM output. Mirrors the
        shape produced by the web UI's ``toJiraMarkdown`` helper so the
        copy-to-clipboard and the rendered panel emit identical text.
        """
        lines: list[str] = []
        if self.overview.strip():
            lines.extend(["## Overview", "", self.overview.strip(), ""])
        if self.affected_repos:
            repos_inline = ", ".join(f"`{r}`" for r in self.affected_repos)
            lines.extend([f"**Affected repos:** {repos_inline}", ""])
        lines.extend([f"## Subtasks ({len(self.subtasks)})", ""])
        for i, st in enumerate(self.subtasks):
            tag_bits: list[str] = []
            if st.repo:
                tag_bits.append(st.repo)
            if st.estimated_complexity and st.estimated_complexity != "unknown":
                tag_bits.append(st.estimated_complexity)
            tag = f"  *({', '.join(tag_bits)})*" if tag_bits else ""
            lines.append(f"### {str(i + 1).zfill(2)} — {st.title}{tag}")
            lines.append("")
            if st.description.strip():
                lines.append(st.description.strip())
                lines.append("")
            if st.files:
                files_inline = ", ".join(f"`{f}`" for f in st.files)
                lines.extend([f"**Files:** {files_inline}", ""])
            if st.acceptance_criteria:
                lines.append("**Acceptance criteria:**")
                lines.extend(f"- {ac}" for ac in st.acceptance_criteria)
                lines.append("")
        if self.risks:
            lines.extend([f"## Risks ({len(self.risks)})", ""])
            lines.extend(f"- {r}" for r in self.risks)
            lines.append("")
        if self.open_questions:
            lines.extend([f"## Open questions ({len(self.open_questions)})", ""])
            lines.extend(f"- {q}" for q in self.open_questions)
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"


class DecomposeRequest(BaseModel):
    """API input."""
    query: str | None = Field(
        default=None,
        description="The natural language question or task.",
    )
    repos: list[str] | None = Field(
        default=None,
        description="Override the configured repo list for this request.",
    )
    mode: Literal["cheap", "deep", "auto"] = Field(
        default="auto",
        description="Legacy field ignored by the main pipeline (Sourcebot + structure pass).",
    )


class DecomposeResponse(BaseModel):
    decomposition: Decomposition
    enriched_query: EnrichedQuery
    markdown_path: str
    markdown: str
    metrics: dict | None = None
