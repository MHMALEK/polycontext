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
