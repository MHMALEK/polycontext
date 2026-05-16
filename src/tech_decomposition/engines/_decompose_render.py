from __future__ import annotations

from datetime import datetime
from pathlib import Path

from ..config import Settings
from ..models import Decomposition, EnrichedQuery, RetrievedContext, Snippet


def _gitlab_link(settings: Settings, repo: str, path: str, ref: str, line: int | None) -> str | None:
    project = settings.gitlab_projects.get(repo)
    if not project:
        return None
    base = settings.gitlab_base_url.rstrip("/")
    suffix = f"#L{line}" if line else ""
    return f"{base}/{project}/-/blob/{ref}/{path}{suffix}"


def attach_gitlab_links(
    decomp: Decomposition, ctx: RetrievedContext, settings: Settings
) -> Decomposition:
    """Populate Subtask.file_links from configured GitLab project paths and per-repo HEAD."""
    head_by_repo = {rc.repo: rc.head_sha for rc in ctx.repos}
    for st in decomp.subtasks:
        head = head_by_repo.get(st.repo, "HEAD")
        st.file_links = [
            link for f in st.files
            if (link := _gitlab_link(settings, st.repo, f, head, None))
        ]
    return decomp


def render_markdown(
    *,
    decomp: Decomposition,
    enriched: EnrichedQuery,
    ctx: RetrievedContext,
    settings: Settings,
) -> str:
    head_by_repo = {rc.repo: rc.head_sha for rc in ctx.repos}
    lines: list[str] = []
    lines.append(f"# {decomp.query}")
    lines.append("")
    lines.append(f"_Generated {decomp.generated_at.isoformat(timespec='seconds')}Z "
                 f"— enrich={decomp.enrichment_model}, decompose={decomp.decomposition_model}_")
    lines.append("")

    lines.append("## Overview")
    lines.append(decomp.overview.strip() or "_(none)_")
    lines.append("")

    lines.append("## Affected repos")
    for r in decomp.affected_repos:
        head = head_by_repo.get(r, "HEAD")
        path = settings.gitlab_projects.get(r)
        if path:
            url = f"{settings.gitlab_base_url.rstrip('/')}/{path}/-/tree/{head}"
            lines.append(f"- [`{r}`]({url}) @ `{head[:8]}`")
        else:
            lines.append(f"- `{r}` @ `{head[:8]}`")
    lines.append("")

    if decomp.risks:
        lines.append("## Risks")
        for r in decomp.risks:
            lines.append(f"- {r}")
        lines.append("")

    if decomp.open_questions:
        lines.append("## Open questions")
        for q in decomp.open_questions:
            lines.append(f"- {q}")
        lines.append("")

    lines.append("## Subtasks")
    for i, st in enumerate(decomp.subtasks, 1):
        lines.append(f"### {i}. {st.title}  _(repo: `{st.repo}`, complexity: {st.estimated_complexity})_")
        if st.description:
            lines.append("")
            lines.append(st.description.strip())
        if st.file_links:
            lines.append("")
            lines.append("**Files:**")
            for f, link in zip(st.files, st.file_links):
                lines.append(f"- [`{f}`]({link})")
        elif st.files:
            lines.append("")
            lines.append("**Files:**")
            for f in st.files:
                lines.append(f"- `{f}`")
        if st.acceptance_criteria:
            lines.append("")
            lines.append("**Acceptance criteria:**")
            for c in st.acceptance_criteria:
                lines.append(f"- [ ] {c}")
        lines.append("")

    lines.append("---")
    lines.append("## Retrieval debug")
    lines.append(f"- enrichment intent: `{enriched.intent}`, confidence: `{enriched.confidence}`")
    lines.append(f"- search queries: {', '.join(f'`{q}`' for q in enriched.search_queries) or '_(none)_'}")
    lines.append(f"- entities: {', '.join(f'`{e}`' for e in enriched.entities) or '_(none)_'}")
    lines.append(f"- code keywords: {', '.join(f'`{k}`' for k in enriched.code_keywords) or '_(none)_'}")
    lines.append(f"- snippets returned: {ctx.total_snippets} ({ctx.total_chars:,} chars)")
    for rc in ctx.repos:
        from collections import Counter
        by_src = Counter(s.source for s in rc.snippets)
        breakdown = ", ".join(f"{src}={n}" for src, n in sorted(by_src.items())) or "—"
        lines.append(f"  - `{rc.repo}` @ `{rc.head_sha[:8]}`: {len(rc.snippets)} total ({breakdown})")

    return "\n".join(lines).rstrip() + "\n"


def write_markdown(markdown: str, decomp: Decomposition, settings: Settings) -> Path:
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out = settings.output_dir / f"{stamp}-query.md"
    out.write_text(markdown)
    return out


def snippet_link(s: Snippet, settings: Settings, head: str) -> str | None:
    """Helper for ad-hoc snippet→GitLab links if you want them in the markdown."""
    project = settings.gitlab_projects.get(s.repo)
    if not project:
        return None
    base = settings.gitlab_base_url.rstrip("/")
    return f"{base}/{project}/-/blob/{head}/{s.path}#L{s.line_start}-{s.line_end}"
