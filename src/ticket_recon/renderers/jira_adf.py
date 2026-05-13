"""JiraADFRenderer: convert markdown to Atlassian Document Format JSON."""
from __future__ import annotations

import json

from ..core.context import RunContext
from ..core.protocols import EngineResult, Rendered
from ._adf import markdown_to_adf


class JiraADFRenderer:
    name = "jira_adf"

    def render(self, result: EngineResult, ctx: RunContext) -> Rendered:
        adf = markdown_to_adf(result.answer_markdown)
        return Rendered(
            format="jira_adf",
            body=json.dumps(adf),
            metadata={"engine": result.engine, **result.extra},
        )
