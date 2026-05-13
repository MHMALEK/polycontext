"""JiraCommentSink: posts the JiraADF rendering as a comment on the ticket.

Reads the ticket key from the rendered metadata (set by JiraADFRenderer, which
in turn copies it from EngineResult.extra). Silently no-ops with a log if the
key is missing rather than failing the whole pipeline — sinks should be
forgiving of inputs the engine couldn't populate.
"""
from __future__ import annotations

import json
import logging

from ..core.context import RunContext
from ..core.protocols import Rendered, SinkResult
from ..clients.jira import post_comment

log = logging.getLogger(__name__)


class JiraCommentSink:
    name = "jira_comment"
    prefers = "jira_adf"

    async def deliver(self, rendered: Rendered, ctx: RunContext) -> SinkResult:
        ticket_key = rendered.metadata.get("ticket_key")
        if not ticket_key:
            log.warning("JiraCommentSink: no ticket_key in rendered metadata; skipping")
            return SinkResult(sink=self.name, location=None, extra={"skipped": True})
        if rendered.format != "jira_adf":
            log.warning(
                "JiraCommentSink: expected jira_adf rendering, got %r; skipping",
                rendered.format,
            )
            return SinkResult(sink=self.name, location=None, extra={"skipped": True})
        try:
            adf = json.loads(rendered.body)
        except json.JSONDecodeError as e:
            log.warning("JiraCommentSink: invalid ADF JSON: %s", e)
            return SinkResult(sink=self.name, location=None, extra={"skipped": True})
        try:
            comment_id = await post_comment(
                settings=ctx.settings, key=ticket_key, adf_body=adf,
            )
        except Exception as e:
            log.warning("post_to_jira failed: %s", e)
            return SinkResult(sink=self.name, location=None, extra={"error": str(e)})
        return SinkResult(
            sink=self.name,
            location=f"jira:{ticket_key}/comment/{comment_id}",
            extra={"ticket_key": ticket_key, "comment_id": comment_id},
        )
