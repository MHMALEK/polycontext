/**
 * The actual flow: fetch a Jira ticket, decompose it via tech-decomposition,
 * post the result back as a comment.
 *
 * Both `cli.ts` and `server.ts` call into this. Keeping the logic here makes
 * it testable in isolation and ensures the CLI and HTTP entry behave the same.
 */

import { decompositionToAdf } from "./adf.js";
import { type Config } from "./config.js";
import { JiraClient } from "./jira.js";
import { TechDecompClient } from "./tech_decomp.js";

export type RunResult = {
  ticket: { key: string; summary: string; status: string; issueType: string };
  decompose: {
    adapter: string;
    model?: string;
    durationMs?: number;
    tokensIn?: number;
    tokensOut?: number;
    costUsd?: number;
    subtaskCount: number;
    affectedRepos: string[];
  };
  comment: { id: string; url: string };
};

/**
 * Build the `query` we send to tech-decomposition.
 *
 * Jira gives us summary + description; the LLM does better when both are
 * present in one block with the ticket key for context.
 */
function buildQueryFromTicket(t: {
  key: string;
  summary: string;
  description: string;
}): string {
  const parts: string[] = [`[${t.key}] ${t.summary}`.trim()];
  if (t.description?.trim()) {
    parts.push("", t.description.trim());
  }
  return parts.join("\n");
}

export async function decomposeTicket(
  config: Config,
  args: { ticketKey: string },
): Promise<RunResult> {
  const jira = new JiraClient(config.jira);
  const td = new TechDecompClient(config.techDecomp);

  // 1. Fetch the ticket
  const issue = await jira.getIssue(args.ticketKey);
  if (!issue.summary && !issue.description) {
    throw new Error(
      `ticket ${issue.key} has neither summary nor description — nothing to decompose`,
    );
  }

  // 2. Decompose
  const query = buildQueryFromTicket(issue);
  const r = await td.decompose({ query });
  const d = r.result.decomposition;
  const m = r.result.metrics;

  // 3. Post as comment (ADF)
  const adf = decompositionToAdf({
    d,
    adapter: r.result.adapter,
    model: m.model,
    durationMs: m.duration_ms,
  });
  const comment = await jira.addComment(issue.key, adf);

  return {
    ticket: {
      key: issue.key,
      summary: issue.summary,
      status: issue.status,
      issueType: issue.issueType,
    },
    decompose: {
      adapter: r.result.adapter,
      model: m.model,
      durationMs: m.duration_ms,
      tokensIn: m.tokens_in,
      tokensOut: m.tokens_out,
      costUsd: m.cost_usd,
      subtaskCount: d.subtasks.length,
      affectedRepos: d.affected_repos,
    },
    comment: {
      id: comment.id,
      url: `${config.jira.baseUrl}/browse/${issue.key}?focusedCommentId=${comment.id}`,
    },
  };
}
