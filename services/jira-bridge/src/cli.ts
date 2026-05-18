/**
 * jira-bridge CLI — `tsx src/cli.ts TRT-123 [TRT-124 ...]`
 *
 * For each ticket key: fetch from Jira → run /v1/adapters/.../decompose →
 * post the result as a comment → print a one-line summary to stdout.
 *
 * Exit code: non-zero if ANY ticket failed, zero if all succeeded.
 */

import { loadConfig } from "./config.js";
import { decomposeTicket } from "./decompose_ticket.js";

const HELP = `\
jira-bridge — fetch a Jira ticket, run it through tech-decomposition, post
the result as a comment.

Usage:
  tsx src/cli.ts TICKET_KEY [TICKET_KEY ...]
  npm run cli -- TICKET_KEY [TICKET_KEY ...]

Env (see .env.example):
  JIRA_BASE_URL, JIRA_EMAIL, JIRA_API_TOKEN
  TECH_DECOMP_URL, TECH_DECOMP_ADAPTER

Example:
  npm run cli -- TRT-123 TRT-456
`;

async function main(): Promise<number> {
  const args = process.argv.slice(2);
  if (args.length === 0 || args[0] === "--help" || args[0] === "-h") {
    process.stdout.write(HELP);
    return args.length === 0 ? 2 : 0;
  }

  let config;
  try {
    config = loadConfig();
  } catch (e) {
    process.stderr.write(`config error: ${(e as Error).message}\n`);
    return 2;
  }

  let failed = 0;
  for (const key of args) {
    const k = key.trim().toUpperCase();
    process.stdout.write(`→ ${k}: fetching + decomposing…\n`);
    try {
      const result = await decomposeTicket(config, { ticketKey: k });
      process.stdout.write(
        `  ok: ${result.decompose.subtaskCount} subtasks across ` +
          `${result.decompose.affectedRepos.length} repos, ` +
          `${result.decompose.adapter}` +
          (result.decompose.model ? `/${result.decompose.model}` : "") +
          (typeof result.decompose.durationMs === "number"
            ? `, ${(result.decompose.durationMs / 1000).toFixed(1)}s`
            : "") +
          "\n",
      );
      process.stdout.write(`  comment: ${result.comment.url}\n`);
    } catch (e) {
      failed += 1;
      process.stderr.write(`  fail: ${(e as Error).message}\n`);
    }
  }

  return failed === 0 ? 0 : 1;
}

main().then(
  (code) => process.exit(code),
  (e) => {
    process.stderr.write(`fatal: ${(e as Error).message}\n`);
    process.exit(1);
  },
);
