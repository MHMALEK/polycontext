/**
 * Tiny Jira Cloud REST v3 client.
 *
 * We only need two endpoints:
 *   GET  /rest/api/3/issue/{key}            — fetch ticket
 *   POST /rest/api/3/issue/{key}/comment    — add comment
 *
 * Auth: HTTP Basic, base64(email:apiToken). API tokens are created at
 * https://id.atlassian.com/manage-profile/security/api-tokens — they're
 * the only credential that works for cloud Basic auth.
 *
 * Bodies in v3 use Atlassian Document Format (ADF). The helper in
 * src/adf.ts converts markdown → minimal ADF so callers can stay in
 * markdown land.
 */

import type { Config } from "./config.js";

export type JiraIssue = {
  key: string;
  summary: string;
  description: string;
  issueType: string;
  status: string;
  raw: Record<string, unknown>;
};

export type AdfDoc = {
  type: "doc";
  version: 1;
  content: unknown[];
};

export class JiraClient {
  private readonly baseUrl: string;
  private readonly authHeader: string;

  constructor(cfg: Config["jira"]) {
    this.baseUrl = cfg.baseUrl;
    this.authHeader =
      "Basic " + Buffer.from(`${cfg.email}:${cfg.apiToken}`).toString("base64");
  }

  /** Fetch an issue by key (e.g. "TRT-123"). */
  async getIssue(key: string): Promise<JiraIssue> {
    const url = `${this.baseUrl}/rest/api/3/issue/${encodeURIComponent(key)}`;
    const r = await fetch(url, {
      headers: {
        Authorization: this.authHeader,
        Accept: "application/json",
      },
    });
    if (!r.ok) {
      const body = await r.text();
      throw new Error(
        `jira GET /issue/${key} → ${r.status}: ${body.slice(0, 400)}`,
      );
    }
    const data = (await r.json()) as {
      key: string;
      fields: {
        summary?: string;
        description?: unknown;
        issuetype?: { name?: string };
        status?: { name?: string };
      };
    };
    return {
      key: data.key,
      summary: data.fields.summary ?? "",
      description: adfToPlainText(data.fields.description),
      issueType: data.fields.issuetype?.name ?? "",
      status: data.fields.status?.name ?? "",
      raw: data as unknown as Record<string, unknown>,
    };
  }

  /** Post a comment whose body is given as an ADF document. */
  async addComment(key: string, adfBody: AdfDoc): Promise<{ id: string }> {
    const url = `${this.baseUrl}/rest/api/3/issue/${encodeURIComponent(
      key,
    )}/comment`;
    const r = await fetch(url, {
      method: "POST",
      headers: {
        Authorization: this.authHeader,
        Accept: "application/json",
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ body: adfBody }),
    });
    if (!r.ok) {
      const body = await r.text();
      throw new Error(
        `jira POST /issue/${key}/comment → ${r.status}: ${body.slice(0, 400)}`,
      );
    }
    const data = (await r.json()) as { id: string };
    return { id: data.id };
  }
}

/**
 * Walk an ADF document and concatenate any text nodes into one big plaintext
 * string. Good enough for piping a Jira description into an LLM prompt — we
 * don't need to preserve formatting on the way IN.
 */
export function adfToPlainText(doc: unknown): string {
  if (typeof doc === "string") return doc; // legacy v2 wiki markup
  if (!doc || typeof doc !== "object") return "";
  const parts: string[] = [];
  const walk = (node: unknown): void => {
    if (!node || typeof node !== "object") return;
    const n = node as { type?: string; text?: string; content?: unknown[] };
    if (n.type === "text" && typeof n.text === "string") {
      parts.push(n.text);
    }
    if (n.type === "paragraph" || n.type === "heading") {
      const before = parts.length;
      for (const c of n.content ?? []) walk(c);
      if (parts.length > before) parts.push("\n");
    } else if (n.type === "bulletList" || n.type === "orderedList") {
      for (const c of n.content ?? []) walk(c);
    } else if (n.type === "listItem") {
      parts.push("- ");
      for (const c of n.content ?? []) walk(c);
    } else {
      for (const c of n.content ?? []) walk(c);
    }
  };
  walk(doc);
  return parts.join("").replace(/\n{3,}/g, "\n\n").trim();
}
