/**
 * Serena MCP config builder.
 *
 * Serena (https://github.com/oraios/serena) is an LSP-backed MCP server
 * that exposes semantic code tools — `find_symbol`, `find_references`,
 * `get_symbols_overview`, `search_for_pattern`, etc. — that go beyond
 * the plain-text grep/read tools each adapter already has.
 *
 * Expected deployment: run Serena out-of-band (e.g. `uvx --from
 * git+https://github.com/oraios/serena serena-mcp-server --transport sse
 * --port 9121 --context ide-assistant --project <REPOS_ROOT>`) and point
 * `SERENA_URL` at its SSE endpoint. When unset, MCP wiring is a no-op
 * and adapters fall back to their built-in workspace tools.
 *
 * Only wired into adapters whose SDK accepts `mcpServers` natively:
 * Cursor (`Agent.create({ mcpServers })`) and Claude Agent SDK
 * (`query({ options: { mcpServers } })`). Cline uses `createTool()`
 * (not MCP) and Gemini's `@google/genai` has its own function-call
 * protocol — both are out of scope here.
 */

type SerenaHttpLikeConfig = {
  type: "sse" | "http";
  url: string;
  headers?: Record<string, string>;
};

export type SerenaEnv = {
  url: string;
  /** Inferred from URL path: `/sse` → SSE, anything else → streamable HTTP. */
  transport: "sse" | "http";
  apiKey: string | undefined;
};

/**
 * Parsed Serena environment, or `undefined` when `SERENA_URL` is unset.
 *
 * Adapters whose SDK uses class-based MCP wiring (OpenAI Agents SDK) call
 * this and build their own `MCPServerStreamableHttp` / `MCPServerSSE`
 * instance. Adapters whose SDK takes a JSON config (Cursor, Claude Agent)
 * use `serenaMcpConfig()` below.
 */
export function serenaEnv(): SerenaEnv | undefined {
  const url = (process.env.SERENA_URL || "").trim();
  if (!url) return undefined;
  const transport: "sse" | "http" = url.endsWith("/sse") ? "sse" : "http";
  const apiKey = (process.env.SERENA_API_KEY || "").trim() || undefined;
  return { url, transport, apiKey };
}

/**
 * MCP config object accepted by both Cursor SDK and Claude Agent SDK
 * (`mcpServers?: Record<name, { type: "sse" | "http", url, headers? }>`).
 */
export function serenaMcpConfig(): Record<string, SerenaHttpLikeConfig> | undefined {
  const env = serenaEnv();
  if (!env) return undefined;
  const cfg: SerenaHttpLikeConfig = { type: env.transport, url: env.url };
  if (env.apiKey) cfg.headers = { Authorization: `Bearer ${env.apiKey}` };
  return { serena: cfg };
}
