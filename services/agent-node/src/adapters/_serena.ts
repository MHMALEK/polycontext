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

/**
 * Transport is inferred from the URL: paths ending in `/sse` use the (legacy
 * but still supported) SSE transport; anything else uses streamable HTTP,
 * which is Serena's recommended transport going forward.
 *
 * Both Cursor SDK (`{ type: "sse" | "http", url }`) and Claude Agent SDK
 * (`{ type: "sse" | "http", url }`) accept the same shape, so a single
 * helper feeds both adapters.
 */
export function serenaMcpConfig(): Record<string, SerenaHttpLikeConfig> | undefined {
  const url = (process.env.SERENA_URL || "").trim();
  if (!url) return undefined;
  const transport: "sse" | "http" = url.endsWith("/sse") ? "sse" : "http";
  const apiKey = (process.env.SERENA_API_KEY || "").trim();
  const cfg: SerenaHttpLikeConfig = { type: transport, url };
  if (apiKey) cfg.headers = { Authorization: `Bearer ${apiKey}` };
  return { serena: cfg };
}
