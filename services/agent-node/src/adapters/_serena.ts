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

type SerenaSseConfig = {
  type: "sse";
  url: string;
  headers?: Record<string, string>;
};

/**
 * Returns an MCP `Record<name, config>` when `SERENA_URL` is set,
 * otherwise `undefined`. The shape ({ type: "sse", url, headers? })
 * is the intersection accepted by both Cursor and Claude SDKs.
 */
export function serenaMcpConfig(): Record<string, SerenaSseConfig> | undefined {
  const url = (process.env.SERENA_URL || "").trim();
  if (!url) return undefined;
  const apiKey = (process.env.SERENA_API_KEY || "").trim();
  const cfg: SerenaSseConfig = { type: "sse", url };
  if (apiKey) cfg.headers = { Authorization: `Bearer ${apiKey}` };
  return { serena: cfg };
}
