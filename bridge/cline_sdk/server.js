/**
 * cline-sdk-bridge — tiny HTTP server wrapping @cline/sdk.
 *
 * Why: the Python adapter in src/tech_decomposition/adapters/_cline_sdk.py
 * can't import a Node-only SDK directly. This server exposes the SDK over
 * HTTP so the Python adapter can call it like it calls OpenHands.
 *
 * Endpoints:
 *   GET  /health          → {ok: true}
 *   POST /run             → run a single prompt to completion
 *
 * /run body shape:
 *   {
 *     systemPrompt?: string,    // optional override
 *     prompt:        string,
 *     cwd?:          string,    // working directory the agent sees
 *     providerId:    string,    // e.g. "google", "anthropic", "openai"
 *     modelId:       string,    // e.g. "gemini-2.5-pro"
 *     apiKey:        string,    // provider key
 *     maxIterations?: number,   // cap on agent turns; default 30
 *     timeoutSec?:   number,    // hard cap; default 600
 *   }
 *
 * /run response:
 *   {
 *     ok: boolean,
 *     answer: string,                  // final assistant text (concatenated)
 *     tokensIn?: number,
 *     tokensOut?: number,
 *     costUsd?: number,
 *     events: { type: string, ... }[], // raw event tail (last 20) for debugging
 *     error?: string,
 *   }
 *
 * Keeping the bridge intentionally dumb: no tool registration, no session
 * persistence, no streaming to the client. Add those in v2 once the basic
 * shape is validated end-to-end through the bake-off.
 */
import Fastify from "fastify";
import { Agent, createTool } from "@cline/sdk";


/**
 * find_code tool: lets the agent call into our existing Sourcebot index.
 *
 * Sourcebot lives outside this container (on the host's :3000 by default).
 * We reach it via ``SOURCEBOT_URL`` env var — on Docker Desktop the value
 * ``http://host.docker.internal:3000`` resolves to the host. The compose
 * file also adds ``host.docker.internal:host-gateway`` for Linux parity.
 *
 * Returns a plain text bundle the agent can read directly. Each result
 * carries ``repo / path:lines`` plus the code excerpt — the same shape
 * Sourcebot's web UI shows.
 */
const findCodeTool = createTool({
  name: "find_code",
  description:
    "Search the codebase for relevant files. Use this BEFORE making claims about how the code works. " +
    "Returns up to N snippets with file paths, line ranges, and code excerpts. " +
    "Query should be specific code-like terms (function names, symbols, distinctive strings).",
  inputSchema: {
    type: "object",
    properties: {
      query: {
        type: "string",
        description: "Search query — keywords, symbols, or short phrases.",
      },
      max_results: {
        type: "integer",
        description: "Cap on results returned. Default 10, max 30.",
      },
    },
    required: ["query"],
  },
  execute: async ({ query, max_results }) => {
    const url = (process.env.SOURCEBOT_URL || "").replace(/\/+$/, "");
    const key = process.env.SOURCEBOT_API_KEY || "";
    if (!url) return { text: "find_code unavailable: SOURCEBOT_URL not set on the bridge." };
    const cap = Math.min(Math.max(1, max_results ?? 10), 30);

    try {
      const r = await fetch(`${url}/api/search`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(key ? { Authorization: `Bearer ${key}` } : {}),
        },
        body: JSON.stringify({
          query,
          matches: cap,
          contextLines: 6,
          isRegexEnabled: false,
          isCaseSensitivityEnabled: false,
        }),
      });
      if (!r.ok) return { text: `find_code: sourcebot returned ${r.status}: ${(await r.text()).slice(0, 300)}` };
      const data = await r.json();
      const lines = [];
      let n = 0;
      for (const f of data.files || []) {
        for (const m of f.chunkMatches || f.chunks || f.matches || []) {
          n += 1;
          const start = m.rangeStart?.lineNumber ?? m.lineNumber ?? "?";
          const end = m.rangeEnd?.lineNumber ?? start;
          const content = (m.content || m.text || "").trimEnd();
          lines.push(`[${n}] ${f.repository || "?"} :: ${f.fileName}:${start}-${end}`);
          lines.push(content);
          lines.push("");
          if (n >= cap) break;
        }
        if (n >= cap) break;
      }
      if (n === 0) return { text: `find_code: no matches for ${JSON.stringify(query)}.` };
      return { text: lines.join("\n") };
    } catch (e) {
      return { text: `find_code error: ${e?.message || String(e)}` };
    }
  },
});

const fastify = Fastify({ logger: { level: process.env.LOG_LEVEL || "info" } });

fastify.get("/health", async () => ({ ok: true }));

fastify.post("/run", async (request, reply) => {
  const body = request.body ?? {};
  const {
    systemPrompt,
    prompt,
    cwd,
    providerId,
    modelId,
    apiKey,
    maxIterations = 30,
    timeoutSec = 600,
    // When true, the bridge registers the find_code tool so the agent can
    // call into Sourcebot. SOURCEBOT_URL on the bridge's env must be set.
    enableFindCode = false,
  } = body;

  if (!prompt || typeof prompt !== "string") {
    return reply.code(400).send({ ok: false, error: "prompt (string) is required" });
  }
  if (!providerId || !modelId || !apiKey) {
    return reply
      .code(400)
      .send({ ok: false, error: "providerId, modelId, apiKey are all required" });
  }

  // Event tail is for debugging only — the canonical answer lives on
  // ``result.outputText`` and usage on ``result.usage`` per the
  // ``AgentRunResult`` interface in ``@cline/shared``.
  const events = [];

  // Best-effort cwd injection — the SDK may not honor it directly, but
  // setting process.cwd() before agent construction is the safe bet for
  // tools that resolve paths from the current working directory.
  const originalCwd = process.cwd();
  if (cwd) {
    try { process.chdir(cwd); } catch { /* nonexistent path — let it fail loudly downstream */ }
  }

  let agent;
  try {
    const tools = enableFindCode ? [findCodeTool] : [];
    agent = new Agent({
      providerId,
      modelId,
      apiKey,
      maxIterations,
      systemPrompt,  // SDK accepts; undefined → SDK default
      tools,
    });

    if (typeof agent.subscribe === "function") {
      agent.subscribe((event) => {
        events.push(event);
        if (events.length > 200) events.shift();  // bound memory on chatty runs
      });
    }

    const result = await withTimeout(agent.run(prompt), timeoutSec * 1000);

    // ``AgentRunResult`` from @cline/shared:
    //   { agentId, runId, status, iterations, outputText, messages, usage, error? }
    const finalText = result?.outputText ?? "";
    const usage = result?.usage ?? {};

    return {
      ok: true,
      answer: finalText,
      status: result?.status,
      iterations: result?.iterations,
      tokensIn: usage.inputTokens,
      tokensOut: usage.outputTokens,
      cacheReadTokens: usage.cacheReadTokens,
      cacheWriteTokens: usage.cacheWriteTokens,
      costUsd: usage.totalCost,
      events: events.slice(-20),  // last 20 events for debugging
      error: result?.error ? String(result.error?.message ?? result.error) : undefined,
    };
  } catch (err) {
    return reply.code(502).send({
      ok: false,
      error: `${err?.name || "Error"}: ${err?.message || String(err)}`,
      events: events.slice(-20),
    });
  } finally {
    if (cwd) {
      try { process.chdir(originalCwd); } catch { /* ignore */ }
    }
  }
});

function withTimeout(promise, ms) {
  return new Promise((resolve, reject) => {
    const t = setTimeout(() => reject(new Error(`bridge timeout after ${ms}ms`)), ms);
    promise.then(
      (v) => { clearTimeout(t); resolve(v); },
      (e) => { clearTimeout(t); reject(e); },
    );
  });
}

const PORT = Number(process.env.PORT || 3040);
const HOST = process.env.HOST || "0.0.0.0";

try {
  await fastify.listen({ port: PORT, host: HOST });
  fastify.log.info(`cline-sdk-bridge listening on http://${HOST}:${PORT}`);
} catch (err) {
  fastify.log.error(err);
  process.exit(1);
}
