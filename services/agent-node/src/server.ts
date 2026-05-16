/**
 * Agent runtime service — Cursor SDK, Cline SDK, Claude Agent SDK (Claude Code),
 * Gemini (@google/genai), OpenAI Agents SDK (@openai/agents), Sourcebot.
 * Python FastAPI adapters call this over HTTP.
 */
import Fastify from "fastify";
import { runCursor } from "./adapters/cursor.js";
import { runCline } from "./adapters/cline.js";
import { runClaudeCode } from "./adapters/claude_code.js";
import { runGemini } from "./adapters/gemini.js";
import { runOpenAIAgents } from "./adapters/openai_agents.js";
import { askSourcebotBlocking } from "./adapters/sourcebot.js";

const fastify = Fastify({ logger: { level: process.env.LOG_LEVEL || "info" } });

fastify.get("/health", async () => ({
  ok: true,
  service: "tech-decomposition-agent-node",
}));

fastify.get("/adapters", async () => ({
  adapters: [
    { name: "cursor", capabilities: ["ask", "decompose", "implement"] },
    { name: "cline_sdk", capabilities: ["ask", "decompose", "implement"] },
    { name: "claude_code", capabilities: ["ask", "decompose", "implement"] },
    { name: "gemini", capabilities: ["ask", "decompose", "implement"] },
    { name: "openai_agents", capabilities: ["ask", "decompose", "implement"] },
    { name: "sourcebot", capabilities: ["ask"] },
  ],
}));

fastify.post("/adapters/cursor/run", async (request, reply) => {
  const b = (request.body ?? {}) as Record<string, unknown>;
  const apiKey = (b.apiKey as string) || process.env.CURSOR_API_KEY || "";
  if (!b.prompt || typeof b.prompt !== "string") {
    return reply.code(400).send({ ok: false, error: "prompt (string) required" });
  }
  if (!b.cwd || typeof b.cwd !== "string") {
    return reply.code(400).send({ ok: false, error: "cwd (string) required" });
  }
  if (!apiKey) {
    return reply.code(400).send({ ok: false, error: "apiKey or CURSOR_API_KEY required" });
  }
  const out = await runCursor({
    systemPrompt: (b.systemPrompt as string) || "",
    prompt: b.prompt as string,
    cwd: b.cwd as string,
    apiKey,
    modelId: (b.modelId as string) || process.env.CURSOR_SDK_MODEL || "composer-2",
    timeoutSec: (b.timeoutSec as number) ?? 600,
  });
  if (!out.ok) {
    return reply.code(502).send(out);
  }
  return out;
});

fastify.post("/adapters/claude_code/run", async (request, reply) => {
  const b = (request.body ?? {}) as Record<string, unknown>;
  const apiKey =
    (b.apiKey as string) || process.env.ANTHROPIC_API_KEY || "";
  if (!b.prompt || typeof b.prompt !== "string") {
    return reply.code(400).send({ ok: false, error: "prompt (string) required" });
  }
  if (!b.cwd || typeof b.cwd !== "string") {
    return reply.code(400).send({ ok: false, error: "cwd (string) required" });
  }
  if (!apiKey) {
    return reply
      .code(400)
      .send({ ok: false, error: "apiKey or ANTHROPIC_API_KEY required" });
  }
  const out = await runClaudeCode({
    systemPrompt: b.systemPrompt as string | undefined,
    prompt: b.prompt as string,
    cwd: b.cwd as string,
    apiKey,
    model:
      (b.model as string) ||
      process.env.CLAUDE_CODE_MODEL ||
      "claude-sonnet-4-5",
    maxTurns: b.maxTurns as number | undefined,
    timeoutSec: (b.timeoutSec as number) ?? 600,
  });
  if (!out.ok) {
    return reply.code(502).send(out);
  }
  return out;
});

fastify.post("/adapters/cline/run", async (request, reply) => {
  const b = (request.body ?? {}) as Record<string, unknown>;
  if (!b.prompt || typeof b.prompt !== "string") {
    return reply.code(400).send({ ok: false, error: "prompt (string) required" });
  }
  if (!b.providerId || !b.modelId || !b.apiKey) {
    return reply
      .code(400)
      .send({ ok: false, error: "providerId, modelId, apiKey required" });
  }
  const out = await runCline({
    systemPrompt: b.systemPrompt as string | undefined,
    prompt: b.prompt as string,
    cwd: b.cwd as string | undefined,
    providerId: b.providerId as string,
    modelId: b.modelId as string,
    apiKey: b.apiKey as string,
    maxIterations: (b.maxIterations as number) ?? 30,
    timeoutSec: (b.timeoutSec as number) ?? 600,
    enableFindCode: Boolean(b.enableFindCode),
  });
  if (!out.ok) {
    return reply.code(502).send(out);
  }
  return out;
});

fastify.post("/adapters/gemini/run", async (request, reply) => {
  const b = (request.body ?? {}) as Record<string, unknown>;
  if (!b.prompt || typeof b.prompt !== "string") {
    return reply.code(400).send({ ok: false, error: "prompt (string) required" });
  }
  const apiKey =
    (b.apiKey as string) || process.env.GEMINI_API_KEY || "";
  if (!apiKey) {
    return reply
      .code(400)
      .send({ ok: false, error: "apiKey or GEMINI_API_KEY required" });
  }
  const out = await runGemini({
    systemPrompt: (b.systemPrompt as string) || undefined,
    prompt: b.prompt as string,
    apiKey,
    modelId: (b.modelId as string) || process.env.GEMINI_SDK_MODEL,
    timeoutSec: (b.timeoutSec as number) ?? 600,
    cwd: (b.cwd as string) || undefined,
    maxToolRounds: (b.maxToolRounds as number) ?? undefined,
  });
  if (!out.ok) {
    return reply.code(502).send(out);
  }
  return out;
});

fastify.post("/adapters/openai_agents/run", async (request, reply) => {
  const b = (request.body ?? {}) as Record<string, unknown>;
  if (!b.prompt || typeof b.prompt !== "string") {
    return reply.code(400).send({ ok: false, error: "prompt (string) required" });
  }
  const apiKey = (b.apiKey as string) || process.env.OPENAI_API_KEY || "";
  if (!apiKey) {
    return reply
      .code(400)
      .send({ ok: false, error: "apiKey or OPENAI_API_KEY required" });
  }
  const out = await runOpenAIAgents({
    systemPrompt: (b.systemPrompt as string) || undefined,
    prompt: b.prompt as string,
    apiKey,
    modelId: (b.modelId as string) || process.env.OPENAI_AGENTS_SDK_MODEL,
    timeoutSec: (b.timeoutSec as number) ?? 600,
    maxTurns: (b.maxTurns as number) ?? undefined,
    cwd: (b.cwd as string) || undefined,
  });
  if (!out.ok) {
    return reply.code(502).send(out);
  }
  return out;
});

fastify.post("/adapters/sourcebot/ask", async (request, reply) => {
  const b = (request.body ?? {}) as Record<string, unknown>;
  const url = (b.sourcebotUrl as string) || "";
  const key = (b.sourcebotApiKey as string) || "";
  const question = (b.question as string) || "";
  if (!url || !key) {
    return reply
      .code(400)
      .send({ ok: false, error: "sourcebotUrl and sourcebotApiKey required" });
  }
  if (!question.trim()) {
    return reply.code(400).send({ ok: false, error: "question required" });
  }
  const out = await askSourcebotBlocking({
    sourcebotUrl: url,
    sourcebotApiKey: key,
    question,
    repos: b.repos as string[] | undefined,
    maxSteps: b.maxSteps as number | undefined,
    timeoutSec: (b.timeoutSec as number) ?? 300,
  });
  if (!out.ok) {
    return reply.code(502).send(out);
  }
  return out;
});

const PORT = Number(process.env.PORT || 13100);
const HOST = process.env.HOST || "0.0.0.0";

try {
  await fastify.listen({ port: PORT, host: HOST });
  fastify.log.info(`agent-node listening on http://${HOST}:${PORT}`);
} catch (err) {
  fastify.log.error(err);
  process.exit(1);
}
