/**
 * jira-bridge HTTP server (Fastify).
 *
 * Two routes:
 *
 *   GET  /health
 *     Liveness; also confirms env config loaded.
 *
 *   POST /run
 *     Body: { "key": "TRT-123" }   (or `ticketKey`)
 *     If JIRA_BRIDGE_SHARED_SECRET is set, requires
 *       `Authorization: Bearer <secret>`.
 *     Returns the RunResult shape from decompose_ticket.ts.
 *
 * Designed as an internal service — a Jira automation rule or a Slack
 * /shortcut handler can POST to /run with a ticket key and get the
 * decomposition filed back as a comment.
 */

import Fastify from "fastify";
import { loadConfig, type Config } from "./config.js";
import { decomposeTicket } from "./decompose_ticket.js";

let config: Config;
try {
  config = loadConfig();
} catch (e) {
  process.stderr.write(`config error: ${(e as Error).message}\n`);
  process.exit(2);
}

const fastify = Fastify({
  logger: { level: process.env.LOG_LEVEL || "info" },
});

fastify.get("/health", async () => ({
  ok: true,
  service: "jira-bridge",
  techDecompUrl: config.techDecomp.url,
  techDecompAdapter: config.techDecomp.adapter,
}));

function authorizeOrReject(headers: Record<string, unknown>): string | null {
  if (!config.server.sharedSecret) return null;
  const v =
    (typeof headers.authorization === "string" && headers.authorization) || "";
  const expected = `Bearer ${config.server.sharedSecret}`;
  if (v !== expected) return "unauthorized";
  return null;
}

fastify.post("/run", async (request, reply) => {
  const reject = authorizeOrReject(request.headers as Record<string, unknown>);
  if (reject) {
    return reply.code(401).send({ ok: false, error: reject });
  }
  const body = (request.body ?? {}) as Record<string, unknown>;
  const key =
    ((body.key as string) || (body.ticketKey as string) || "").trim();
  if (!key) {
    return reply
      .code(400)
      .send({ ok: false, error: "body.key (ticket key) required" });
  }
  try {
    const result = await decomposeTicket(config, {
      ticketKey: key.toUpperCase(),
    });
    return { ok: true, ...result };
  } catch (e) {
    request.log.error({ err: e }, "decomposeTicket failed");
    return reply.code(502).send({ ok: false, error: (e as Error).message });
  }
});

try {
  await fastify.listen({ port: config.server.port, host: config.server.host });
  fastify.log.info(
    `jira-bridge listening on http://${config.server.host}:${config.server.port}`,
  );
} catch (err) {
  fastify.log.error(err);
  process.exit(1);
}
