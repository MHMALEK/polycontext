/**
 * OpenCode — @opencode-ai/sdk/v2 session API with structured JSON for decompose.
 * @see https://opencode.ai/docs/sdk/
 */
import fsp from "node:fs/promises";

import { createOpencode, createOpencodeClient } from "@opencode-ai/sdk/v2";
import type {
  AssistantMessage,
  OpencodeClient,
  Part,
  StructuredOutputError,
  TextPart,
} from "@opencode-ai/sdk/v2";
import { serenaEnv } from "./_serena.js";

export type OpencodeRunBody = {
  systemPrompt?: string;
  prompt: string;
  cwd?: string;
  /** IANA-style model id (`provider/model`, e.g. `anthropic/claude-sonnet-4-20250514`). */
  model?: string;
  /** Prefer with a stable server (`OPENCODE_BASE_URL`). See SDK "Client only" section. */
  baseUrl?: string;
  timeoutSec?: number;
  /** Enables JSON schema validated output (Structured Output docs). */
  structured?: boolean;
  structuredRetryCount?: number;
  /** With apiKey — registers provider credentials via auth.set. */
  providerID?: string;
  apiKey?: string;
  /** OpenCode agent to run under. Defaults to "build" so the read-side tools
   * (read, grep, glob, ls) are wired in. Without this OpenCode's prompt API
   * runs a no-tool path and the model can't iterate beyond the prefetched
   * grounding block. */
  agent?: string;
};

const DECOMPOSITION_JSON_SCHEMA = {
  type: "object",
  description:
    "Tech decomposition for an engineering ticket: overview, repos, risks, questions, subtasks.",
  properties: {
    query: {
      type: "string",
      description: "The original ticket or task text echoed back verbatim when possible.",
    },
    overview: {
      type: "string",
      description: "2–4 concise sentences framing what engineers need to do (no markdown fences).",
    },
    affected_repos: {
      type: "array",
      items: { type: "string", description: "Repo key/name from workspace (must exist on disk)." },
      description: "Repositories touched or investigated for this work.",
    },
    risks: {
      type: "array",
      items: { type: "string" },
      description: "Technical or product risks to watch.",
    },
    open_questions: {
      type: "array",
      items: { type: "string" },
      description: "Unanswered gaps an engineer must clarify.",
    },
    subtasks: {
      type: "array",
      description: "Pickup-sized units of work; paths grounded when possible.",
      items: {
        type: "object",
        properties: {
          title: { type: "string", description: "Short actionable title." },
          description: { type: "string", description: "What to implement or verify." },
          repo: {
            type: "string",
            description: "Repo identifier this task belongs to.",
          },
          files: {
            type: "array",
            items: { type: "string" },
            description: "Repo-relative paths (omit if unknown).",
          },
          file_links: { type: "array", items: { type: "string" }, description: "Permalinks if known." },
          acceptance_criteria: {
            type: "array",
            items: { type: "string" },
            description: "Concrete done-when bullets.",
          },
          estimated_complexity: {
            type: "string",
            enum: ["small", "medium", "large", "unknown"],
            description: "Relative size.",
          },
        },
        required: ["title", "description", "repo", "estimated_complexity"],
      },
    },
    enrichment_model: {
      type: "string",
      description: "Optional; empty string if unset.",
    },
    decomposition_model: {
      type: "string",
      description: "Adapter or model tag for this decomposition.",
    },
  },
  required: ["query", "overview", "affected_repos", "subtasks"],
};

function parseModel(spec: string | undefined): { providerID: string; modelID: string } | undefined {
  const s = (spec || "").trim();
  const i = s.indexOf("/");
  if (i <= 0 || i === s.length - 1) return undefined;
  return { providerID: s.slice(0, i).trim(), modelID: s.slice(i + 1).trim() };
}

function isStructuredOutputError(e: unknown): e is StructuredOutputError {
  return !!e && typeof e === "object" && "name" in e && (e as { name?: string }).name === "StructuredOutputError";
}

function textFromParts(parts: Part[] | undefined): string {
  if (!parts?.length) return "";
  const out: string[] = [];
  for (const p of parts) if (p.type === "text") out.push((p as TextPart).text);
  return out.join("").trim();
}

function withTimeout<T>(promise: Promise<T>, ms: number): Promise<T> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`timeout after ${ms}ms`)), ms);
    promise.then(
      (v) => {
        clearTimeout(timer);
        resolve(v);
      },
      (e) => {
        clearTimeout(timer);
        reject(e);
      }
    );
  });
}

async function resolveDirectory(raw: string | undefined): Promise<string | undefined> {
  if (!raw?.trim()) return undefined;
  const dir = raw.trim();
  try {
    const st = await fsp.stat(dir);
    return st.isDirectory() ? dir : undefined;
  } catch {
    return undefined;
  }
}

export async function runOpencode(body: OpencodeRunBody): Promise<{
  ok: boolean;
  answer?: string;
  error?: string;
  structuredOutputFailed?: boolean;
  tokensIn?: number;
  tokensOut?: number;
  costUsd?: number;
  model?: string;
  /** Number of tool invocations across the session — non-zero confirms the
   * agentic loop fired. Counted from the FULL message log (session.messages),
   * not just the final response message which only contains the summary. */
  toolCalls?: number;
  /** Distinct tool names invoked (e.g. ``["read","grep"]``). */
  toolNames?: string[];
  /** Repo-relative paths the agent actually read via the ``read`` tool.
   * Surfaced so the context_recall scorer can compute recall for the
   * opencode adapter without changes. */
  groundingPaths?: string[];
}> {
  const timeoutMs = Math.max(1, (body.timeoutSec ?? 600) * 1000);
  let serverClose: (() => void) | undefined;

  try {
    const cwd = await resolveDirectory(body.cwd);
    const envBase = (process.env.OPENCODE_BASE_URL || "").trim();
    const bodyBase = (body.baseUrl || "").trim();
    const baseUrl = bodyBase || envBase;

    const modelSpec =
      (body.model || "").trim() ||
      (process.env.OPENCODE_MODEL || "").trim() ||
      "anthropic/claude-sonnet-4-20250514";
    const resolvedModel = parseModel(modelSpec);

    let client: OpencodeClient;
    if (baseUrl) {
      client = createOpencodeClient({
        baseUrl,
        ...(cwd ? { directory: cwd } : {}),
      });
    } else {
      // Serena MCP is registered through OpenCode's embedded-server config when
      // SERENA_URL is set. OpenCode only has a single `type: "remote"` MCP shape
      // — auto-detection handles SSE vs streamable HTTP from the URL.
      const sEnv = serenaEnv();
      const mcp = sEnv
        ? {
            serena: {
              type: "remote" as const,
              url: sEnv.url,
              enabled: true,
              ...(sEnv.apiKey
                ? { headers: { Authorization: `Bearer ${sEnv.apiKey}` } }
                : {}),
            },
          }
        : undefined;
      const oc = await createOpencode({
        timeout: Number(process.env.OPENCODE_SERVER_START_TIMEOUT_MS || 30_000),
        hostname: (process.env.OPENCODE_HOSTNAME || "127.0.0.1").trim() || "127.0.0.1",
        port: Number(process.env.OPENCODE_PORT || 4096) || 4096,
        config: { model: modelSpec, ...(mcp ? { mcp } : {}) },
      });
      serverClose = () => oc.server.close();
      client = oc.client;
    }

    if (body.providerID && body.apiKey) {
      const authSet = await client.auth.set({
        providerID: body.providerID,
        auth: { type: "api", key: body.apiKey },
      });
      // The v2 SDK returns `{ data, error, request, response }`. On a
      // connect-refused (e.g. baseUrl pointing at 127.0.0.1 from inside Docker
      // when opencode is on the host) the SDK swallows the network error into
      // an empty `{}` envelope — including the response status helps diagnose
      // that case vs a real auth/protocol failure.
      if (authSet.error) {
        return {
          ok: false,
          error: `auth.set failed: ${JSON.stringify(authSet.error).slice(0, 420)} (response status ${authSet.response?.status})`,
        };
      }
    }

    // Pass agent + model at session.create per the CLI's pattern
    // (cmd/run.ts:370-388 in sst/opencode@dev). The agent registered on the
    // session governs which tools the loop has access to; without it the
    // model defaults to whichever agent the server resolves and may not be
    // able to use read/grep/glob.
    const created = await client.session.create({
      directory: cwd,
      title: "tech-decomposition",
      agent: (body.agent || "build").trim(),
      model:
        resolvedModel != null &&
        resolvedModel.providerID.trim() !== "" &&
        resolvedModel.modelID.trim() !== ""
          ? {
              providerID: resolvedModel.providerID,
              id: resolvedModel.modelID,
            }
          : undefined,
    });

    if (created.error || !created.data?.id) {
      return {
        ok: false,
        error: created.error
          ? JSON.stringify(created.error).slice(0, 500)
          : "session.create did not return a session id",
      };
    }

    const sessionID = created.data.id;
    try {
      const format = body.structured
        ? {
            type: "json_schema" as const,
            schema: DECOMPOSITION_JSON_SCHEMA,
            retryCount:
              typeof body.structuredRetryCount === "number" &&
              Number.isFinite(body.structuredRetryCount)
                ? body.structuredRetryCount
                : 2,
          }
        : undefined;

      // ``agent`` selects which OpenCode agent runs the loop. ``build`` is the
      // default full-tool agent (read/grep/glob/ls/etc all allowed via its
      // permission ruleset). We pass it explicitly so this code path is the
      // same as ``opencode run --agent build``.
      //
      // We deliberately do NOT pass ``tools: {...}`` here — per the SDK type
      // ``Config.tools`` (gen/types.gen.d.ts:927), this map is a DISABLE
      // filter applied on top of agent permissions, not an enable list. All
      // tools default to true; passing them as true is a no-op. We mask
      // writes via the permission ruleset on session.create instead.
      const agentName = (body.agent || "build").trim();
      const promptRes = await withTimeout(
        client.session.prompt({
          sessionID,
          directory: cwd,
          system: (body.systemPrompt || "").trim() || undefined,
          model:
            resolvedModel != null
              ? {
                  providerID: resolvedModel.providerID,
                  modelID: resolvedModel.modelID,
                }
              : undefined,
          agent: agentName,
          ...(format ? { format } : {}),
          parts: [{ type: "text", text: body.prompt }],
        }),
        timeoutMs
      );

      if (promptRes.error) {
        return {
          ok: false,
          error: JSON.stringify(promptRes.error).slice(0, 500),
        };
      }

      const data = promptRes.data;
      if (!data?.info) {
        return { ok: false, error: "prompt returned no assistant message envelope" };
      }

      const assistant = data.info as AssistantMessage;

      const errUnknown = assistant.error as unknown | undefined;
      if (errUnknown !== undefined && isStructuredOutputError(errUnknown)) {
        const soErr = assistant.error as StructuredOutputError;
        return {
          ok: false,
          structuredOutputFailed: true,
          error: `${soErr.name}: ${soErr.data.message} (retries: ${String(soErr.data.retries)})`,
        };
      }

      if (assistant.error) {
        const e = assistant.error;
        const nm = "name" in e ? String(e.name) : "Error";
        const msg =
          e &&
          typeof e === "object" &&
          "data" in e &&
          typeof (e as { data?: { message?: unknown } }).data?.message === "string"
            ? ((e as { data: { message: string } }).data.message ?? "")
            : JSON.stringify(e).slice(0, 300);
        return { ok: false, error: `${nm}: ${msg}` };
      }

      let answer: string | undefined;

      if (body.structured && assistant.structured != null) {
        answer =
          typeof assistant.structured === "string"
            ? assistant.structured
            : JSON.stringify(assistant.structured);
      }

      if (answer === undefined || !answer.trim()) {
        answer = textFromParts(data.parts) || "";
      }

      const tokens = assistant.tokens;
      // The SDK's session.prompt returns ONLY the final assistant message in
      // ``data`` — the agentic loop's intermediate messages (each carrying a
      // ``type:"tool"`` part) live in the session's message log. Fetch all
      // messages to get the full tool trace AND the actual file paths the
      // agent read (used by the recall scorer downstream).
      const allMessages = await client.session.messages({
        sessionID,
        directory: cwd,
      });
      const messageList = Array.isArray(allMessages?.data) ? allMessages.data : [];
      const allParts: unknown[] = [];
      for (const msg of messageList) {
        const ps = Array.isArray((msg as { parts?: unknown[] }).parts)
          ? (msg as { parts: unknown[] }).parts
          : [];
        for (const p of ps) allParts.push(p);
      }
      const toolCallParts = allParts.filter(
        (p) => p && typeof p === "object" && (p as { type?: string }).type === "tool",
      );
      const toolNames = Array.from(
        new Set(
          toolCallParts
            .map((p) => (p as { tool?: string }).tool)
            .filter((n): n is string => typeof n === "string" && n.length > 0),
        ),
      );
      // Extract repo-relative paths from ``read`` tool calls so the recall
      // scorer can compare them against each case's ``expected_files``.
      const readPaths = new Set<string>();
      const cwdAbs = cwd ?? "";
      for (const p of toolCallParts) {
        const tp = p as {
          tool?: string;
          state?: { input?: { filePath?: string } };
        };
        if (tp.tool !== "read") continue;
        const fp = tp.state?.input?.filePath;
        if (typeof fp !== "string" || !fp) continue;
        const rel =
          cwdAbs && fp.startsWith(cwdAbs + "/")
            ? fp.slice(cwdAbs.length + 1)
            : fp;
        readPaths.add(rel.replace(/\\/g, "/"));
      }

      return {
        ok: true,
        answer,
        tokensIn: typeof tokens?.input === "number" ? tokens.input : undefined,
        tokensOut: typeof tokens?.output === "number" ? tokens.output : undefined,
        costUsd: typeof assistant.cost === "number" ? assistant.cost : undefined,
        model: modelSpec,
        toolCalls: toolCallParts.length,
        toolNames,
        groundingPaths: Array.from(readPaths).sort(),
      };
    } finally {
      await client.session.delete({ sessionID, directory: cwd }).catch(() => undefined);
    }
  } catch (err) {
    const e = err as Error;
    return {
      ok: false,
      error: `${e?.name || "Error"}: ${e?.message || String(err)}`,
    };
  } finally {
    if (serverClose) serverClose();
  }
}
