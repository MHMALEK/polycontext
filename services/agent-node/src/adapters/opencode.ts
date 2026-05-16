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
      const oc = await createOpencode({
        timeout: Number(process.env.OPENCODE_SERVER_START_TIMEOUT_MS || 30_000),
        hostname: (process.env.OPENCODE_HOSTNAME || "127.0.0.1").trim() || "127.0.0.1",
        port: Number(process.env.OPENCODE_PORT || 4096) || 4096,
        config: { model: modelSpec },
      });
      serverClose = () => oc.server.close();
      client = oc.client;
    }

    if (body.providerID && body.apiKey) {
      const authSet = await client.auth.set({
        providerID: body.providerID,
        auth: { type: "api", key: body.apiKey },
      });
      if (authSet.error) {
        return {
          ok: false,
          error: `auth.set failed: ${JSON.stringify(authSet.error).slice(0, 420)}`,
        };
      }
    }

    const created = await client.session.create({
      directory: cwd,
      title: "tech-decomposition",
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

      return {
        ok: true,
        answer,
        tokensIn: typeof tokens?.input === "number" ? tokens.input : undefined,
        tokensOut: typeof tokens?.output === "number" ? tokens.output : undefined,
        costUsd: typeof assistant.cost === "number" ? assistant.cost : undefined,
        model: modelSpec,
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
