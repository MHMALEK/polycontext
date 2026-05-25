/**
 * OpenCode — @opencode-ai/sdk/v2 session API with structured JSON for decompose.
 *
 * Exposes three entry points:
 *   - runOpencode(...)        — blocking single-shot; returns metrics + answer.
 *   - streamOpencode(...)     — async-generator over normalized stream events
 *                               (text deltas, tool start/end, session.idle, etc.).
 *   - abortOpencodeSession()  — cancels a running session by id.
 *
 * Streaming wires {@link OpencodeClient.event.subscribe} into a UI-shaped event
 * stream so the Python API can re-emit SSE without each consumer re-implementing
 * the OpenCode event taxonomy.
 *
 * @see https://opencode.ai/docs/sdk/
 */
import fsp from "node:fs/promises";

import { createOpencode, createOpencodeClient } from "@opencode-ai/sdk/v2";
import type {
  AssistantMessage,
  Event,
  OpencodeClient,
  Part,
  StructuredOutputError,
  TextPart,
  ToolPart,
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
  /** When false, mask out the read-side tools in session.prompt so the model
   * answers single-shot from whatever context the prompt already contains.
   * Maps to UI "Grounded only (no agent tools)" mode. Defaults to true. */
  toolsEnabled?: boolean;
  /** Reuse an existing session id (chat continuity, prompt caching). When
   * omitted, a fresh session is created and torn down at the end of the call. */
  sessionID?: string;
  /** When true (default false), do NOT delete the session at the end so a
   * follow-up call can reuse it via {@link sessionID}. */
  keepSession?: boolean;
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

/** Tool-state breakdown surfaced in metrics.extra.tool_trace for the UI's
 * Telemetry panel — one row per call with name, status, duration, and a
 * 200-char preview of the title (e.g. "read src/foo.py"). */
type ToolTraceEntry = {
  name: string;
  status: "pending" | "running" | "completed" | "error";
  duration_ms?: number;
  title?: string;
  file_path?: string;
};

function toolTraceFromParts(parts: unknown[]): ToolTraceEntry[] {
  const out: ToolTraceEntry[] = [];
  for (const p of parts) {
    if (!p || typeof p !== "object") continue;
    const tp = p as ToolPart;
    if (tp.type !== "tool") continue;
    const state = tp.state;
    let duration: number | undefined;
    let title: string | undefined;
    let filePath: string | undefined;
    const input = (state && "input" in state ? state.input : {}) as Record<string, unknown> | undefined;
    if (input && typeof input.filePath === "string") filePath = input.filePath;
    if (state && (state.status === "completed" || state.status === "error")) {
      const s = state as { time: { start: number; end: number }; title?: string };
      duration = Math.max(0, Math.round(s.time.end - s.time.start));
      title = s.title?.slice(0, 200);
    } else if (state && state.status === "running") {
      title = state.title?.slice(0, 200);
    }
    out.push({
      name: tp.tool || "?",
      status: state?.status || "pending",
      ...(typeof duration === "number" ? { duration_ms: duration } : {}),
      ...(title ? { title } : {}),
      ...(filePath ? { file_path: filePath } : {}),
    });
  }
  return out;
}

/** Shape the agent-node returns to the Python adapter — kept stable across
 * blocking and streaming entry points so the Python side has one parser. */
export type OpencodeRunResult = {
  ok: boolean;
  answer?: string;
  error?: string;
  structuredOutputFailed?: boolean;
  tokensIn?: number;
  tokensOut?: number;
  tokensReasoning?: number;
  cacheReadTokens?: number;
  cacheWriteTokens?: number;
  costUsd?: number;
  model?: string;
  /** Provider OpenCode resolved (e.g. ``anthropic``, ``openrouter``). */
  providerID?: string;
  /** OpenCode agent that ran (e.g. ``build``). */
  agentMode?: string;
  /** "stop", "length", "error", or whatever the underlying provider reports. */
  finishReason?: string;
  /** Session id used — surfaced so the caller can reuse it on the next turn. */
  sessionID?: string;
  /** Total tool invocations across the session — non-zero confirms the
   * agentic loop fired. Counted from the FULL message log. */
  toolCalls?: number;
  /** Distinct tool names invoked (e.g. ``["read","grep"]``). */
  toolNames?: string[];
  /** Per-call breakdown (name, status, duration, title) for the Telemetry UI. */
  toolTrace?: ToolTraceEntry[];
  /** Sum of completed-tool durations — the agent's "wall in tools" budget. */
  toolWallMs?: number;
  /** Repo-relative paths the agent actually read via the ``read`` tool. */
  groundingPaths?: string[];
};

type ResolveClientResult = {
  client: OpencodeClient;
  serverClose?: () => void;
  cwd: string | undefined;
  modelSpec: string;
  resolvedModel: { providerID: string; modelID: string } | undefined;
};

/** Build (or connect to) an OpenCode client + register credentials. Shared by
 * runOpencode and streamOpencode — both need identical setup. */
async function resolveClient(body: OpencodeRunBody): Promise<ResolveClientResult> {
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
  let serverClose: (() => void) | undefined;
  if (baseUrl) {
    client = createOpencodeClient({
      baseUrl,
      ...(cwd ? { directory: cwd } : {}),
    });
  } else {
    const sEnv = serenaEnv();
    const mcp = sEnv
      ? {
          serena: {
            type: "remote" as const,
            url: sEnv.url,
            enabled: true,
            ...(sEnv.apiKey ? { headers: { Authorization: `Bearer ${sEnv.apiKey}` } } : {}),
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
    if (authSet.error) {
      const status = authSet.response?.status;
      throw new Error(
        `auth.set failed: ${JSON.stringify(authSet.error).slice(0, 420)} (response status ${status})`
      );
    }
  }

  return { client, serverClose, cwd, modelSpec, resolvedModel };
}

/** Build the per-call tools mask: ``undefined`` means "all tools on", a record
 * means each tool is explicitly disabled in session.prompt. */
function toolsMask(toolsEnabled: boolean | undefined): Record<string, boolean> | undefined {
  if (toolsEnabled !== false) return undefined;
  return {
    read: false,
    grep: false,
    glob: false,
    ls: false,
    bash: false,
    edit: false,
    write: false,
    multiedit: false,
    patch: false,
    webfetch: false,
  };
}

/** Aggregate token+cost telemetry from the assistant message + full message log. */
function aggregateMetrics(
  assistant: AssistantMessage,
  parts: unknown[]
): {
  tokensIn?: number;
  tokensOut?: number;
  tokensReasoning?: number;
  cacheReadTokens?: number;
  cacheWriteTokens?: number;
  costUsd?: number;
  finishReason?: string;
  providerID?: string;
  agentMode?: string;
  toolTrace: ToolTraceEntry[];
  toolWallMs: number;
} {
  const tokens = assistant.tokens;
  const trace = toolTraceFromParts(parts);
  const toolWallMs = trace.reduce((acc, t) => acc + (t.duration_ms || 0), 0);
  return {
    tokensIn: typeof tokens?.input === "number" ? tokens.input : undefined,
    tokensOut: typeof tokens?.output === "number" ? tokens.output : undefined,
    tokensReasoning: typeof tokens?.reasoning === "number" ? tokens.reasoning : undefined,
    cacheReadTokens: typeof tokens?.cache?.read === "number" ? tokens.cache.read : undefined,
    cacheWriteTokens: typeof tokens?.cache?.write === "number" ? tokens.cache.write : undefined,
    costUsd: typeof assistant.cost === "number" ? assistant.cost : undefined,
    finishReason: typeof assistant.finish === "string" ? assistant.finish : undefined,
    providerID: assistant.providerID,
    agentMode: assistant.mode,
    toolTrace: trace,
    toolWallMs,
  };
}

/** Read every part across every message in the session — used for the
 * post-call tool trace + grounding-paths extraction (the prompt response only
 * carries the FINAL assistant message). */
async function readAllParts(
  client: OpencodeClient,
  sessionID: string,
  cwd: string | undefined
): Promise<unknown[]> {
  const allMessages = await client.session.messages({ sessionID, directory: cwd });
  const messageList = Array.isArray(allMessages?.data) ? allMessages.data : [];
  const parts: unknown[] = [];
  for (const msg of messageList) {
    const ps = Array.isArray((msg as { parts?: unknown[] }).parts)
      ? (msg as { parts: unknown[] }).parts
      : [];
    for (const p of ps) parts.push(p);
  }
  return parts;
}

function groundingPathsFromParts(parts: unknown[], cwd: string | undefined): string[] {
  const out = new Set<string>();
  const cwdAbs = cwd ?? "";
  for (const p of parts) {
    if (!p || typeof p !== "object") continue;
    const tp = p as { type?: string; tool?: string; state?: { input?: { filePath?: string } } };
    if (tp.type !== "tool" || tp.tool !== "read") continue;
    const fp = tp.state?.input?.filePath;
    if (typeof fp !== "string" || !fp) continue;
    const rel = cwdAbs && fp.startsWith(cwdAbs + "/") ? fp.slice(cwdAbs.length + 1) : fp;
    out.add(rel.replace(/\\/g, "/"));
  }
  return Array.from(out).sort();
}

export async function runOpencode(body: OpencodeRunBody): Promise<OpencodeRunResult> {
  const timeoutMs = Math.max(1, (body.timeoutSec ?? 600) * 1000);
  let serverClose: (() => void) | undefined;
  let createdSessionID: string | undefined;
  let sessionID = (body.sessionID || "").trim() || undefined;
  let client: OpencodeClient | undefined;
  let cwd: string | undefined;

  try {
    const ctx = await resolveClient(body);
    serverClose = ctx.serverClose;
    cwd = ctx.cwd;
    client = ctx.client;
    const { modelSpec, resolvedModel } = ctx;

    if (!sessionID) {
      // Pass agent + model at session.create per the CLI's pattern
      // (cmd/run.ts:370-388 in sst/opencode@dev). The agent governs which
      // tools the loop has access to; without it the model defaults to
      // whichever agent the server resolves.
      const created = await client.session.create({
        directory: cwd,
        title: "tech-decomposition",
        agent: (body.agent || "build").trim(),
        model:
          resolvedModel != null &&
          resolvedModel.providerID.trim() !== "" &&
          resolvedModel.modelID.trim() !== ""
            ? { providerID: resolvedModel.providerID, id: resolvedModel.modelID }
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
      sessionID = created.data.id;
      createdSessionID = sessionID;
    }

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

    const agentName = (body.agent || "build").trim();
    const tools = toolsMask(body.toolsEnabled);
    const promptRes = await withTimeout(
      client.session.prompt({
        sessionID,
        directory: cwd,
        system: (body.systemPrompt || "").trim() || undefined,
        model:
          resolvedModel != null
            ? { providerID: resolvedModel.providerID, modelID: resolvedModel.modelID }
            : undefined,
        agent: agentName,
        ...(tools ? { tools } : {}),
        ...(format ? { format } : {}),
        parts: [{ type: "text", text: body.prompt }],
      }),
      timeoutMs
    );

    if (promptRes.error) {
      return { ok: false, sessionID, error: JSON.stringify(promptRes.error).slice(0, 500) };
    }

    const data = promptRes.data;
    if (!data?.info) {
      return { ok: false, sessionID, error: "prompt returned no assistant message envelope" };
    }
    const assistant = data.info as AssistantMessage;

    const errUnknown = assistant.error as unknown | undefined;
    if (errUnknown !== undefined && isStructuredOutputError(errUnknown)) {
      const soErr = assistant.error as StructuredOutputError;
      return {
        ok: false,
        sessionID,
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
      return { ok: false, sessionID, error: `${nm}: ${msg}` };
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

    const allParts = await readAllParts(client, sessionID, cwd);
    const agg = aggregateMetrics(assistant, allParts);
    const toolNames = Array.from(
      new Set(
        agg.toolTrace
          .map((t) => t.name)
          .filter((n): n is string => typeof n === "string" && n.length > 0)
      )
    );

    return {
      ok: true,
      answer,
      sessionID,
      tokensIn: agg.tokensIn,
      tokensOut: agg.tokensOut,
      tokensReasoning: agg.tokensReasoning,
      cacheReadTokens: agg.cacheReadTokens,
      cacheWriteTokens: agg.cacheWriteTokens,
      costUsd: agg.costUsd,
      finishReason: agg.finishReason,
      providerID: agg.providerID,
      agentMode: agg.agentMode,
      model: modelSpec,
      toolCalls: agg.toolTrace.length,
      toolNames,
      toolTrace: agg.toolTrace,
      toolWallMs: agg.toolWallMs,
      groundingPaths: groundingPathsFromParts(allParts, cwd),
    };
  } catch (err) {
    const e = err as Error;
    return { ok: false, sessionID, error: `${e?.name || "Error"}: ${e?.message || String(err)}` };
  } finally {
    // Only delete sessions we created AND the caller didn't ask to keep. A
    // reused (caller-supplied) sessionID is the caller's responsibility.
    if (createdSessionID && !body.keepSession) {
      await client?.session
        .delete({ sessionID: createdSessionID, directory: cwd })
        .catch(() => undefined);
    }
    if (serverClose) serverClose();
  }
}

// ---------------------------------------------------------------------------
// Streaming
// ---------------------------------------------------------------------------

/** Normalized event shape forwarded to the Python API. Stable across SDK
 * versions so the UI doesn't see raw OpenCode internals. */
export type OpencodeStreamEvent =
  | { kind: "session"; sessionID: string }
  | { kind: "status"; status: string; attempt?: number; message?: string }
  | { kind: "text.delta"; text: string }
  | { kind: "reasoning.delta"; text: string }
  | {
      kind: "tool.update";
      callID: string;
      tool: string;
      status: "pending" | "running" | "completed" | "error";
      title?: string;
      filePath?: string;
      durationMs?: number;
      error?: string;
    }
  | { kind: "todo"; todos: Array<{ content: string; status: string; priority: string }> }
  | { kind: "file.edited"; path: string }
  | { kind: "error"; error: string }
  | { kind: "done"; result: OpencodeRunResult };

type PromptOutcome =
  | { kind: "pending" }
  | { kind: "ok"; assistant: AssistantMessage; parts: Part[] }
  | { kind: "err"; error: string }
  | { kind: "structured-failed"; error: string };

/** Drive a prompt and stream normalized events for the UI. The SSE bridge in
 * server.ts iterates this generator and writes each event as `data: <json>`. */
export async function* streamOpencode(
  body: OpencodeRunBody,
  signal?: AbortSignal
): AsyncGenerator<OpencodeStreamEvent, void, void> {
  let serverClose: (() => void) | undefined;
  let createdSessionID: string | undefined;
  let sessionID = (body.sessionID || "").trim() || undefined;
  let client: OpencodeClient | undefined;
  let cwd: string | undefined;
  let stop: (() => void) | undefined;

  try {
    const ctx = await resolveClient(body);
    serverClose = ctx.serverClose;
    cwd = ctx.cwd;
    client = ctx.client;
    const { modelSpec, resolvedModel } = ctx;

    if (!sessionID) {
      const created = await client.session.create({
        directory: cwd,
        title: "tech-decomposition",
        agent: (body.agent || "build").trim(),
        model:
          resolvedModel != null &&
          resolvedModel.providerID.trim() !== "" &&
          resolvedModel.modelID.trim() !== ""
            ? { providerID: resolvedModel.providerID, id: resolvedModel.modelID }
            : undefined,
      });
      if (created.error || !created.data?.id) {
        yield {
          kind: "error",
          error: created.error
            ? JSON.stringify(created.error).slice(0, 500)
            : "session.create did not return a session id",
        };
        return;
      }
      sessionID = created.data.id;
      createdSessionID = sessionID;
    }
    yield { kind: "session", sessionID };

    // Subscribe BEFORE submitting the prompt so we don't miss early events.
    // event.subscribe returns a ServerSentEventsResult with a .stream async iterator.
    const sub = await client.event.subscribe();
    const iterable = sub.stream;
    // Queue events from the SSE iterator into a local channel so we can both
    // pump events AND await the prompt promise concurrently.
    const channel: OpencodeStreamEvent[] = [];
    let pendingResolve: (() => void) | undefined;
    const notify = () => {
      const r = pendingResolve;
      pendingResolve = undefined;
      r?.();
    };
    let streamClosed = false;
    // Track tool-call inputs by callID so we can attach filePath when the
    // session.next.tool.called event fires.
    const toolInputByCall = new Map<string, Record<string, unknown>>();
    const toolStartByCall = new Map<string, number>();
    const toolNameByCall = new Map<string, string>();

    const consumeEvent = (event: Event) => {
      const t = (event as { type: string }).type;
      // Only forward events that mention our sessionID — `event.subscribe()`
      // is project-wide, so a noisy concurrent session would otherwise pollute
      // our SSE stream.
      const props = (event as { properties?: Record<string, unknown> }).properties;
      const evSession = props && typeof props.sessionID === "string" ? (props.sessionID as string) : undefined;
      if (evSession && sessionID && evSession !== sessionID) return;

      if (t === "session.status" && props) {
        const status = props.status as { type: string; attempt?: number; message?: string };
        if (status?.type === "retry") {
          channel.push({
            kind: "status",
            status: "retry",
            attempt: status.attempt,
            message: status.message,
          });
        } else if (status?.type) {
          channel.push({ kind: "status", status: status.type });
        }
      } else if (t === "session.error" && props) {
        const e = props.error as { name?: string; data?: unknown } | undefined;
        const nm = e && "name" in e ? String(e.name) : "Error";
        const msg = e && typeof e === "object" && "data" in e
          ? JSON.stringify(e.data).slice(0, 400)
          : JSON.stringify(e).slice(0, 400);
        channel.push({ kind: "error", error: `${nm}: ${msg}` });
      } else if (t === "session.idle") {
        channel.push({ kind: "status", status: "idle" });
      } else if (t === "session.next.text.delta" && props) {
        const delta = typeof props.delta === "string" ? props.delta : "";
        if (delta) channel.push({ kind: "text.delta", text: delta });
      } else if (t === "session.next.reasoning.delta" && props) {
        const delta = typeof props.delta === "string" ? props.delta : "";
        if (delta) channel.push({ kind: "reasoning.delta", text: delta });
      } else if (t === "session.next.tool.called" && props) {
        const callID = String(props.callID || "");
        const tool = String(props.tool || "?");
        const input = (props.input as Record<string, unknown>) || {};
        toolInputByCall.set(callID, input);
        toolStartByCall.set(callID, Date.now());
        toolNameByCall.set(callID, tool);
        const filePath = typeof input.filePath === "string" ? (input.filePath as string) : undefined;
        channel.push({
          kind: "tool.update",
          callID,
          tool,
          status: "running",
          ...(filePath ? { filePath } : {}),
        });
      } else if (t === "session.next.tool.success" && props) {
        const callID = String(props.callID || "");
        const start = toolStartByCall.get(callID);
        const filePath = (() => {
          const input = toolInputByCall.get(callID);
          return input && typeof input.filePath === "string" ? (input.filePath as string) : undefined;
        })();
        const tool = toolNameByCall.get(callID) || "?";
        const durationMs = typeof start === "number" ? Math.max(0, Date.now() - start) : undefined;
        channel.push({
          kind: "tool.update",
          callID,
          tool,
          status: "completed",
          ...(filePath ? { filePath } : {}),
          ...(typeof durationMs === "number" ? { durationMs } : {}),
        });
      } else if (t === "session.next.tool.failed" && props) {
        const callID = String(props.callID || "");
        const tool = toolNameByCall.get(callID) || "?";
        const err = props.error as { message?: string } | undefined;
        const errorMsg = err?.message || JSON.stringify(err).slice(0, 200);
        const start = toolStartByCall.get(callID);
        const durationMs = typeof start === "number" ? Math.max(0, Date.now() - start) : undefined;
        channel.push({
          kind: "tool.update",
          callID,
          tool,
          status: "error",
          error: errorMsg,
          ...(typeof durationMs === "number" ? { durationMs } : {}),
        });
      } else if (t === "todo.updated" && props) {
        const todos = Array.isArray(props.todos) ? (props.todos as Array<{ content: string; status: string; priority: string }>) : [];
        channel.push({ kind: "todo", todos });
      } else if (t === "file.edited" && props) {
        const file = typeof props.file === "string" ? props.file : undefined;
        if (file) channel.push({ kind: "file.edited", path: file });
      }
      // We rely on session.prompt's return value to know when this turn is
      // done — session.idle can fire for transient sub-step transitions.
      notify();
    };

    // Start pulling events; promise resolves on stream end.
    const eventPump = (async () => {
      try {
        for await (const ev of iterable as AsyncIterable<Event>) {
          if (signal?.aborted) break;
          consumeEvent(ev);
        }
      } catch (err) {
        const e = err as Error;
        channel.push({ kind: "error", error: `event stream: ${e?.message || String(err)}` });
        notify();
      } finally {
        streamClosed = true;
        notify();
      }
    })();
    stop = () => {
      streamClosed = true;
      notify();
    };

    const agentName = (body.agent || "build").trim();
    const tools = toolsMask(body.toolsEnabled);
    const timeoutMs = Math.max(1, (body.timeoutSec ?? 600) * 1000);

    const promptPromise = withTimeout(
      client.session.prompt({
        sessionID,
        directory: cwd,
        system: (body.systemPrompt || "").trim() || undefined,
        model:
          resolvedModel != null
            ? { providerID: resolvedModel.providerID, modelID: resolvedModel.modelID }
            : undefined,
        agent: agentName,
        ...(tools ? { tools } : {}),
        parts: [{ type: "text", text: body.prompt }],
      }),
      timeoutMs
    );

    let promptDone = false;
    // Wrap outcome in a holder object so TS doesn't narrow it through the
    // synchronous control flow — the closure below assigns it asynchronously.
    const outcomeRef: { value: PromptOutcome } = { value: { kind: "pending" } };
    promptPromise
      .then((res) => {
        if (res.error) {
          outcomeRef.value = { kind: "err", error: JSON.stringify(res.error).slice(0, 500) };
        } else if (!res.data?.info) {
          outcomeRef.value = { kind: "err", error: "prompt returned no assistant envelope" };
        } else {
          const info = res.data.info as AssistantMessage;
          const errUnknown = (info as { error?: unknown }).error;
          if (errUnknown !== undefined && isStructuredOutputError(errUnknown)) {
            outcomeRef.value = {
              kind: "structured-failed",
              error: `StructuredOutputError: ${(errUnknown as StructuredOutputError).data.message}`,
            };
          } else if (errUnknown) {
            const e = errUnknown as { name?: string };
            const nm = e && "name" in e ? String(e.name) : "Error";
            outcomeRef.value = { kind: "err", error: `${nm}: ${JSON.stringify(e).slice(0, 300)}` };
          } else {
            outcomeRef.value = { kind: "ok", assistant: info, parts: (res.data.parts ?? []) as Part[] };
          }
        }
        promptDone = true;
        notify();
      })
      .catch((err) => {
        const e = err as Error;
        outcomeRef.value = { kind: "err", error: `${e?.name || "Error"}: ${e?.message || String(err)}` };
        promptDone = true;
        notify();
      });

    // Pump until prompt resolves AND the channel is drained.
    while (true) {
      while (channel.length) {
        const ev = channel.shift();
        if (ev) yield ev;
      }
      if (promptDone) break;
      if (streamClosed && !promptDone) {
        // The event stream died unexpectedly — keep waiting on the prompt, but
        // we won't get more deltas. Spin sleep until promptDone flips.
        await new Promise((r) => setTimeout(r, 50));
        continue;
      }
      if (signal?.aborted) {
        channel.push({ kind: "error", error: "aborted" });
        promptDone = true;
        break;
      }
      await new Promise<void>((resolve) => {
        pendingResolve = resolve;
        // Safety: don't hang forever if a notify is dropped.
        setTimeout(resolve, 250);
      });
    }
    while (channel.length) {
      const ev = channel.shift();
      if (ev) yield ev;
    }

    // Build the final result envelope.
    const outcome = outcomeRef.value;
    if (outcome.kind === "pending") {
      yield {
        kind: "done",
        result: { ok: false, sessionID, error: "prompt did not resolve", model: modelSpec },
      };
    } else if (outcome.kind === "err") {
      yield {
        kind: "done",
        result: {
          ok: false,
          sessionID,
          error: outcome.error,
          model: modelSpec,
        },
      };
    } else if (outcome.kind === "structured-failed") {
      yield {
        kind: "done",
        result: {
          ok: false,
          sessionID,
          structuredOutputFailed: true,
          error: outcome.error,
          model: modelSpec,
        },
      };
    } else {
      const assistant = outcome.assistant;
      const allParts = await readAllParts(client, sessionID, cwd);
      const agg = aggregateMetrics(assistant, allParts);
      const toolNames = Array.from(new Set(agg.toolTrace.map((t) => t.name).filter(Boolean)));
      const answer =
        textFromParts(outcome.parts) ||
        (typeof (assistant as { structured?: unknown }).structured === "string"
          ? (assistant as { structured: string }).structured
          : "");
      yield {
        kind: "done",
        result: {
          ok: true,
          answer,
          sessionID,
          tokensIn: agg.tokensIn,
          tokensOut: agg.tokensOut,
          tokensReasoning: agg.tokensReasoning,
          cacheReadTokens: agg.cacheReadTokens,
          cacheWriteTokens: agg.cacheWriteTokens,
          costUsd: agg.costUsd,
          finishReason: agg.finishReason,
          providerID: agg.providerID,
          agentMode: agg.agentMode,
          model: modelSpec,
          toolCalls: agg.toolTrace.length,
          toolNames,
          toolTrace: agg.toolTrace,
          toolWallMs: agg.toolWallMs,
          groundingPaths: groundingPathsFromParts(allParts, cwd),
        },
      };
    }

    await eventPump.catch(() => undefined);
  } catch (err) {
    const e = err as Error;
    yield { kind: "error", error: `${e?.name || "Error"}: ${e?.message || String(err)}` };
  } finally {
    stop?.();
    if (createdSessionID && !body.keepSession) {
      await client?.session.delete({ sessionID: createdSessionID, directory: cwd }).catch(() => undefined);
    }
    if (serverClose) serverClose();
  }
}

/** Cancel a running prompt mid-flight. Intended for the UI's "Stop" button. */
export async function abortOpencodeSession(opts: {
  sessionID: string;
  cwd?: string;
  baseUrl?: string;
}): Promise<{ ok: boolean; error?: string }> {
  try {
    const cwd = await resolveDirectory(opts.cwd);
    const envBase = (process.env.OPENCODE_BASE_URL || "").trim();
    const bodyBase = (opts.baseUrl || "").trim();
    const baseUrl = bodyBase || envBase;
    if (!baseUrl) {
      // session.abort requires an HTTP endpoint to hit. When agent-node owns
      // the embedded server, we don't have its baseUrl outside the process —
      // the simplest path is to require a running server with OPENCODE_BASE_URL.
      return { ok: false, error: "abort requires OPENCODE_BASE_URL to point at a running server" };
    }
    const client = createOpencodeClient({ baseUrl, ...(cwd ? { directory: cwd } : {}) });
    const res = await client.session.abort({ sessionID: opts.sessionID, ...(cwd ? { directory: cwd } : {}) });
    if (res.error) return { ok: false, error: JSON.stringify(res.error).slice(0, 400) };
    return { ok: true };
  } catch (err) {
    const e = err as Error;
    return { ok: false, error: `${e?.name || "Error"}: ${e?.message || String(err)}` };
  }
}
