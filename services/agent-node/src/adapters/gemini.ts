/**
 * Gemini (`@google/genai`) — optional workspace access via CallableTool + AFC
 * (see Google Gen AI JS docs: automatic function calling with CallableTool).
 * @see https://googleapis.github.io/js-genai/
 */
import {
  GoogleGenAI,
  Type,
  FunctionCallingConfigMode,
  createPartFromFunctionResponse,
  type CallableTool,
  type Content,
  type FunctionCall,
  type Part,
  type Schema,
  type Tool,
} from "@google/genai";
import {
  workspaceReadFile,
  workspaceListDir,
  workspaceSearchFiles,
  workspaceGrepSearch,
  DEFAULT_MAX_BYTES,
} from "./workspace_tools.js";

export type GeminiRunBody = {
  systemPrompt?: string;
  prompt: string;
  apiKey: string;
  modelId?: string;
  timeoutSec?: number;
  /** Workspace root on the agent-node host (e.g. REPOS_ROOT); enables CallableTool + AFC. */
  cwd?: string;
  maxToolRounds?: number;
  /**
   * Optional Gemini ``responseSchema``. When set, the API forces the model to
   * emit JSON matching this schema for the final response. Tool calls during
   * AFC iterations are unaffected; only the terminal response is constrained.
   *
   * Used by the Python decompose path to make Gemini natively produce a
   * ``Decomposition``-shaped JSON object — the downstream structurer then
   * just re-validates it via pydantic-ai (fast path) rather than re-prompting
   * a separate Flash model.
   */
  responseSchema?: Record<string, unknown>;
  /**
   * When false, run the no-tools path even when ``cwd`` is set. The
   * model answers single-shot from whatever the prompt already contains
   * (used for the UI's "Grounded only" and "Direct" modes). Defaults
   * to true — i.e. tools are wired in whenever a workspace cwd is
   * present, which is the historical behavior.
   */
  toolsEnabled?: boolean;
};


const FILE_TOOLS: Tool = {
  functionDeclarations: [
    {
      name: "read_file",
      description:
        "Read a UTF-8 text file under the workspace. Path is relative to the workspace root (e.g. traceability/src/foo.ts).",
      parameters: {
        type: Type.OBJECT,
        properties: {
          path: {
            type: Type.STRING,
            description: "File path relative to workspace root",
          },
          maxBytes: {
            type: Type.INTEGER,
            description: `Optional max bytes (default ${DEFAULT_MAX_BYTES})`,
          },
        },
        required: ["path"],
      },
    },
    {
      name: "list_directory",
      description:
        "List non-hidden files and folders in a directory under the workspace. Path is relative to the workspace root; use . for the root.",
      parameters: {
        type: Type.OBJECT,
        properties: {
          path: {
            type: Type.STRING,
            description: "Directory path relative to workspace root",
          },
        },
        required: ["path"],
      },
    },
    {
      name: "search_files",
      description: "Search for files by name pattern in the workspace.",
      parameters: {
        type: Type.OBJECT,
        properties: {
          pattern: {
            type: Type.STRING,
            description: "The name pattern to search for (e.g. 'user' or '.ts')",
          },
        },
        required: ["pattern"],
      },
    },
    {
      name: "grep_search",
      description: "Search inside file contents for a specific string or regex pattern in the workspace.",
      parameters: {
        type: Type.OBJECT,
        properties: {
          query: {
            type: Type.STRING,
            description: "The string or regex pattern to search for inside files",
          },
        },
        required: ["query"],
      },
    },
  ],
};

async function executeToolCall(
  cwd: string,
  fc: FunctionCall
): Promise<Record<string, unknown>> {
  const name = fc.name ?? "";
  const args = fc.args ?? {};
  if (name === "read_file") {
    const p = String(args.path ?? "");
    const maxB = Number(args.maxBytes);
    const cap =
      Number.isFinite(maxB) && maxB > 0 ? Math.min(maxB, 500_000) : DEFAULT_MAX_BYTES;
    return workspaceReadFile(cwd, p, cap);
  }
  if (name === "list_directory") {
    return workspaceListDir(cwd, String(args.path ?? "."));
  }
  if (name === "search_files") {
    return workspaceSearchFiles(cwd, String(args.pattern ?? ""));
  }
  if (name === "grep_search") {
    return workspaceGrepSearch(cwd, String(args.query ?? ""));
  }
  return { error: `unknown tool: ${name}` };
}

export type ToolCallRecord = {
  name: string;
  argsPreview: string;
  resultPreview: string;
  durationMs: number;
};

/** Implements {@link CallableTool} so the SDK runs the AFC loop (see js-genai docs).
 *
 * Records every call (name, args summary, result summary, latency) into a
 * caller-provided array so the bridge can show what the model actually did.
 * Previously we only counted calls and had no visibility into the sequence —
 * which made it impossible to tell why a question got the wrong answer.
 */
class WorkspaceFsCallableTool implements CallableTool {
  constructor(
    private readonly cwd: string,
    private readonly trace: ToolCallRecord[],
  ) {}

  async tool(): Promise<Tool> {
    return FILE_TOOLS;
  }

  async callTool(functionCalls: FunctionCall[]): Promise<Part[]> {
    const parts: Part[] = [];
    let idx = 0;
    for (const fc of functionCalls) {
      const t0 = Date.now();
      const result = await executeToolCall(this.cwd, fc);
      const id = fc.id?.trim() || `call_${idx++}`;
      parts.push(
        createPartFromFunctionResponse(id, fc.name ?? "unknown", result)
      );
      this.trace.push({
        name: fc.name ?? "unknown",
        argsPreview: JSON.stringify(fc.args ?? {}).slice(0, 200),
        resultPreview: _summarizeToolResult(result).slice(0, 200),
        durationMs: Date.now() - t0,
      });
    }
    return parts;
  }
}

/**
 * Streaming variant of WorkspaceFsCallableTool. Emits a tool.update event
 * (status="running") before each call and (status="completed"|"error") after.
 * Lets the SSE bridge surface tool calls live without changing the AFC loop.
 */
export class StreamingWorkspaceFsCallableTool implements CallableTool {
  constructor(
    private readonly cwd: string,
    private readonly trace: ToolCallRecord[],
    private readonly onEvent: (ev: GeminiStreamEvent) => void,
  ) {}

  async tool(): Promise<Tool> {
    return FILE_TOOLS;
  }

  async callTool(functionCalls: FunctionCall[]): Promise<Part[]> {
    const parts: Part[] = [];
    let idx = 0;
    for (const fc of functionCalls) {
      const callID = fc.id?.trim() || `call_${idx++}`;
      const tool = fc.name ?? "unknown";
      const args = fc.args ?? {};
      const filePath =
        typeof (args as { path?: unknown }).path === "string"
          ? (args as { path: string }).path
          : undefined;
      this.onEvent({
        kind: "tool.update",
        callID,
        tool,
        status: "running",
        ...(filePath ? { filePath } : {}),
      });
      const t0 = Date.now();
      let result: Record<string, unknown>;
      let errored = false;
      try {
        result = await executeToolCall(this.cwd, fc);
        if ("error" in result && typeof result.error === "string") errored = true;
      } catch (err) {
        const e = err as Error;
        result = { error: `${e?.name || "Error"}: ${e?.message || String(err)}` };
        errored = true;
      }
      const durationMs = Date.now() - t0;
      parts.push(createPartFromFunctionResponse(callID, tool, result));
      this.trace.push({
        name: tool,
        argsPreview: JSON.stringify(args).slice(0, 200),
        resultPreview: _summarizeToolResult(result).slice(0, 200),
        durationMs,
      });
      this.onEvent({
        kind: "tool.update",
        callID,
        tool,
        status: errored ? "error" : "completed",
        durationMs,
        ...(filePath ? { filePath } : {}),
        ...(errored && typeof result.error === "string"
          ? { error: result.error }
          : {}),
      });
    }
    return parts;
  }
}

/** Normalized event union for the gemini SSE bridge. Mirrors the opencode
 * stream shape so the UI consumer (api.ts adapterAskStream) handles both. */
export type GeminiStreamEvent =
  | { kind: "session"; sessionID: string }
  | { kind: "status"; status: string }
  | { kind: "text.delta"; text: string }
  | {
      kind: "tool.update";
      callID: string;
      tool: string;
      status: "running" | "completed" | "error";
      filePath?: string;
      durationMs?: number;
      error?: string;
    }
  | { kind: "error"; error: string }
  | {
      kind: "done";
      result: {
        ok: boolean;
        answer?: string;
        model?: string;
        tokensIn?: number;
        tokensOut?: number;
        thoughtsTokens?: number;
        toolCalls?: number;
        toolTrace?: ToolCallRecord[];
        durationMs?: number;
        error?: string;
      };
    };

/** Streaming variant of runGemini. Yields text deltas (each model token batch
 * arrives as one chunk) and tool.update events while the AFC loop runs
 * transparently in the SDK. Terminal event is `done` with full metrics. */
export async function* streamGemini(body: GeminiRunBody): AsyncGenerator<GeminiStreamEvent, void, void> {
  const started = Date.now();
  const apiKey = body.apiKey;
  if (!apiKey.trim()) {
    yield { kind: "done", result: { ok: false, error: "apiKey required" } };
    return;
  }
  const modelId = body.modelId || process.env.GEMINI_SDK_MODEL || "gemini-2.5-pro";
  const timeoutMs = (body.timeoutSec ?? 600) * 1000;
  const cwd = (body.cwd || "").trim();
  const toolsEnabled = body.toolsEnabled !== false;
  yield { kind: "session", sessionID: `gemini:${modelId}` };
  yield { kind: "status", status: "running" };

  const ai = new GoogleGenAI({ apiKey });
  const buffered: GeminiStreamEvent[] = [];
  const toolTrace: ToolCallRecord[] = [];
  const onToolEvent = (ev: GeminiStreamEvent) => buffered.push(ev);

  // Build params identical to runGemini so the only delta is generateContent
  // → generateContentStream. Keeps response shapes + AFC behavior consistent.
  let promptText = body.prompt;
  let toolsConfig: Partial<Parameters<typeof ai.models.generateContentStream>[0]["config"]> = {};
  if (cwd && toolsEnabled) {
    const callable = new StreamingWorkspaceFsCallableTool(cwd, toolTrace, onToolEvent);
    const maxRemoteCalls = Math.min(80, Math.max(1, body.maxToolRounds ?? 48));
    promptText = `${body.prompt}\n\n[Workspace root on server: ${cwd}. Use read_file, list_directory, search_files, and grep_search to inspect code.]`;
    toolsConfig = {
      tools: [callable],
      toolConfig: { functionCallingConfig: { mode: FunctionCallingConfigMode.AUTO } },
      automaticFunctionCalling: { disable: false, maximumRemoteCalls: maxRemoteCalls },
    };
  }

  let finalText = "";
  let lastUsage: { promptTokenCount?: number; candidatesTokenCount?: number; thoughtsTokenCount?: number } | undefined;
  try {
    const stream = await ai.models.generateContentStream({
      model: modelId,
      contents: promptText,
      config: {
        ...(body.systemPrompt?.trim() ? { systemInstruction: body.systemPrompt } : {}),
        thinkingConfig: { thinkingBudget: -1, includeThoughts: false },
        temperature: 0.2,
        httpOptions: { timeout: timeoutMs },
        ...toolsConfig,
      },
    });

    for await (const chunk of stream) {
      // Drain tool events that fired during this chunk first.
      while (buffered.length) {
        const ev = buffered.shift();
        if (ev) yield ev;
      }
      const text = chunk.text;
      if (typeof text === "string" && text.length > 0) {
        finalText += text;
        yield { kind: "text.delta", text };
      }
      if (chunk.usageMetadata) lastUsage = chunk.usageMetadata;
    }
    // Drain any trailing tool events.
    while (buffered.length) {
      const ev = buffered.shift();
      if (ev) yield ev;
    }

    yield {
      kind: "done",
      result: {
        ok: true,
        answer: finalText,
        model: modelId,
        tokensIn: lastUsage?.promptTokenCount,
        tokensOut: lastUsage?.candidatesTokenCount,
        thoughtsTokens: lastUsage?.thoughtsTokenCount,
        toolCalls: toolTrace.length,
        toolTrace,
        durationMs: Date.now() - started,
      },
    };
  } catch (err) {
    const e = err as Error;
    yield {
      kind: "done",
      result: {
        ok: false,
        error: `${e?.name || "Error"}: ${e?.message || String(e)}`,
        model: modelId,
        durationMs: Date.now() - started,
      },
    };
  }
}

function _summarizeToolResult(result: Record<string, unknown>): string {
  if ("error" in result && typeof result.error === "string") {
    return `ERROR: ${result.error}`;
  }
  // Each tool returns one of { content, entries, files, matches } strings.
  for (const key of ["content", "entries", "files", "matches"]) {
    const v = result[key];
    if (typeof v === "string") {
      const firstLines = v.split("\n").slice(0, 3).join(" / ");
      return `${key}(${v.length}c): ${firstLines}`;
    }
  }
  return JSON.stringify(result).slice(0, 200);
}

function countFunctionResponsesInHistory(history: Content[] | undefined): number {
  if (!history?.length) return 0;
  let n = 0;
  for (const c of history) {
    for (const p of c.parts ?? []) {
      if (p && typeof p === "object" && "functionResponse" in p && p.functionResponse) {
        n += 1;
      }
    }
  }
  return n;
}

export async function runGemini(body: GeminiRunBody): Promise<{
  ok: boolean;
  answer?: string;
  error?: string;
  model?: string;
  durationMs?: number;
  tokensIn?: number;
  tokensOut?: number;
  toolCalls?: number;
  toolTrace?: ToolCallRecord[];
  thoughtsTokens?: number;
}> {
  const started = Date.now();
  const apiKey = body.apiKey;
  if (!apiKey.trim()) {
    return { ok: false, error: "apiKey required" };
  }
  const modelId =
    body.modelId || process.env.GEMINI_SDK_MODEL || "gemini-2.5-pro";
  const timeoutMs = (body.timeoutSec ?? 600) * 1000;
  const cwd = (body.cwd || "").trim();
  // Honor the per-request tools_enabled flag from the Python adapter. When
  // false, take the no-tools path even with a valid workspace cwd — the
  // model answers single-shot from whatever's already in the prompt.
  // This implements the UI's "Grounded only" / "Direct" modes for gemini.
  const toolsEnabled = body.toolsEnabled !== false; // default true

  if (!cwd || !toolsEnabled) {
    const ai = new GoogleGenAI({ apiKey });
    try {
      const response = await ai.models.generateContent({
        model: modelId,
        contents: body.prompt,
        config: {
          ...(body.systemPrompt?.trim()
            ? { systemInstruction: body.systemPrompt }
            : {}),
          thinkingConfig: { thinkingBudget: -1, includeThoughts: false },
          temperature: 0.2,
          httpOptions: { timeout: timeoutMs },
        },
      });
      const usage = response.usageMetadata;
      return {
        ok: true,
        answer: response.text ?? "",
        model: modelId,
        durationMs: Date.now() - started,
        tokensIn: usage?.promptTokenCount,
        tokensOut: usage?.candidatesTokenCount,
        toolCalls: 0,
      };
    } catch (err) {
      const e = err as Error;
      return {
        ok: false,
        error: `${e?.name || "Error"}: ${e?.message || String(e)}`,
        model: modelId,
        durationMs: Date.now() - started,
      };
    }
  }

  // Bumped from 24 → 48 default to give Gemini room to do thorough multi-step
  // exploration on complex tickets. Cursor's composer-2 routinely makes 25–30
  // tool calls per question in our eval; capping Gemini at 24 was leaving it
  // truncated mid-investigation on the harder cases.
  const maxRemoteCalls = Math.min(80, Math.max(1, body.maxToolRounds ?? 48));
  const ai = new GoogleGenAI({ apiKey });
  const toolTrace: ToolCallRecord[] = [];
  const callable = new WorkspaceFsCallableTool(cwd, toolTrace);

  const promptText = `${body.prompt}\n\n[Workspace root on server: ${cwd}. Use read_file, list_directory, search_files, and grep_search to inspect code.]`;

  try {
    const response = await ai.models.generateContent({
      model: modelId,
      contents: promptText,
      config: {
        systemInstruction: body.systemPrompt?.trim(),
        tools: [callable],
        toolConfig: {
          functionCallingConfig: { mode: FunctionCallingConfigMode.AUTO },
        },
        automaticFunctionCalling: {
          disable: false,
          maximumRemoteCalls: maxRemoteCalls,
        },
        // Extended thinking ON with an automatic budget. Gemini 2.5 Pro has
        // thinking but the default budget is conservative — without explicitly
        // setting -1 (automatic, model decides) it under-reasons on
        // multi-step code questions. This matches what gemini-cli and
        // Sourcebot's Gemini integration do internally.
        thinkingConfig: {
          thinkingBudget: -1,
          includeThoughts: false,
        },
        // Lower temperature gives more grounded answers — less inclined to
        // invent file paths. The forced-tool-use system prompt still does
        // the heavy lifting; this just trims the variance.
        temperature: 0.2,
        // When the caller wants a schema-constrained response (decompose
        // path), force the API to emit JSON matching the schema. Tool calls
        // during AFC are still free-form; only the terminal response is
        // constrained. Cuts the downstream pydantic-ai structurer call out
        // for the common case where Gemini natively produces a valid
        // Decomposition.
        ...(body.responseSchema
          ? {
              responseMimeType: "application/json",
              responseSchema: body.responseSchema as Schema,
            }
          : {}),
        httpOptions: { timeout: timeoutMs },
      },
    });

    const um = response.usageMetadata;
    // Prefer the locally-recorded trace (every callTool() invocation) since
    // it's authoritative; fall back to history-counting if for some reason
    // the trace is empty.
    const toolCalls =
      toolTrace.length ||
      countFunctionResponsesInHistory(response.automaticFunctionCallingHistory);

    return {
      ok: true,
      answer: response.text ?? "",
      model: modelId,
      durationMs: Date.now() - started,
      tokensIn: um?.promptTokenCount,
      tokensOut: um?.candidatesTokenCount,
      thoughtsTokens: (um as { thoughtsTokenCount?: number } | undefined)?.thoughtsTokenCount,
      toolCalls,
      toolTrace,
    };
  } catch (err) {
    const e = err as Error;
    return {
      ok: false,
      error: `${e?.name || "Error"}: ${e?.message || String(e)}`,
      model: modelId,
      durationMs: Date.now() - started,
    };
  }
}
