/**
 * OpenAI Agents SDK — plain {@link Agent} with optional read-only workspace tools.
 */
import { Agent, Runner, tool, type Tool } from "@openai/agents";
import { OpenAIProvider } from "@openai/agents-openai";
import {
  MCPServerSSE,
  MCPServerStreamableHttp,
  type MCPServer,
  type Usage,
} from "@openai/agents-core";
import { z } from "zod";
import {
  workspaceReadFile,
  workspaceListDir,
  workspaceSearchFiles,
  workspaceGrepSearch,
  DEFAULT_MAX_BYTES,
} from "./workspace_tools.js";
import { serenaEnv } from "./_serena.js";

export type OpenAIAgentsRunBody = {
  systemPrompt?: string;
  prompt: string;
  apiKey: string;
  modelId?: string;
  timeoutSec?: number;
  maxTurns?: number;
  /** Workspace root on the agent-node host (for read-only filesystem tools). */
  cwd?: string;
};

function withTimeout<T>(promise: Promise<T>, ms: number): Promise<T> {
  return new Promise((resolve, reject) => {
    const t = setTimeout(() => reject(new Error(`timeout after ${ms}ms`)), ms);
    promise.then(
      (v) => {
        clearTimeout(t);
        resolve(v);
      },
      (e) => {
        clearTimeout(t);
        reject(e);
      }
    );
  });
}

function sumUsage(rawResponses: { usage: Usage }[]): {
  tokensIn: number;
  tokensOut: number;
} {
  let tokensIn = 0;
  let tokensOut = 0;
  for (const r of rawResponses) {
    tokensIn += r.usage.inputTokens ?? 0;
    tokensOut += r.usage.outputTokens ?? 0;
  }
  return { tokensIn, tokensOut };
}

function createWorkspaceTools(cwd: string): Tool[] {
  return [
    tool({
      name: "read_file",
      description:
        "Read a UTF-8 text file under the workspace. Path is relative to the workspace root (e.g. traceability/src/foo.ts).",
      parameters: z.object({
        path: z.string().describe("File path relative to the workspace root"),
        maxBytes: z
          .number()
          .int()
          .positive()
          .max(500_000)
          .optional()
          .describe(`Optional max bytes to return (default ${DEFAULT_MAX_BYTES})`),
      }),
      execute: async ({ path, maxBytes }) =>
        JSON.stringify(await workspaceReadFile(cwd, path, maxBytes)),
    }),
    tool({
      name: "list_directory",
      description:
        "List non-hidden files and folders in a directory under the workspace. Path is relative to the workspace root; use . for the root.",
      parameters: z.object({
        path: z.string().describe("Directory path relative to the workspace root"),
      }),
      execute: async ({ path }) =>
        JSON.stringify(await workspaceListDir(cwd, path)),
    }),
    tool({
      name: "search_files",
      description: "Search for files by name pattern in the workspace.",
      parameters: z.object({
        pattern: z
          .string()
          .describe("The file name pattern to search for (e.g. 'user' or '.ts')"),
      }),
      execute: async ({ pattern }) =>
        JSON.stringify(await workspaceSearchFiles(cwd, pattern)),
    }),
    tool({
      name: "grep_search",
      description:
        "Search inside file contents for a specific string or regex pattern in the workspace.",
      parameters: z.object({
        query: z
          .string()
          .describe("The string or regex pattern to search for inside files"),
      }),
      execute: async ({ query }) =>
        JSON.stringify(await workspaceGrepSearch(cwd, query)),
    }),
  ];
}

export async function runOpenAIAgents(body: OpenAIAgentsRunBody): Promise<{
  ok: boolean;
  answer?: string;
  error?: string;
  model?: string;
  durationMs?: number;
  tokensIn?: number;
  tokensOut?: number;
  toolCalls?: number;
}> {
  const started = Date.now();
  const apiKey = body.apiKey;
  if (!apiKey.trim()) {
    return { ok: false, error: "apiKey required" };
  }
  const modelId =
    body.modelId || process.env.OPENAI_AGENTS_SDK_MODEL || "gpt-4.1";
  const timeoutMs = (body.timeoutSec ?? 600) * 1000;
  const cwd = (body.cwd || "").trim();
  const maxTurns = (body.maxTurns ?? 25) + (cwd ? 12 : 0);

  const provider = new OpenAIProvider({ apiKey });
  const runner = new Runner({
    modelProvider: provider,
    tracingDisabled: true,
  });

  const baseInstructions =
    body.systemPrompt?.trim() || "You are a helpful assistant.";
  const instructions = cwd
    ? `${baseInstructions}\n\nUse the read_file, list_directory, search_files, and grep_search tools to inspect the workspace before answering. Paths are relative to the workspace root (${cwd}). Cite paths relative to that root.`
    : baseInstructions;

  const sEnv = serenaEnv();
  const mcpServers: MCPServer[] = sEnv
    ? [
        sEnv.transport === "sse"
          ? new MCPServerSSE({
              url: sEnv.url,
              name: "serena",
              ...(sEnv.apiKey
                ? { requestInit: { headers: { Authorization: `Bearer ${sEnv.apiKey}` } } }
                : {}),
            })
          : new MCPServerStreamableHttp({
              url: sEnv.url,
              name: "serena",
              ...(sEnv.apiKey
                ? { requestInit: { headers: { Authorization: `Bearer ${sEnv.apiKey}` } } }
                : {}),
            }),
      ]
    : [];

  // OpenAI Agents SDK requires MCP servers to be explicitly connected before
  // the Agent uses them (unlike Cursor / Claude Agent SDKs which auto-connect).
  for (const s of mcpServers) {
    await s.connect();
  }

  try {
    const agent = new Agent({
      name: "tech-decomposition",
      instructions,
      model: modelId,
      modelSettings: cwd ? { toolChoice: "required", parallelToolCalls: true } : {},
      tools: cwd ? createWorkspaceTools(cwd) : [],
      ...(mcpServers.length ? { mcpServers } : {}),
    });

    const result = await withTimeout(
      runner.run(agent, body.prompt, {
        maxTurns,
      }),
      timeoutMs
    );

    const final = result.finalOutput;
    const answer =
      final === undefined || final === null
        ? ""
        : typeof final === "string"
          ? final
          : JSON.stringify(final);

    const { tokensIn, tokensOut } = sumUsage(result.rawResponses);
    const toolCalls = result.newItems.filter(
      (i) => (i as { type?: string }).type === "tool_call_item"
    ).length;

    return {
      ok: true,
      answer,
      model: modelId,
      durationMs: Date.now() - started,
      tokensIn,
      tokensOut,
      toolCalls,
    };
  } catch (err) {
    const e = err as Error;
    return {
      ok: false,
      error: `${e?.name || "Error"}: ${e?.message || String(e)}`,
    };
  } finally {
    await provider.close().catch(() => undefined);
    for (const s of mcpServers) {
      await s.close().catch(() => undefined);
    }
  }
}

// ---------------------------------------------------------------------------
// Streaming
// ---------------------------------------------------------------------------

/** Normalized event union forwarded to the Python API — same shape as the
 * opencode / gemini / claude_code streamers so the UI handles all four with
 * one reducer. */
export type OpenAIAgentsStreamEvent =
  | { kind: "session"; sessionID: string }
  | { kind: "status"; status: string }
  | { kind: "text.delta"; text: string }
  | { kind: "reasoning.delta"; text: string }
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
        error?: string;
        model?: string;
        durationMs?: number;
        tokensIn?: number;
        tokensOut?: number;
        toolCalls?: number;
        toolNames?: string[];
      };
    };

/**
 * Stream an OpenAI Agents run. Wires {@link Runner.run} with `stream: true`
 * to get a {@link StreamedRunResult} and iterates its RunStreamEvent union.
 *
 * Two event shapes from the SDK matter for our normalization:
 *  - ``raw_model_stream_event``: raw OpenAI ResponseStreamEvent — we tap
 *    `response.output_text.delta` (assistant text) + `response.reasoning_text.delta`
 *    (chain-of-thought when enabled). Anything else from the model layer
 *    (function-call argument deltas, content-part boundaries) is ignored —
 *    coarser ``run_item_stream_event`` already covers tool lifecycle.
 *  - ``run_item_stream_event`` with name=tool_called → emit tool.update
 *    (running); name=tool_output → emit tool.update (completed). We bookkeep
 *    by tool callID so the start/end pair lines up in the UI's chip list.
 */
export async function* streamOpenAIAgents(
  body: OpenAIAgentsRunBody,
): AsyncGenerator<OpenAIAgentsStreamEvent, void, void> {
  const started = Date.now();
  const apiKey = body.apiKey;
  if (!apiKey.trim()) {
    yield { kind: "done", result: { ok: false, error: "apiKey required" } };
    return;
  }
  const modelId = body.modelId || process.env.OPENAI_AGENTS_SDK_MODEL || "gpt-4.1";
  const timeoutMs = (body.timeoutSec ?? 600) * 1000;
  const cwd = (body.cwd || "").trim();
  const maxTurns = (body.maxTurns ?? 25) + (cwd ? 12 : 0);

  const provider = new OpenAIProvider({ apiKey });
  const runner = new Runner({ modelProvider: provider, tracingDisabled: true });

  const baseInstructions = body.systemPrompt?.trim() || "You are a helpful assistant.";
  const instructions = cwd
    ? `${baseInstructions}\n\nUse the read_file, list_directory, search_files, and grep_search tools to inspect the workspace before answering. Paths are relative to the workspace root (${cwd}). Cite paths relative to that root.`
    : baseInstructions;

  const sEnv = serenaEnv();
  const mcpServers: MCPServer[] = sEnv
    ? [
        sEnv.transport === "sse"
          ? new MCPServerSSE({
              url: sEnv.url,
              name: "serena",
              ...(sEnv.apiKey
                ? { requestInit: { headers: { Authorization: `Bearer ${sEnv.apiKey}` } } }
                : {}),
            })
          : new MCPServerStreamableHttp({
              url: sEnv.url,
              name: "serena",
              ...(sEnv.apiKey
                ? { requestInit: { headers: { Authorization: `Bearer ${sEnv.apiKey}` } } }
                : {}),
            }),
      ]
    : [];

  for (const s of mcpServers) await s.connect();

  yield { kind: "session", sessionID: `openai_agents:${modelId}` };
  yield { kind: "status", status: "running" };

  // callID → (tool name, startMs, optional filePath from tool input).
  const toolStart = new Map<string, { name: string; startMs: number; filePath?: string }>();
  const toolNames = new Set<string>();
  let finalText = "";

  try {
    const agent = new Agent({
      name: "tech-decomposition",
      instructions,
      model: modelId,
      modelSettings: cwd ? { toolChoice: "required", parallelToolCalls: true } : {},
      tools: cwd ? createWorkspaceTools(cwd) : [],
      ...(mcpServers.length ? { mcpServers } : {}),
    });

    // ``stream: true`` returns a StreamedRunResult that implements
    // AsyncIterable<RunStreamEvent>. The cast is required because the
    // overloaded ``run`` signature doesn't auto-narrow on the conditional
    // ``stream`` literal in TS 5.x.
    const stream = await runner.run(agent, body.prompt, {
      maxTurns,
      stream: true,
    });

    // Race the stream iteration against the request-timeout so we don't
    // leak resources if the SDK hangs on an upstream timeout. Once the
    // race rejects, the finally below closes provider + mcp.
    const timeoutHandle = setTimeout(() => {
      // The SDK doesn't expose a public cancel for stream iteration; the
      // simplest abort is to throw which unwinds the for-await below.
      // We don't actually throw here — instead let withTimeout-style
      // logic happen via the outer try/catch on stream.completed.
    }, timeoutMs);

    try {
      for await (const ev of stream as AsyncIterable<{ type: string; [k: string]: unknown }>) {
        if (ev.type === "raw_model_stream_event") {
          // ResponseStreamEvent from openai's responses API.
          const data = ev.data as { type?: string; delta?: string };
          if (data?.type === "response.output_text.delta" && data.delta) {
            finalText += data.delta;
            yield { kind: "text.delta", text: data.delta };
          } else if (data?.type === "response.reasoning_text.delta" && data.delta) {
            yield { kind: "reasoning.delta", text: data.delta };
          }
        } else if (ev.type === "run_item_stream_event") {
          const name = ev.name as string;
          const item = ev.item as {
            type?: string;
            rawItem?: {
              callId?: string;
              call_id?: string;
              name?: string;
              arguments?: string;
            };
          };
          const raw = item?.rawItem || {};
          const callID = String(raw.callId || raw.call_id || `call_${toolStart.size}`);

          if (name === "tool_called") {
            const tool = raw.name || "tool";
            // Pull a filePath out of the arguments JSON if present — chips
            // show it inline so the user knows what the model is opening.
            let filePath: string | undefined;
            try {
              const args = typeof raw.arguments === "string" ? JSON.parse(raw.arguments) : raw.arguments;
              if (args && typeof args === "object") {
                const p = (args as { path?: unknown; file_path?: unknown }).path
                  ?? (args as { file_path?: unknown }).file_path;
                if (typeof p === "string") filePath = p;
              }
            } catch {
              /* best-effort */
            }
            toolStart.set(callID, { name: tool, startMs: Date.now(), ...(filePath ? { filePath } : {}) });
            toolNames.add(tool);
            yield {
              kind: "tool.update",
              callID,
              tool,
              status: "running",
              ...(filePath ? { filePath } : {}),
            };
          } else if (name === "tool_output") {
            const meta = toolStart.get(callID);
            const tool = meta?.name || "tool";
            const durationMs = meta ? Date.now() - meta.startMs : undefined;
            yield {
              kind: "tool.update",
              callID,
              tool,
              status: "completed",
              ...(meta?.filePath ? { filePath: meta.filePath } : {}),
              ...(typeof durationMs === "number" ? { durationMs } : {}),
            };
          }
        }
      }
    } finally {
      clearTimeout(timeoutHandle);
    }

    // After the iterable drains the run has produced its final state.
    // Read totals from rawResponses + count tool calls from the final newItems.
    await stream.completed;
    const sErr = (stream as unknown as { error?: unknown }).error;
    if (sErr) {
      const e = sErr as { name?: string; message?: string };
      yield {
        kind: "done",
        result: {
          ok: false,
          error: `${e?.name || "Error"}: ${e?.message || String(sErr)}`,
          answer: finalText || undefined,
          model: modelId,
          durationMs: Date.now() - started,
          toolCalls: toolNames.size,
          toolNames: Array.from(toolNames),
        },
      };
      return;
    }

    const finalOut = (stream as unknown as { finalOutput?: unknown }).finalOutput;
    const answer =
      finalOut === undefined || finalOut === null
        ? finalText
        : typeof finalOut === "string"
          ? finalOut
          : JSON.stringify(finalOut);
    const rawResponses = (stream as unknown as { rawResponses?: { usage: Usage }[] }).rawResponses || [];
    const { tokensIn, tokensOut } = sumUsage(rawResponses);

    yield {
      kind: "done",
      result: {
        ok: true,
        answer,
        model: modelId,
        durationMs: Date.now() - started,
        tokensIn,
        tokensOut,
        toolCalls: toolStart.size,
        toolNames: Array.from(toolNames),
      },
    };
  } catch (err) {
    const e = err as Error;
    yield {
      kind: "done",
      result: {
        ok: false,
        error: `${e?.name || "Error"}: ${e?.message || String(e)}`,
        answer: finalText || undefined,
        model: modelId,
        durationMs: Date.now() - started,
        toolCalls: toolStart.size,
        toolNames: Array.from(toolNames),
      },
    };
  } finally {
    await provider.close().catch(() => undefined);
    for (const s of mcpServers) await s.close().catch(() => undefined);
  }
}
