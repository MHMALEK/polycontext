/**
 * OpenAI Agents SDK — plain {@link Agent} with optional read-only workspace tools.
 */
import { Agent, Runner, tool, type Tool } from "@openai/agents";
import { OpenAIProvider } from "@openai/agents-openai";
import type { Usage } from "@openai/agents-core";
import { z } from "zod";
import {
  workspaceReadFile,
  workspaceListDir,
  workspaceSearchFiles,
  workspaceGrepSearch,
  DEFAULT_MAX_BYTES,
} from "./workspace_tools.js";

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

  try {
    const agent = new Agent({
      name: "tech-decomposition",
      instructions,
      model: modelId,
      modelSettings: cwd ? { toolChoice: "required", parallelToolCalls: true } : {},
      tools: cwd ? createWorkspaceTools(cwd) : [],
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
  }
}
