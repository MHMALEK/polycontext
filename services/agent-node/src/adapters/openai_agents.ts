/**
 * OpenAI Agents SDK — plain {@link Agent}, or {@link SandboxAgent} + local sandbox
 * when `cwd` is set (see @openai/agents-js README / SandboxAgent + UnixLocalSandboxClient).
 */
import path from "node:path";
import { Agent, Runner } from "@openai/agents";
import { OpenAIProvider } from "@openai/agents-openai";
import { SandboxAgent, localDir } from "@openai/agents/sandbox";
import { UnixLocalSandboxClient } from "@openai/agents/sandbox/local";
import type { Usage } from "@openai/agents-core";

export type OpenAIAgentsRunBody = {
  systemPrompt?: string;
  prompt: string;
  apiKey: string;
  modelId?: string;
  timeoutSec?: number;
  maxTurns?: number;
  /** Mounts this directory as `workspace` in a local Unix sandbox (filesystem tools). */
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
    ? `${baseInstructions}\n\nInspect the mounted workspace (manifest entry "workspace") to read code before answering. Cite paths relative to the workspace.`
    : baseInstructions;

  try {
    const agent = cwd
      ? new SandboxAgent({
          name: "tech-decomposition",
          instructions,
          model: modelId,
          defaultManifest: {
            entries: {
              workspace: localDir({ src: path.resolve(cwd) }),
            },
          },
        })
      : new Agent({
          name: "tech-decomposition",
          instructions,
          model: modelId,
        });

    const result = await withTimeout(
      runner.run(agent, body.prompt, {
        maxTurns,
        ...(cwd
          ? {
              sandbox: {
                client: new UnixLocalSandboxClient(),
              },
            }
          : {}),
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
