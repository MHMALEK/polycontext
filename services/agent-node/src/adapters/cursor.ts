/** Cursor SDK local execution — matches cookbook quickstart + stream then wait. */

import { Agent } from "@cursor/sdk";
import { serenaMcpConfig } from "./_serena.js";

export type CursorRunBody = {
  systemPrompt?: string;
  prompt: string;
  cwd: string;
  apiKey: string;
  modelId?: string;
  timeoutSec?: number;
};

function modelIdFrom(model: unknown): string | undefined {
  if (!model || typeof model !== "object") return undefined;
  const m = model as Record<string, unknown>;
  return (m.id as string) || (m.modelId as string) || (m.name as string);
}

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

export async function runCursor(body: CursorRunBody): Promise<{
  ok: boolean;
  answer?: string;
  error?: string;
  runId?: string;
  agentId?: string;
  model?: string;
  durationMs?: number;
  tokensIn?: number;
  tokensOut?: number;
  cacheReadTokens?: number;
  cacheWriteTokens?: number;
  toolCalls?: number;
  events?: unknown[];
  status?: string;
}> {
  const modelId = body.modelId || process.env.CURSOR_SDK_MODEL || "composer-2";
  const timeoutMs = (body.timeoutSec ?? 600) * 1000;
  const started = Date.now();
  const events: unknown[] = [];
  const usage: Record<string, number> = {};
  let toolCalls = 0;
  let agent: Awaited<ReturnType<typeof Agent.create>> | undefined;

  try {
    const mcpServers = serenaMcpConfig();
    agent = await Agent.create({
      apiKey: body.apiKey,
      name: "tech-decomposition agent-node",
      model: { id: modelId },
      local: { cwd: body.cwd },
      ...(mcpServers ? { mcpServers } : {}),
    });

    const message = body.systemPrompt
      ? `${body.systemPrompt}\n\n${body.prompt}`
      : body.prompt;
    const assistantChunks: string[] = [];
    const run = await agent.send(message, {
      local: { force: true },
      onDelta: ({ update }: { update?: { type?: string; usage?: Record<string, number> } }) => {
        if (!update || typeof update !== "object") return;
        if (update.type === "turn-ended" && update.usage) {
          Object.assign(usage, update.usage);
        }
        if (
          update.type === "tool-call-started" ||
          update.type === "tool-call-completed" ||
          update.type === "partial-tool-call"
        ) {
          toolCalls += 1;
        }
      },
      onStep: ({ step }: { step?: { type?: string } }) => {
        events.push({ type: "step", stepType: step?.type });
        if (events.length > 100) events.shift();
      },
    });

    await withTimeout(consumeStream(run as { stream: () => AsyncIterable<unknown> }, assistantChunks, events), timeoutMs);
    const result = await withTimeout(run.wait(), timeoutMs);

    const answer =
      (result.result as string | undefined) || assistantChunks.join("");
    const r = result as unknown as Record<string, unknown>;
    const usageFromResult = r["usage"] as
      | { inputTokens?: number; outputTokens?: number; cacheReadTokens?: number; cacheWriteTokens?: number }
      | undefined;

    if (result.status !== "finished") {
      return {
        ok: false,
        status: String(result.status),
        error: `cursor sdk run ended with status ${result.status}`,
        answer,
        runId: result.id,
        events: events.slice(-20),
      };
    }

    return {
      ok: true,
      answer,
      runId: result.id,
      agentId: agent.agentId,
      model: modelIdFrom(result.model) || modelId,
      durationMs:
        (result.durationMs as number | undefined) ?? Date.now() - started,
      tokensIn: usageFromResult?.inputTokens ?? (usage as { inputTokens?: number }).inputTokens,
      tokensOut: usageFromResult?.outputTokens ?? (usage as { outputTokens?: number }).outputTokens,
      cacheReadTokens: (usage as { cacheReadTokens?: number }).cacheReadTokens,
      cacheWriteTokens: (usage as { cacheWriteTokens?: number }).cacheWriteTokens,
      toolCalls,
      events: events.slice(-20),
    };
  } catch (err) {
    const e = err as Error;
    return {
      ok: false,
      error: `${e?.name || "Error"}: ${e?.message || String(e)}`,
      events: events.slice(-20),
    };
  } finally {
    try {
      const a = agent as { [Symbol.asyncDispose]?: () => Promise<void>; close?: () => void } | undefined;
      const dispose = a?.[Symbol.asyncDispose];
      if (typeof dispose === "function") {
        await dispose.call(agent);
      } else {
        a?.close?.();
      }
    } catch {
      // ignore
    }
  }
}

async function consumeStream(
  run: { stream: () => AsyncIterable<unknown> },
  assistantChunks: string[],
  events: unknown[]
) {
  for await (const raw of run.stream()) {
    const event = raw as {
      type?: string;
      message?: { content?: Array<{ type?: string; text?: string; name?: string }> };
      name?: string;
      status?: string;
      text?: string;
    };
    switch (event.type) {
      case "assistant":
        for (const block of event.message?.content ?? []) {
          if (block.type === "text" && block.text) {
            assistantChunks.push(block.text);
          } else if (block.name) {
            events.push({ type: "tool", name: block.name, status: "requested" });
            if (events.length > 100) events.shift();
          }
        }
        break;
      case "tool_call":
        events.push({
          type: "tool",
          name: event.name,
          status: event.status,
        });
        if (events.length > 100) events.shift();
        break;
      case "status":
        events.push({
          type: "status",
          status: event.status,
          message: (event as { message?: string }).message,
        });
        if (events.length > 100) events.shift();
        break;
      case "task":
        events.push({
          type: "task",
          status: event.status,
          text: event.text,
        });
        if (events.length > 100) events.shift();
        break;
      default:
        break;
    }
  }
}
