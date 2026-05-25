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

// ---------------------------------------------------------------------------
// Streaming
// ---------------------------------------------------------------------------

/** Normalized event union — same shape as the other adapters' streamers. */
export type CursorStreamEvent =
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
        toolNames?: string[];
      };
    };

/**
 * Stream a Cursor SDK run. Uses ``agent.send(...).stream()`` to iterate
 * raw SDK events and maps them to our normalized union.
 *
 * Cursor's stream emits:
 *  - assistant.message with content blocks: type="text" (text deltas) and
 *    block.name set for tool-call blocks (we tap the tool name).
 *  - tool_call: status transitions (we map to running/completed/error).
 *  - status: high-level run state we pass through as kind="status".
 *
 * Note: Cursor doesn't expose per-token deltas the same way Anthropic /
 * Gemini / OpenAI do — assistant.message blocks tend to be full-line or
 * fully-formed chunks. We still emit them as "text.delta" so the UI feels
 * live; the user just gets larger chunks than from the other adapters.
 */
export async function* streamCursor(body: CursorRunBody): AsyncGenerator<CursorStreamEvent, void, void> {
  const modelId = body.modelId || process.env.CURSOR_SDK_MODEL || "composer-2";
  const timeoutMs = (body.timeoutSec ?? 600) * 1000;
  const started = Date.now();
  const usage: Record<string, number> = {};
  const toolStart = new Map<string, { name: string; startMs: number; filePath?: string }>();
  const toolNames = new Set<string>();
  let agent: Awaited<ReturnType<typeof Agent.create>> | undefined;
  let assistantText = "";

  yield { kind: "session", sessionID: `cursor:${modelId}` };
  yield { kind: "status", status: "running" };

  try {
    const mcpServers = serenaMcpConfig();
    agent = await Agent.create({
      apiKey: body.apiKey,
      name: "tech-decomposition agent-node",
      model: { id: modelId },
      local: { cwd: body.cwd },
      ...(mcpServers ? { mcpServers } : {}),
    });

    const message = body.systemPrompt ? `${body.systemPrompt}\n\n${body.prompt}` : body.prompt;
    const run = await agent.send(message, {
      local: { force: true },
      onDelta: ({ update }: { update?: { type?: string; usage?: Record<string, number> } }) => {
        if (update?.type === "turn-ended" && update.usage) {
          Object.assign(usage, update.usage);
        }
      },
    });

    // Buffer events from the SDK stream into a channel so we can yield
    // them as we go. ``run.stream()`` is the SDK's async iterable.
    const stream = (run as { stream: () => AsyncIterable<Record<string, unknown>> }).stream();
    const startWall = Date.now();

    for await (const ev of stream) {
      if (Date.now() - startWall > timeoutMs) {
        yield { kind: "error", error: `timeout after ${timeoutMs}ms` };
        break;
      }
      const type = ev.type as string | undefined;
      if (type === "assistant") {
        const content = (ev as { message?: { content?: Array<{ type?: string; text?: string; name?: string; id?: string; input?: Record<string, unknown> }> } }).message?.content;
        for (const block of content || []) {
          if (block.type === "text" && block.text) {
            assistantText += block.text;
            yield { kind: "text.delta", text: block.text };
          } else if (block.name) {
            // Inline tool-use block in the assistant message — emit
            // running. The matching tool_call lifecycle event later fills
            // in the duration on completion.
            const callID = block.id || `${block.name}_${toolStart.size}`;
            const filePath =
              typeof block.input?.path === "string"
                ? (block.input.path as string)
                : typeof block.input?.file_path === "string"
                  ? (block.input.file_path as string)
                  : undefined;
            if (!toolStart.has(callID)) {
              toolStart.set(callID, { name: block.name, startMs: Date.now(), ...(filePath ? { filePath } : {}) });
              toolNames.add(block.name);
              yield {
                kind: "tool.update",
                callID,
                tool: block.name,
                status: "running",
                ...(filePath ? { filePath } : {}),
              };
            }
          }
        }
      } else if (type === "tool_call") {
        const name = (ev.name as string | undefined) || "tool";
        const status = (ev.status as string | undefined) || "";
        const id = (ev as { id?: string }).id || `${name}_call`;
        if (status === "started" || status === "in_progress") {
          if (!toolStart.has(id)) {
            toolStart.set(id, { name, startMs: Date.now() });
            toolNames.add(name);
            yield { kind: "tool.update", callID: id, tool: name, status: "running" };
          }
        } else if (status === "completed" || status === "succeeded") {
          const meta = toolStart.get(id);
          const durationMs = meta ? Date.now() - meta.startMs : undefined;
          yield {
            kind: "tool.update",
            callID: id,
            tool: meta?.name || name,
            status: "completed",
            ...(meta?.filePath ? { filePath: meta.filePath } : {}),
            ...(typeof durationMs === "number" ? { durationMs } : {}),
          };
        } else if (status === "failed" || status === "errored" || status === "error") {
          const meta = toolStart.get(id);
          const durationMs = meta ? Date.now() - meta.startMs : undefined;
          yield {
            kind: "tool.update",
            callID: id,
            tool: meta?.name || name,
            status: "error",
            ...(typeof durationMs === "number" ? { durationMs } : {}),
          };
        }
      } else if (type === "status") {
        const status = (ev.status as string | undefined) || "";
        if (status) yield { kind: "status", status };
      }
    }

    // Wait for the run to finalize so we can read totals + final status.
    const result = await withTimeout(run.wait(), Math.max(1000, timeoutMs - (Date.now() - started)));
    const r = result as unknown as Record<string, unknown>;
    const usageFromResult = r["usage"] as
      | { inputTokens?: number; outputTokens?: number; cacheReadTokens?: number; cacheWriteTokens?: number }
      | undefined;
    const finalAnswer = (result.result as string | undefined) || assistantText;

    if (result.status !== "finished") {
      yield {
        kind: "done",
        result: {
          ok: false,
          error: `cursor sdk run ended with status ${result.status}`,
          answer: finalAnswer,
          runId: result.id,
          model: modelIdFrom(result.model) || modelId,
          durationMs: (result.durationMs as number | undefined) ?? Date.now() - started,
          toolCalls: toolStart.size,
          toolNames: Array.from(toolNames),
        },
      };
      return;
    }
    yield {
      kind: "done",
      result: {
        ok: true,
        answer: finalAnswer,
        runId: result.id,
        agentId: agent.agentId,
        model: modelIdFrom(result.model) || modelId,
        durationMs: (result.durationMs as number | undefined) ?? Date.now() - started,
        tokensIn: usageFromResult?.inputTokens ?? (usage as { inputTokens?: number }).inputTokens,
        tokensOut: usageFromResult?.outputTokens ?? (usage as { outputTokens?: number }).outputTokens,
        cacheReadTokens: (usage as { cacheReadTokens?: number }).cacheReadTokens,
        cacheWriteTokens: (usage as { cacheWriteTokens?: number }).cacheWriteTokens,
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
        answer: assistantText || undefined,
        model: modelId,
        durationMs: Date.now() - started,
        toolCalls: toolStart.size,
        toolNames: Array.from(toolNames),
      },
    };
  } finally {
    try {
      const a = agent as { [Symbol.asyncDispose]?: () => Promise<void>; close?: () => void } | undefined;
      const dispose = a?.[Symbol.asyncDispose];
      if (typeof dispose === "function") await dispose.call(agent);
      else a?.close?.();
    } catch {
      // ignore
    }
  }
}
