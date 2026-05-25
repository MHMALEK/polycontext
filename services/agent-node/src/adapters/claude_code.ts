/** Claude Code via @anthropic-ai/claude-agent-sdk (query API). */

import { query } from "@anthropic-ai/claude-agent-sdk";
import { serenaMcpConfig } from "./_serena.js";

export type ClaudeCodeRunBody = {
  systemPrompt?: string;
  prompt: string;
  cwd: string;
  apiKey: string;
  model?: string;
  timeoutSec?: number;
  maxTurns?: number;
};

/** Normalized event union forwarded to the Python API. Mirrors the
 * opencode/gemini stream shape so the UI reducer handles all adapters
 * with one code path. */
export type ClaudeCodeStreamEvent =
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
        sessionId?: string;
        model?: string;
        numTurns?: number;
        tokensIn?: number;
        tokensOut?: number;
        cacheReadTokens?: number;
        cacheWriteTokens?: number;
        costUsd?: number;
        durationMs?: number;
        stopReason?: string | null;
        toolCalls?: number;
        toolNames?: string[];
      };
    };

function scheduleAbort(ac: AbortController, ms: number): () => void {
  if (ms <= 0) return () => {};
  const t = setTimeout(() => {
    ac.abort(new Error(`timeout after ${ms}ms`));
  }, ms);
  return () => clearTimeout(t);
}

function assistantTextFromMessage(message: unknown): string {
  if (!message || typeof message !== "object") return "";
  const m = message as { content?: unknown };
  const content = m.content;
  if (!Array.isArray(content)) return "";
  const parts: string[] = [];
  for (const block of content) {
    if (!block || typeof block !== "object") continue;
    const b = block as { type?: string; text?: string };
    if (b.type === "text" && typeof b.text === "string") parts.push(b.text);
  }
  return parts.join("");
}

export async function runClaudeCode(body: ClaudeCodeRunBody): Promise<{
  ok: boolean;
  answer?: string;
  error?: string;
  sessionId?: string;
  model?: string;
  numTurns?: number;
  tokensIn?: number;
  tokensOut?: number;
  cacheReadTokens?: number;
  cacheWriteTokens?: number;
  costUsd?: number;
  durationMs?: number;
  stopReason?: string | null;
  toolCalls?: number;
  events?: unknown[];
}> {
  const timeoutMs = (body.timeoutSec ?? 600) * 1000;
  const abortController = new AbortController();
  const clearTimer = scheduleAbort(abortController, timeoutMs);
  const events: unknown[] = [];
  let toolCalls = 0;
  let lastAssistantFallback = "";

  const model = body.model || process.env.CLAUDE_CODE_MODEL || "claude-sonnet-4-5";
  const mcpServers = serenaMcpConfig();

  const q = query({
    prompt: body.prompt,
    options: {
      cwd: body.cwd,
      model,
      systemPrompt: body.systemPrompt,
      maxTurns: body.maxTurns,
      abortController,
      persistSession: false,
      permissionMode: "bypassPermissions",
      allowDangerouslySkipPermissions: true,
      tools: { type: "preset", preset: "claude_code" },
      ...(mcpServers ? { mcpServers } : {}),
      env: {
        ...process.env,
        ANTHROPIC_API_KEY: body.apiKey,
      },
    },
  });

  try {
    for await (const msg of q) {
      events.push({ type: msg.type, subtype: "subtype" in msg ? msg.subtype : undefined });
      if (events.length > 120) events.shift();

      if (msg.type === "assistant") {
        const text = assistantTextFromMessage(
          (msg as { message?: unknown }).message
        );
        if (text) lastAssistantFallback = text;
        const content = (msg as { message?: { content?: unknown[] } }).message
          ?.content;
        if (Array.isArray(content)) {
          for (const block of content) {
            if (
              block &&
              typeof block === "object" &&
              (block as { type?: string }).type === "tool_use"
            ) {
              toolCalls += 1;
            }
          }
        }
      }

      if (msg.type === "result") {
        if (msg.subtype === "success") {
          const m = msg as {
            result: string;
            session_id: string;
            num_turns: number;
            stop_reason: string | null;
            duration_ms: number;
            total_cost_usd: number;
            usage: {
              input_tokens: number;
              output_tokens: number;
              cache_read_input_tokens?: number | null;
              cache_creation_input_tokens?: number | null;
            };
          };
          return {
            ok: true,
            answer: m.result || lastAssistantFallback,
            sessionId: m.session_id,
            model,
            numTurns: m.num_turns,
            tokensIn: m.usage?.input_tokens,
            tokensOut: m.usage?.output_tokens,
            cacheReadTokens: m.usage?.cache_read_input_tokens ?? undefined,
            cacheWriteTokens: m.usage?.cache_creation_input_tokens ?? undefined,
            costUsd: m.total_cost_usd,
            durationMs: m.duration_ms,
            stopReason: m.stop_reason,
            toolCalls,
            events: events.slice(-20),
          };
        }

        const err = msg as {
          subtype: string;
          errors: string[];
          session_id?: string;
          num_turns: number;
          duration_ms: number;
          total_cost_usd: number;
          usage: {
            input_tokens: number;
            output_tokens: number;
            cache_read_input_tokens?: number | null;
          };
        };
        return {
          ok: false,
          error: err.errors?.length
            ? err.errors.join("; ")
            : `claude code finished with ${err.subtype}`,
          answer: lastAssistantFallback || undefined,
          sessionId: err.session_id,
          model,
          numTurns: err.num_turns,
          tokensIn: err.usage?.input_tokens,
          tokensOut: err.usage?.output_tokens,
          cacheReadTokens: err.usage?.cache_read_input_tokens ?? undefined,
          costUsd: err.total_cost_usd,
          durationMs: err.duration_ms,
          toolCalls,
          events: events.slice(-20),
        };
      }
    }

    return {
      ok: false,
      error: "claude code stream ended without a result message",
      answer: lastAssistantFallback || undefined,
      model,
      toolCalls,
      events: events.slice(-20),
    };
  } catch (err) {
    const e = err as Error;
    const name = e?.name || "Error";
    const aborted =
      name === "AbortError" || /aborted|timeout/i.test(e?.message || "");
    return {
      ok: false,
      error: aborted
        ? `aborted: ${e?.message || "timeout or cancel"}`
        : `${name}: ${e?.message || String(e)}`,
      answer: lastAssistantFallback || undefined,
      model,
      toolCalls,
      events: events.slice(-20),
    };
  } finally {
    clearTimer();
    try {
      q.close();
    } catch {
      /* ignore */
    }
  }
}

/**
 * Streaming variant of runClaudeCode. Wires the SDK's
 * ``includePartialMessages: true`` flag so SDKPartialAssistantMessage
 * (stream_event) messages fire per-token, then maps the raw Anthropic
 * streaming events to our normalized union.
 *
 * Events emitted:
 *   - session: first message with a session_id we see
 *   - text.delta: from BetaContentBlockDelta (type=text_delta)
 *   - reasoning.delta: from BetaContentBlockDelta (type=thinking_delta)
 *   - tool.update (running): when an assistant message contains tool_use blocks
 *   - tool.update (completed/error): when a user message contains tool_result
 *   - done: from the SDK's terminal result message
 */
export async function* streamClaudeCode(
  body: ClaudeCodeRunBody,
): AsyncGenerator<ClaudeCodeStreamEvent, void, void> {
  const timeoutMs = (body.timeoutSec ?? 600) * 1000;
  const abortController = new AbortController();
  const clearTimer = scheduleAbort(abortController, timeoutMs);
  const model = body.model || process.env.CLAUDE_CODE_MODEL || "claude-sonnet-4-5";
  const mcpServers = serenaMcpConfig();

  // Tool-call bookkeeping: when a tool_use block appears in an assistant
  // message, we emit "running" and stash (id → name + startMs + filePath).
  // When the matching tool_result appears in a later user message, we
  // emit "completed" or "error" with the elapsed wall time.
  const toolStart = new Map<string, { name: string; startMs: number; filePath?: string }>();
  const toolNames = new Set<string>();
  let sessionEmitted = false;
  let finalText = "";
  let toolCallCount = 0;

  yield { kind: "session", sessionID: `claude_code:${model}` };
  yield { kind: "status", status: "running" };

  const q = query({
    prompt: body.prompt,
    options: {
      cwd: body.cwd,
      model,
      systemPrompt: body.systemPrompt,
      maxTurns: body.maxTurns,
      abortController,
      persistSession: false,
      permissionMode: "bypassPermissions",
      allowDangerouslySkipPermissions: true,
      tools: { type: "preset", preset: "claude_code" },
      includePartialMessages: true,
      ...(mcpServers ? { mcpServers } : {}),
      env: { ...process.env, ANTHROPIC_API_KEY: body.apiKey },
    },
  });

  try {
    for await (const msg of q) {
      const m = msg as Record<string, unknown>;
      const sessionID = (m.session_id || m.sessionId) as string | undefined;
      if (sessionID && !sessionEmitted) {
        sessionEmitted = true;
        // Replace the synthetic session marker with the real session id.
        yield { kind: "session", sessionID };
      }

      // Per-token deltas — emitted while the model is still producing.
      if (m.type === "stream_event") {
        const event = m.event as { type?: string; delta?: { type?: string; text?: string } } | undefined;
        if (event?.type === "content_block_delta" && event.delta?.text) {
          if (event.delta.type === "text_delta") {
            finalText += event.delta.text;
            yield { kind: "text.delta", text: event.delta.text };
          } else if (event.delta.type === "thinking_delta") {
            yield { kind: "reasoning.delta", text: event.delta.text };
          }
        }
        continue;
      }

      // Assistant messages: scan content blocks for tool_use → emit running.
      // Text blocks are already covered by the per-token stream_events above,
      // so we don't double-emit them here.
      if (m.type === "assistant") {
        const content = (m.message as { content?: unknown[] } | undefined)?.content;
        if (Array.isArray(content)) {
          for (const block of content) {
            const b = block as {
              type?: string;
              id?: string;
              name?: string;
              input?: Record<string, unknown>;
            };
            if (b.type === "tool_use" && b.id) {
              const tool = b.name || "tool";
              const filePath =
                typeof b.input?.file_path === "string"
                  ? (b.input.file_path as string)
                  : typeof b.input?.path === "string"
                    ? (b.input.path as string)
                    : undefined;
              toolStart.set(b.id, { name: tool, startMs: Date.now(), ...(filePath ? { filePath } : {}) });
              toolNames.add(tool);
              toolCallCount += 1;
              yield {
                kind: "tool.update",
                callID: b.id,
                tool,
                status: "running",
                ...(filePath ? { filePath } : {}),
              };
            }
          }
        }
        continue;
      }

      // User messages with role=user contain tool_result blocks back to us.
      // The SDK fires these once the tool's been invoked and returned.
      if (m.type === "user") {
        const content = (m.message as { content?: unknown[] } | undefined)?.content;
        if (Array.isArray(content)) {
          for (const block of content) {
            const b = block as {
              type?: string;
              tool_use_id?: string;
              is_error?: boolean;
              content?: unknown;
            };
            if (b.type === "tool_result" && b.tool_use_id) {
              const meta = toolStart.get(b.tool_use_id);
              const tool = meta?.name || "tool";
              const durationMs = meta ? Date.now() - meta.startMs : undefined;
              const errored = Boolean(b.is_error);
              const errStr =
                errored && typeof b.content === "string"
                  ? b.content.slice(0, 200)
                  : undefined;
              yield {
                kind: "tool.update",
                callID: b.tool_use_id,
                tool,
                status: errored ? "error" : "completed",
                ...(meta?.filePath ? { filePath: meta.filePath } : {}),
                ...(typeof durationMs === "number" ? { durationMs } : {}),
                ...(errStr ? { error: errStr } : {}),
              };
            }
          }
        }
        continue;
      }

      // Terminal result message — same shape runClaudeCode reads.
      if (m.type === "result") {
        if (m.subtype === "success") {
          const r = m as unknown as {
            result: string;
            session_id: string;
            num_turns: number;
            stop_reason: string | null;
            duration_ms: number;
            total_cost_usd: number;
            usage: {
              input_tokens: number;
              output_tokens: number;
              cache_read_input_tokens?: number | null;
              cache_creation_input_tokens?: number | null;
            };
          };
          yield {
            kind: "done",
            result: {
              ok: true,
              answer: r.result || finalText,
              sessionId: r.session_id,
              model,
              numTurns: r.num_turns,
              tokensIn: r.usage?.input_tokens,
              tokensOut: r.usage?.output_tokens,
              cacheReadTokens: r.usage?.cache_read_input_tokens ?? undefined,
              cacheWriteTokens: r.usage?.cache_creation_input_tokens ?? undefined,
              costUsd: r.total_cost_usd,
              durationMs: r.duration_ms,
              stopReason: r.stop_reason,
              toolCalls: toolCallCount,
              toolNames: Array.from(toolNames),
            },
          };
        } else {
          const r = m as unknown as {
            subtype: string;
            errors?: string[];
            session_id?: string;
            num_turns: number;
            duration_ms: number;
            total_cost_usd: number;
            usage: { input_tokens: number; output_tokens: number };
          };
          yield {
            kind: "done",
            result: {
              ok: false,
              error: r.errors?.join("; ") || `claude code finished with ${r.subtype}`,
              answer: finalText || undefined,
              sessionId: r.session_id,
              model,
              numTurns: r.num_turns,
              tokensIn: r.usage?.input_tokens,
              tokensOut: r.usage?.output_tokens,
              costUsd: r.total_cost_usd,
              durationMs: r.duration_ms,
              toolCalls: toolCallCount,
              toolNames: Array.from(toolNames),
            },
          };
        }
        return;
      }
    }

    yield {
      kind: "done",
      result: {
        ok: false,
        error: "claude code stream ended without a result message",
        answer: finalText || undefined,
        model,
        toolCalls: toolCallCount,
        toolNames: Array.from(toolNames),
      },
    };
  } catch (err) {
    const e = err as Error;
    const name = e?.name || "Error";
    const aborted = name === "AbortError" || /aborted|timeout/i.test(e?.message || "");
    yield {
      kind: "done",
      result: {
        ok: false,
        error: aborted
          ? `aborted: ${e?.message || "timeout or cancel"}`
          : `${name}: ${e?.message || String(e)}`,
        answer: finalText || undefined,
        model,
        toolCalls: toolCallCount,
        toolNames: Array.from(toolNames),
      },
    };
  } finally {
    clearTimer();
    try {
      q.close();
    } catch {
      /* ignore */
    }
  }
}
