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
