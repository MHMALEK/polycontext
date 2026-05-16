/** Sourcebot Q&A — POST /api/chat/blocking only (no MCP fallback in agent-node). */

export type SourcebotAskBody = {
  sourcebotUrl: string;
  sourcebotApiKey: string;
  question: string;
  repos?: string[];
  maxSteps?: number;
  timeoutSec?: number;
};

function xKey(apiKey: string): string {
  return apiKey.startsWith("sourcebot-") ? apiKey : `sourcebot-${apiKey}`;
}

export async function askSourcebotBlocking(body: SourcebotAskBody): Promise<{
  ok: boolean;
  answer?: string;
  error?: string;
  model?: string;
  chatId?: string;
  chatUrl?: string;
  wallSeconds?: number;
}> {
  const base = body.sourcebotUrl.replace(/\/+$/, "");
  const url = `${base}/api/chat/blocking`;
  const headers: Record<string, string> = {
    "X-Sourcebot-Api-Key": xKey(body.sourcebotApiKey),
    "Content-Type": "application/json",
  };
  const payload: Record<string, unknown> = { query: body.question.trim() };
  if (body.repos?.length) payload.repos = body.repos;
  if (body.maxSteps != null) {
    if (body.maxSteps < 1 || body.maxSteps > 50) {
      return { ok: false, error: "maxSteps must be between 1 and 50" };
    }
    payload.maxSteps = body.maxSteps;
  }

  const t0 = Date.now();
  const timeoutMs = (body.timeoutSec ?? 300) * 1000;
  const ac = new AbortController();
  const timer = setTimeout(() => ac.abort(), timeoutMs);

  try {
    const resp = await fetch(url, {
      method: "POST",
      headers,
      body: JSON.stringify(payload),
      signal: ac.signal,
    });
    clearTimeout(timer);
    const wallSeconds = Math.round((Date.now() - t0) / 100) / 10;
    if (!resp.ok) {
      const detail = (await resp.text()).slice(0, 500);
      return { ok: false, error: `HTTP ${resp.status}: ${detail}` };
    }
    const j = (await resp.json()) as {
      answer?: string;
      languageModel?: { model?: string; displayName?: string };
      chatId?: string;
      chatUrl?: string;
    };
    const answer = (j.answer || "").trim();
    if (!answer) return { ok: false, error: "empty answer from /api/chat/blocking" };
    const lm = j.languageModel || {};
    return {
      ok: true,
      answer,
      model: (lm.model as string) || (lm.displayName as string),
      chatId: j.chatId,
      chatUrl: j.chatUrl,
      wallSeconds,
    };
  } catch (e) {
    clearTimeout(timer);
    const err = e as Error;
    if (err.name === "AbortError") {
      return { ok: false, error: `Sourcebot request timed out after ${body.timeoutSec ?? 300}s` };
    }
    return { ok: false, error: err?.message || String(e) };
  }
}
