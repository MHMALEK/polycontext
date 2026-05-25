// Thin wrapper around the FastAPI surface.
//
// Every code-agent request goes through ``/v1/adapters/{name}/...`` — there
// is no longer a generic ``/ask`` endpoint. ``AskResponse`` is the flattened
// shape the UI uses to render an adapter result; we keep it here (rather
// than the raw AdapterAskResult) because the existing answer view binds to
// these fields.

export interface AskResponse {
  engine: string;
  answer: string;
  citations: Array<Record<string, unknown>>;
  model?: string | null;
  transport?: string | null;
  wall_seconds?: number | null;
  input_tokens?: number | null;
  output_tokens?: number | null;
  cost_usd?: number | null;
  markdown_path?: string | null;
  run_id: string;
}

export interface RunListItem {
  id: string;
  mode: string;
  status: string;
  engine?: string | null;
  model?: string | null;
  total_seconds?: number | null;
  total_cost_usd?: number | null;
  input_tokens?: number | null;
  output_tokens?: number | null;
  input_preview?: string | null;
  output_format?: string | null;
  /** Canonical chat id (first turn's run id); may equal `id` on the first message. */
  thread_id?: string | null;
  created_at: string;
  completed_at?: string | null;
  current_stage?: string | null;
  stages_done?: string[];
}

export interface RunDetail extends RunListItem {
  answer?: string | null;
  citations: Array<Record<string, unknown>>;
  payload: Record<string, unknown>;
  input_ref: unknown;
  error?: string | null;
}

// Hit our own origin (FastAPI mounts UI at /ui in prod; vite proxies in dev).
const BASE = "";

async function jsonReq<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    ...init,
    headers: { "content-type": "application/json", ...(init?.headers || {}) },
  });
  if (!res.ok) {
    const body = await res.text();
    let detail = body;
    try {
      detail = JSON.stringify(JSON.parse(body));
    } catch {
      // not JSON — fall through with the raw text
    }
    throw new Error(`${res.status} ${path}: ${detail.slice(0, 400)}`);
  }
  return (await res.json()) as T;
}

// ---- Adapter bake-off shapes (mirror src/tech_decomposition/adapters/base.py)

export type Capability = "ask" | "decompose";

export interface AdapterInfo {
  name: string;
  installed: boolean;
  capabilities: Capability[];
  description: string;
  health: { ok: boolean; reason?: string };
}

export interface AdapterMetrics {
  duration_ms?: number;
  tokens_in?: number | null;
  tokens_out?: number | null;
  cost_usd?: number | null;
  model?: string | null;
  tool_calls?: number;
  extra?: Record<string, unknown>;
}

export interface BakeoffItem {
  adapter: string;
  ok: boolean;
  status?: number;
  error?: string;
  result?: {
    adapter: string;
    answer?: string;
    markdown?: string;
    decomposition?: Record<string, unknown>;
    citations?: Array<Record<string, unknown>>;
    metrics: AdapterMetrics;
  };
}

// Shape returned by POST /v1/adapters/{name}/{ask,decompose}.
export interface AdapterCallResponse {
  run_id: string;
  thread_id: string;
  result: {
    adapter: string;
    answer?: string;
    markdown?: string;
    decomposition?: Record<string, unknown>;
    citations?: Array<Record<string, unknown>>;
    metrics: AdapterMetrics;
    /** Populated only when the ask request set ``grounded: true``. */
    grounding?: GroundedContext;
  };
}

export interface AdapterAskBody {
  query: string;
  thread_id?: string;
  repos?: string[];
  top_k?: number;
  grounded?: boolean;
  /** Per-request model override. Format is adapter-specific —
   * "openrouter/deepseek/deepseek-v3.2" for opencode,
   * "openrouter:deepseek/deepseek-v3.2" for pipeline,
   * "gemini-2.5-pro" for gemini. */
  model?: string;
  /** When false, disable the adapter's tool use so the model answers
   * single-shot from prefetched context. Defaults to true. */
  tools_enabled?: boolean;
  /** Override the global ANSWER_SHAPER_ENABLED setting for this one call.
   * False = skip the rich-card pydantic-AI pass (saves ~1s + a Flash call).
   * Omitted = use server default. */
  shape?: boolean;
}

/** Rich-card structure returned by the pydantic-AI shaper (api/core/answer_shaper.py).
 * When ``run.payload.structured_answer`` is present, the UI renders this
 * shape instead of the raw markdown answer. */
export interface Citation {
  path: string;
  start_line?: number | null;
  end_line?: number | null;
  note?: string | null;
}
export interface Answer {
  summary: string;
  details: string;
  citations: Citation[];
  confidence: "low" | "medium" | "high";
  caveats: string[];
  next_steps: string[];
}
export interface ShapedAnswer {
  answer: Answer;
  shaper_model?: string | null;
  duration_ms?: number;
  tokens_in?: number | null;
  tokens_out?: number | null;
  used_fallback?: boolean;
  fallback_reason?: string | null;
}

/** Curated per-adapter model entry from GET /v1/adapters/{name}/models. */
export interface AdapterModelInfo {
  id: string;
  name: string;
  provider?: string;
  in_per_m_usd?: number;
  out_per_m_usd?: number;
  context_k?: number;
  note?: string;
}

// Grounding — mirrors core/grounding.py types
export interface GroundingSnippet {
  repo: string;
  path: string;
  start_line: number | null;
  end_line: number | null;
  content: string;
  url: string | null;
  language: string | null;
}

export interface GroundingMetrics {
  duration_ms: number;
  snippet_count: number;
  total_chars: number;
  sources: string[];
  sourcebot_files_seen: number;
  error: string | null;
  extracted_terms: string[];
  search_query: string;
  /** "llm" if the Gemini Flash classifier produced the terms; "regex" otherwise. */
  extractor: "llm" | "regex" | string;
  /** Verdict from the LLM classifier when used: "needs_grounding" / "skip" / "". */
  classifier_decision: string;
  /** One-sentence reasoning from the classifier — surface in tooltips/logs. */
  classifier_reason: string;
  classifier_ms: number;
}

export interface GroundedContext {
  snippets: GroundingSnippet[];
  grounding_block: string;
  metrics: GroundingMetrics;
}

// Subtask + Decomposition mirror src/tech_decomposition/models.py Subtask + Decomposition.
export interface Subtask {
  title: string;
  description: string;
  repo: string;
  files: string[];
  file_links: string[];
  acceptance_criteria: string[];
  estimated_complexity: "small" | "medium" | "large" | "unknown";
}

export interface Decomposition {
  query: string;
  overview: string;
  affected_repos: string[];
  risks: string[];
  open_questions: string[];
  subtasks: Subtask[];
  enrichment_model?: string;
  decomposition_model?: string;
}

export interface AdapterDecomposeBody {
  query: string;
  repos?: string[];
  mode?: "cheap" | "deep" | "auto";
}

export interface AdapterDecomposeResponse {
  run_id: string;
  result: {
    adapter: string;
    decomposition: Decomposition;
    markdown: string;
    metrics: AdapterMetrics;
  };
}

/** Stream-protocol events emitted by /v1/adapters/opencode/stream. Mirrors
 * the OpencodeStreamEvent union in agent-node/src/adapters/opencode.ts plus
 * a synthetic `run` envelope the FastAPI bridge adds up front. */
export type OpencodeStreamEvent =
  | { kind: "run"; run_id: string; thread_id: string }
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
  | { kind: "shaped"; shaped: ShapedAnswer }
  | { kind: "done"; result: Record<string, unknown> };

export const api = {
  health: () => jsonReq<{ status: string }>("/health"),

  adapterAsk: (name: string, body: AdapterAskBody) =>
    jsonReq<AdapterCallResponse>(`/v1/adapters/${encodeURIComponent(name)}/ask`, {
      method: "POST",
      body: JSON.stringify(body),
    }),

  /** Streams OpenCode events as the agent runs. Pass `signal` to cancel.
   *
   * Yields the same JSON union the agent-node SSE bridge emits — the UI
   * appends text.delta to a partial answer, renders tool.update as chips,
   * and stores `sessionID` on first `session` event so the Stop button
   * can call ``opencodeAbort``. The terminal event is `{kind:"done"}`.
   *
   * Uses fetch + ReadableStream rather than EventSource because EventSource
   * is GET-only — we POST a JSON body matching AdapterAskBody. */
  async *adapterAskStream(
    body: AdapterAskBody,
    opts: { signal?: AbortSignal } = {},
  ): AsyncGenerator<OpencodeStreamEvent, void, void> {
    const res = await fetch(`${BASE}/v1/adapters/opencode/stream`, {
      method: "POST",
      headers: { "content-type": "application/json", accept: "text/event-stream" },
      body: JSON.stringify(body),
      signal: opts.signal,
    });
    if (!res.ok || !res.body) {
      const text = await res.text();
      throw new Error(`${res.status} /v1/adapters/opencode/stream: ${text.slice(0, 400)}`);
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buf = "";
    try {
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        // SSE frames are separated by a blank line. Each frame has one or
        // more `data: …` lines.
        let idx: number;
        while ((idx = buf.indexOf("\n\n")) >= 0) {
          const frame = buf.slice(0, idx);
          buf = buf.slice(idx + 2);
          const dataLines = frame
            .split("\n")
            .filter((l) => l.startsWith("data:"))
            .map((l) => l.slice(5).trimStart());
          if (!dataLines.length) continue;
          const payload = dataLines.join("\n");
          try {
            yield JSON.parse(payload) as OpencodeStreamEvent;
          } catch {
            // ignore non-JSON keepalives
          }
        }
      }
    } finally {
      try {
        reader.releaseLock();
      } catch {
        /* ignore */
      }
    }
  },

  /** Stop a streaming opencode run. Returns ``{ ok: true }`` on success. */
  opencodeAbort: (sessionID: string, repos?: string[]) =>
    jsonReq<{ ok: boolean; error?: string }>(`/v1/adapters/opencode/abort`, {
      method: "POST",
      body: JSON.stringify({ session_id: sessionID, ...(repos ? { repos } : {}) }),
    }),

  /** Native-model streaming via pydantic-AI's run_stream — works for ANY
   * adapter as long as the request is in a no-tools mode (Direct or
   * Grounded-only). Bypasses each adapter's SDK and goes straight to
   * Gemini/Anthropic/OpenAI/OpenRouter for token-by-token deltas.
   *
   * The adapter name is carried in the body only so the server can pick
   * a sensible default model — the SDK itself isn't used. */
  async *nativeAskStream(
    body: AdapterAskBody & { adapter?: string },
    opts: { signal?: AbortSignal } = {},
  ): AsyncGenerator<OpencodeStreamEvent, void, void> {
    const res = await fetch(`${BASE}/v1/ask/stream`, {
      method: "POST",
      headers: { "content-type": "application/json", accept: "text/event-stream" },
      body: JSON.stringify(body),
      signal: opts.signal,
    });
    if (!res.ok || !res.body) {
      const text = await res.text();
      throw new Error(`${res.status} /v1/ask/stream: ${text.slice(0, 400)}`);
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buf = "";
    try {
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        let idx: number;
        while ((idx = buf.indexOf("\n\n")) >= 0) {
          const frame = buf.slice(0, idx);
          buf = buf.slice(idx + 2);
          const dataLines = frame
            .split("\n")
            .filter((l) => l.startsWith("data:"))
            .map((l) => l.slice(5).trimStart());
          if (!dataLines.length) continue;
          const payload = dataLines.join("\n");
          try {
            yield JSON.parse(payload) as OpencodeStreamEvent;
          } catch {
            /* ignore keepalives / non-JSON */
          }
        }
      }
    } finally {
      try {
        reader.releaseLock();
      } catch {
        /* ignore */
      }
    }
  },

  groundingRetrieve: (body: {
    query: string;
    repos?: string[];
    top_k?: number;
    context_lines?: number;
  }) =>
    jsonReq<GroundedContext>(`/v1/grounding/retrieve`, {
      method: "POST",
      body: JSON.stringify(body),
    }),

  adapterDecompose: (name: string, body: AdapterDecomposeBody) =>
    jsonReq<AdapterDecomposeResponse>(`/v1/adapters/${encodeURIComponent(name)}/decompose`, {
      method: "POST",
      body: JSON.stringify(body),
    }),

  listRuns: (
    params: {
      mode?: string;
      limit?: number;
      since?: string;
      engine?: string;
      status?: string;
      per_thread?: boolean;
      thread_id?: string;
      order?: "asc" | "desc";
    } = {},
  ) => {
    const q = new URLSearchParams();
    for (const [k, v] of Object.entries(params)) {
      if (v === undefined || v === null) continue;
      if (typeof v === "string" && v === "") continue;
      q.set(k, String(v));
    }
    const suffix = q.toString() ? `?${q}` : "";
    return jsonReq<RunListItem[]>(`/runs${suffix}`);
  },

  listThreadRuns: (threadId: string, params: { limit?: number } = {}) => {
    const q = new URLSearchParams();
    for (const [k, v] of Object.entries(params)) {
      if (v === undefined || v === null) continue;
      if (typeof v === "string" && v === "") continue;
      q.set(k, String(v));
    }
    const suffix = q.toString() ? `?${q}` : "";
    return jsonReq<RunDetail[]>(`/runs/thread/${encodeURIComponent(threadId)}${suffix}`);
  },

  getRun: (id: string) => jsonReq<RunDetail>(`/runs/${encodeURIComponent(id)}`),

  replayRun: (id: string) =>
    jsonReq<{ replayed_from: string; new_run_id: string }>(
      `/runs/${encodeURIComponent(id)}/replay`,
      { method: "POST" },
    ),

  // --- adapter bake-off ---
  listAdapters: () => jsonReq<{ adapters: AdapterInfo[] }>("/v1/adapters"),

  /** Combined model catalog per adapter — curated + live OpenRouter list.
   * ``openrouter_live`` carries ~350 OpenRouter models fetched live (cached
   * 1h server-side), deduped against the curated section. Pass
   * ``include_openrouter=false`` in a query string to skip the live fetch. */
  listAdapterModels: (name: string) =>
    jsonReq<{
      adapter: string;
      models: AdapterModelInfo[];
      openrouter_live: AdapterModelInfo[];
    }>(`/v1/adapters/${encodeURIComponent(name)}/models`),

  bakeoff: (
    job: "ask" | "decompose",
    body: {
      adapters: string[];
      ask?: { query: string; repos?: string[]; top_k?: number };
      decompose?: {
        query?: string;
        repos?: string[];
        mode?: "cheap" | "deep" | "auto";
      };
    },
  ) =>
    jsonReq<{ job: string; results: BakeoffItem[] }>(
      `/v1/bakeoff/${job}`,
      { method: "POST", body: JSON.stringify(body) },
    ),
};
