// Thin wrapper around the FastAPI surface. Same shapes as
// tech_decomposition.api.{AskRequest, AskResponse, RunListItem, RunDetail}.

export type OutputFormat = "markdown" | "html" | "text";
export type AskEngine = "sourcebot" | "local";

export interface AskRequest {
  question: string;
  engine?: AskEngine;
  max_steps?: number | null;
  structure_responses?: boolean;
  format?: OutputFormat;
  write_file?: boolean;
  repos?: string[] | null;
}

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
  created_at: string;
  completed_at?: string | null;
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
    let detail = "";
    try {
      detail = JSON.stringify(await res.json());
    } catch {
      detail = await res.text();
    }
    throw new Error(`${res.status} ${path}: ${detail.slice(0, 400)}`);
  }
  return (await res.json()) as T;
}

export const api = {
  health: () => jsonReq<{ status: string }>("/health"),

  ask: (req: AskRequest) =>
    jsonReq<AskResponse>("/ask", { method: "POST", body: JSON.stringify(req) }),

  listRuns: (params: { mode?: string; limit?: number; since?: string; engine?: string } = {}) => {
    const q = new URLSearchParams();
    for (const [k, v] of Object.entries(params)) {
      if (v !== undefined && v !== null && v !== "") q.set(k, String(v));
    }
    const suffix = q.toString() ? `?${q}` : "";
    return jsonReq<RunListItem[]>(`/runs${suffix}`);
  },

  getRun: (id: string) => jsonReq<RunDetail>(`/runs/${encodeURIComponent(id)}`),

  replayRun: (id: string) =>
    jsonReq<{ replayed_from: string; new_run_id: string }>(
      `/runs/${encodeURIComponent(id)}/replay`,
      { method: "POST" },
    ),
};
