/**
 * Tiny client for tech-decomposition's `/v1/adapters/{name}/decompose` endpoint.
 *
 * Only one call surface is needed here — kept inline so the bridge has no
 * dependency on the tech-decomposition package itself.
 */

import type { Config } from "./config.js";

export type Subtask = {
  title: string;
  description: string;
  repo: string;
  files: string[];
  file_links: string[];
  acceptance_criteria: string[];
  estimated_complexity: "small" | "medium" | "large" | "unknown";
};

export type Decomposition = {
  query: string;
  overview: string;
  affected_repos: string[];
  risks: string[];
  open_questions: string[];
  subtasks: Subtask[];
};

export type DecomposeResult = {
  run_id: string;
  result: {
    adapter: string;
    decomposition: Decomposition;
    markdown: string;
    metrics: {
      duration_ms?: number;
      tokens_in?: number;
      tokens_out?: number;
      cost_usd?: number;
      model?: string;
    };
  };
};

export class TechDecompClient {
  constructor(private readonly cfg: Config["techDecomp"]) {}

  async decompose(opts: { query: string }): Promise<DecomposeResult> {
    const url = `${this.cfg.url}/v1/adapters/${encodeURIComponent(this.cfg.adapter)}/decompose`;
    const ac = new AbortController();
    const t = setTimeout(() => ac.abort(), this.cfg.timeoutSeconds * 1000);
    try {
      const r = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: opts.query,
          ...(this.cfg.repos ? { repos: this.cfg.repos } : {}),
        }),
        signal: ac.signal,
      });
      if (!r.ok) {
        const body = await r.text();
        throw new Error(
          `tech-decomp POST ${url} → ${r.status}: ${body.slice(0, 500)}`,
        );
      }
      return (await r.json()) as DecomposeResult;
    } finally {
      clearTimeout(t);
    }
  }
}
