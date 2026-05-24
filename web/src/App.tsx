import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { api } from "./api";
import type { AdapterInfo, AdapterModelInfo, RunDetail, RunListItem } from "./api";
import { Decompose } from "./Decompose";

type View = "ask" | "decompose";

/** Prefix for a synthetic sidebar row shown until /runs lists the real thread. */
const PENDING_SIDEBAR_PREFIX = "__pending__:";

function phantomHistoryRow(p: {
  id: string;
  preview: string;
  adapter: string;
}): RunListItem {
  const now = new Date().toISOString();
  return {
    id: p.id,
    thread_id: p.id,
    mode: "ask",
    status: "running",
    engine: `${p.adapter}:ask`,
    input_preview: p.preview.slice(0, 160),
    created_at: now,
  };
}

// ---------------------------------------------------------------------------
// Types & utilities
// ---------------------------------------------------------------------------

type LiveProgress = {
  startedAt: number;
  elapsed: number;
  currentStage: string | null;
  stagesDone: string[];
};

type AskFormState = {
  question: string;
  // Name of the adapter the question is routed to via POST /v1/adapters/
  // {adapter}/ask. Defaulted to ``DEFAULT_ADAPTER`` and reconciled against
  // the live adapter list once it loads — see the useEffect below.
  adapter: string;
  // Per-request model override (empty string = adapter default). Populated
  // from GET /v1/adapters/{adapter}/models on adapter change.
  model: string;
  // Operating mode. Maps to (grounded, tools_enabled) on the wire:
  //   "hybrid"        → grounded=true,  tools_enabled=true   (default)
  //   "grounded_only" → grounded=true,  tools_enabled=false  (snippets only)
  //   "agent_only"    → grounded=false, tools_enabled=true   (no prefetch)
  //   "raw"           → grounded=false, tools_enabled=false  (single-shot)
  mode: "hybrid" | "grounded_only" | "agent_only" | "raw";
};

type AskMode = AskFormState["mode"];

const ASK_MODE_LABELS: Record<AskMode, string> = {
  hybrid: "Hybrid (grounded + tools)",
  grounded_only: "Grounded only",
  agent_only: "Agent only (no grounding)",
  raw: "Single-shot (no grounding, no tools)",
};

// Short labels for the segmented control — needs to fit horizontally in
// 4 pills, so we drop the parentheticals and use terse names.
function askModeShortLabel(m: AskMode): string {
  switch (m) {
    case "hybrid": return "Hybrid";
    case "grounded_only": return "Grounded";
    case "agent_only": return "Agent";
    case "raw": return "Direct";
  }
}

const ASK_MODE_DESCRIPTIONS: Record<AskMode, string> = {
  hybrid: "Prefetch Sourcebot+Serena snippets AND let the adapter call its own tools if it wants to. Default.",
  grounded_only: "Prefetch snippets and force the adapter to answer single-shot from them (no read/grep/glob).",
  agent_only: "Skip prefetch; the agent navigates the repo from scratch using its own tools.",
  raw: "No grounding, no tools — the raw query goes to the model directly.",
};

function askModeToWire(mode: AskMode): { grounded: boolean; tools_enabled: boolean } {
  switch (mode) {
    case "hybrid": return { grounded: true, tools_enabled: true };
    case "grounded_only": return { grounded: true, tools_enabled: false };
    case "agent_only": return { grounded: false, tools_enabled: true };
    case "raw": return { grounded: false, tools_enabled: false };
  }
}

// Initial dropdown selection. Overridden at mount if this adapter isn't
// installed or isn't healthy — see the adapter list useEffect.
const DEFAULT_ADAPTER = "cursor";

const ADAPTER_PRIORITY: Record<string, number> = {
  pipeline: 0,    // our custom retrieve-first flow — promote to top
  opencode: 1,    // best cost-quality agentic option we've measured
  cursor: 2,
  gemini: 3,
  claude_code: 4,
  cline_sdk: 5,
  sourcebot: 6,
  openai_agents: 7,
};

/**
 * Friendlier UI labels for the raw adapter registry keys. The registry key
 * (e.g. ``pipeline``) is what the API and eval harness use — don't rename
 * it. The UI just shows a nicer version. Unknown keys fall back to the
 * raw name.
 */
const ADAPTER_DISPLAY_NAMES: Record<string, string> = {
  pipeline: "polycontext",
  opencode: "OpenCode",
  gemini: "Gemini",
  cursor: "Cursor",
  claude_code: "Claude Code",
  cline_sdk: "Cline",
  openai_agents: "OpenAI Agents",
  sourcebot: "Sourcebot",
};

function displayAdapterName(name: string): string {
  return ADAPTER_DISPLAY_NAMES[name] ?? name;
}

type DisplayedRun = {
  question: string;
  answer: string;
  engine?: string | null;
  model?: string | null;
  wall_seconds?: number | null;
  cost_usd?: number | null;
  input_tokens?: number | null;
  output_tokens?: number | null;
  citations: Array<Record<string, unknown>>;
};

const INITIAL_FORM: AskFormState = {
  question: "",
  adapter: DEFAULT_ADAPTER,
  model: "",
  mode: "hybrid",
};

const SUGGESTED_PROMPTS = [
  "Where is the Suppliers page rendered?",
  "How does the authentication flow work?",
  "What endpoints does the API expose?",
];

// Friendly labels for the raw "stage:strategy" identifier the backend emits.
function prettyStage(label: string | null | undefined): string {
  if (!label) return "Working…";
  const [stage, strategy] = label.split(":");
  switch (stage) {
    case "input":
      return "Loading input";
    case "enrich":
      return strategy && strategy !== "(none)"
        ? `Enriching question (${strategy})`
        : "Preparing question";
    case "engine": {
      const s = strategy?.toLowerCase() ?? "";
      if (s.includes("sourcebot")) return "Searching code & reasoning";
      if (s.includes("structured")) return "Calling LLM";
      if (s.includes("decompose")) return "Decomposing ticket";
      return strategy ? `Calling engine (${strategy})` : "Calling engine";
    }
    case "render":
      return "Rendering output";
    case "sink":
      return "Writing output";
    default:
      return label;
  }
}

const PIPELINE_STAGES: { key: string; label: string }[] = [
  { key: "input", label: "Load input" },
  { key: "enrich", label: "Prepare question" },
  { key: "engine", label: "Reason & answer" },
  { key: "render", label: "Render output" },
];

function stageState(
  stageKey: string,
  done: string[],
  current: string | null,
): "done" | "active" | "pending" {
  if (done.some((d) => d.startsWith(stageKey + ":"))) return "done";
  if (current?.startsWith(stageKey + ":")) return "active";
  return "pending";
}

function formatTimeAgo(iso: string): string {
  const d = new Date(iso);
  const diff = (Date.now() - d.getTime()) / 1000;
  if (diff < 60) return `${Math.max(1, Math.round(diff))}s ago`;
  if (diff < 3600) return `${Math.round(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.round(diff / 3600)}h ago`;
  return d.toLocaleDateString();
}

function formatCost(n: number | null | undefined): string {
  if (n == null) return "—";
  if (n < 0.001) return "<$0.001";
  return `$${n.toFixed(n < 0.1 ? 4 : 3)}`;
}

function formatWall(n: number | null | undefined): string {
  if (n == null) return "—";
  return n < 60 ? `${n.toFixed(1)}s` : `${Math.floor(n / 60)}m ${Math.round(n % 60)}s`;
}

function startOfDay(iso: string): number {
  const d = new Date(iso);
  d.setHours(0, 0, 0, 0);
  return d.getTime();
}

function groupHistoryByDay(rows: RunListItem[]): { label: string; items: RunListItem[] }[] {
  const today = startOfDay(new Date().toISOString());
  const yesterday = today - 86400_000;
  const buckets: Record<string, RunListItem[]> = { Today: [], Yesterday: [], Earlier: [] };
  for (const r of rows) {
    const d = startOfDay(r.created_at);
    if (d === today) buckets.Today.push(r);
    else if (d === yesterday) buckets.Yesterday.push(r);
    else buckets.Earlier.push(r);
  }
  return (["Today", "Yesterday", "Earlier"] as const)
    .map((k) => ({ label: k, items: buckets[k] }))
    .filter((g) => g.items.length > 0);
}

function extractQuestion(detail: RunDetail | null): string {
  if (!detail) return "";
  const ref = detail.input_ref;
  if (typeof ref === "string") return ref;
  if (ref && typeof ref === "object" && "query" in ref) {
    const q = (ref as { query: unknown }).query;
    if (typeof q === "string" && q.trim()) return q;
  }
  return (detail.input_preview ?? "").trim();
}

function historyTitle(run: RunListItem): string {
  const t = (run.input_preview ?? "").trim();
  return t || "Untitled question";
}

function engineAdapterLabel(engine: string | undefined | null): string | null {
  if (!engine) return null;
  const i = engine.indexOf(":");
  const name = i > 0 ? engine.slice(0, i) : engine;
  return name || null;
}

/** Canonical chat id: shared by all turns in a session (first turn's run id). */
function threadKey(run: RunListItem | RunDetail): string {
  const t = "thread_id" in run ? run.thread_id : null;
  return (t && String(t).trim()) || run.id;
}

// ---------------------------------------------------------------------------
// Small components
// ---------------------------------------------------------------------------

function RoleLabel({ role, align }: { role: string; align: "left" | "right" }) {
  return (
    <p
      className={`text-[10px] font-semibold uppercase tracking-[0.16em] text-base-content/65 mb-1 ${
        align === "right" ? "text-right pr-1" : "text-left pl-1"
      }`}
    >
      {role}
    </p>
  );
}

function HistoryItem({
  run,
  selected,
  onSelect,
  onReplay,
}: {
  run: RunListItem;
  selected: boolean;
  onSelect: (id: string) => void;
  onReplay: (id: string) => void;
}) {
  const adapter = engineAdapterLabel(run.engine);
  const isPendingSidebar = run.id.startsWith(PENDING_SIDEBAR_PREFIX);
  return (
    <button
      type="button"
      onClick={() => onSelect(threadKey(run))}
      className={`group relative w-full text-left rounded-lg px-3 py-2.5 transition-colors outline-none focus-visible:ring-1 focus-visible:ring-primary/40 ${
        selected
          ? "bg-base-200 border-hairline"
          : "bg-transparent hover:bg-base-200/60 border border-transparent"
      }`}
    >
      <div className="flex items-start gap-2.5">
        {/* Status dot — solid green if completed, pulsing if running,
            muted otherwise. Replaces the larger ringed bullet. */}
        <span
          className={`status-dot mt-[7px] shrink-0 ${
            run.status === "running" ? "pulse" : run.status === "completed" ? "" : "muted"
          }`}
          aria-hidden
        />
        <div className="min-w-0 flex-1">
          <p className="text-[12.5px] leading-snug line-clamp-2 text-base-content/95 font-normal tracking-tight">
            {historyTitle(run)}
          </p>
          <div className="mt-1 flex flex-wrap items-center gap-x-1.5 gap-y-0.5">
            <span className="font-mono text-[10px] tracking-wide text-base-content/45 tabular-nums">
              {formatTimeAgo(run.created_at)}
            </span>
            {adapter && (
              <>
                <span aria-hidden className="text-base-content/25 text-[10px]">·</span>
                <span className="font-mono text-[10px] uppercase tracking-wider text-base-content/55">
                  {displayAdapterName(adapter)}
                </span>
              </>
            )}
            {run.total_seconds != null && (
              <>
                <span aria-hidden className="text-base-content/25 text-[10px]">·</span>
                <span className="font-mono text-[10px] text-base-content/45 tabular-nums">
                  {formatWall(run.total_seconds)}
                </span>
              </>
            )}
            {run.total_cost_usd != null && (
              <>
                <span aria-hidden className="text-base-content/25 text-[10px]">·</span>
                <span className="font-mono text-[10px] text-accent/85 tabular-nums">
                  {formatCost(run.total_cost_usd)}
                </span>
              </>
            )}
          </div>
        </div>
        {!isPendingSidebar && (
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            onReplay(run.id);
          }}
          className="btn btn-ghost btn-xs btn-square shrink-0 opacity-0 group-hover:opacity-100 focus:opacity-100 transition-opacity"
          title="Re-run this question"
          aria-label="Re-run this question"
        >
          ↻
        </button>
        )}
      </div>
    </button>
  );
}

function ProgressTimeline({ progress }: { progress: LiveProgress }) {
  return (
    <div className="rounded-2xl border border-base-300/80 bg-gradient-to-br from-base-100 to-base-200/40 p-5 shadow-inner ring-1 ring-base-content/[0.1] animate-msg-enter">
      <div className="flex items-center gap-3 mb-5">
        <span className="loading loading-spinner loading-sm text-primary" />
        <span className="text-sm font-medium tracking-tight">{prettyStage(progress.currentStage)}…</span>
        <span className="ml-auto font-mono-ui text-xs text-base-content/75 tabular-nums">
          {progress.elapsed.toFixed(1)}s
        </span>
      </div>
      <ol className="flex flex-col gap-2.5">
        {PIPELINE_STAGES.map((s) => {
          const st = stageState(s.key, progress.stagesDone, progress.currentStage);
          return (
            <li key={s.key} className="flex items-center gap-3 text-sm">
              <span
                className={`inline-flex items-center justify-center w-6 h-6 rounded-full shrink-0 text-[11px] font-semibold transition-colors ${
                  st === "done"
                    ? "bg-success/20 text-success"
                    : st === "active"
                      ? "bg-primary/20 text-primary shadow-sm"
                      : "bg-base-300/65 text-base-content/68"
                }`}
              >
                {st === "done" ? "✓" : st === "active" ? "●" : ""}
              </span>
              <span
                className={
                  st === "pending"
                    ? "text-base-content/66"
                    : st === "active"
                      ? "text-base-content font-medium"
                      : "text-base-content/78"
                }
              >
                {s.label}
              </span>
            </li>
          );
        })}
      </ol>
      <p className="mt-5 text-[11px] leading-relaxed text-base-content/70 border-t border-base-300/70 pt-4">
        Adapter calls often take 10–60s while the model searches and drafts an answer.
      </p>
    </div>
  );
}

function QuestionBubble({ text }: { text: string }) {
  if (!text) return null;
  return (
    <div className="flex flex-col items-end animate-msg-enter">
      <RoleLabel role="You" align="right" />
      <div className="max-w-[min(100%,40rem)] rounded-3xl rounded-tr-lg bg-gradient-to-br from-primary to-primary/90 text-primary-content px-4 py-3.5 text-[15px] leading-snug whitespace-pre-wrap shadow-lg shadow-primary/20 ring-1 ring-primary-content/10">
        {text}
      </div>
    </div>
  );
}

function AnswerBody({ text }: { text: string }) {
  return (
    <div className="conversation-prose">
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{text}</ReactMarkdown>
    </div>
  );
}

function Citations({ citations }: { citations: Array<Record<string, unknown>> }) {
  if (!citations.length) return null;
  return (
    <details className="mt-6 rounded-xl border border-base-300/80 bg-base-200/45 shadow-inner ring-1 ring-base-content/[0.08] group open:bg-base-200/55">
      <summary className="cursor-pointer select-none list-none px-4 py-3 text-sm font-medium text-base-content/92 flex items-center gap-3 hover:bg-base-200/40 rounded-xl transition-colors [&::-webkit-details-marker]:hidden">
        <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-base-100/80 text-[11px] font-semibold tabular-nums text-base-content/78 ring-1 ring-base-300/60">
          {citations.length}
        </span>
        <span className="flex-1 min-w-0">
          <span className="block tracking-tight">Sources</span>
          <span className="block text-[11px] font-normal text-base-content/70 mt-0.5">
            Citations from the indexed codebase
          </span>
        </span>
        <span className="text-base-content/60 text-xs transition-transform group-open:rotate-90">›</span>
      </summary>
      <ul className="px-3 pb-3 pt-0 flex flex-col gap-2.5 text-sm border-t border-base-300/70">
        {citations.map((c, i) => {
          const repo = (c.repo as string) || "";
          const path = (c.path as string) || "";
          const ls = c.line_start as number | undefined;
          const le = c.line_end as number | undefined;
          const snippet = (c.snippet as string) || "";
          const range = ls != null ? (le != null && le !== ls ? `:${ls}-${le}` : `:${ls}`) : "";
          return (
            <li
              key={i}
              className="mt-3 first:mt-3 rounded-lg border border-base-300/75 bg-base-100/85 pl-3 pr-3 py-2.5"
            >
              <code className="font-mono-ui text-[11px] leading-relaxed text-base-content/88 block break-all">
                {repo ? `${repo} / ` : ""}
                {path}
                {range}
              </code>
              {snippet && (
                <pre className="mt-2 whitespace-pre-wrap text-[11px] leading-relaxed text-base-content/83 font-mono-ui bg-base-300/50 rounded-md px-2.5 py-2 overflow-x-auto border border-base-300/65">
                  {snippet}
                </pre>
              )}
            </li>
          );
        })}
      </ul>
    </details>
  );
}

function MetaStrip({ run }: { run: DisplayedRun }) {
  // Render the per-run telemetry as typed chips — engine + model are info,
  // cost is amber (cautionary), tokens neutral. This is exactly the kind
  // of metric strip you'd see at the top of a Datadog widget, not the
  // generic gray pills daisyUI gives you.
  type Variant = "default" | "info" | "accent";
  const items: Array<{ label: string; value: string; variant?: Variant }> = [];
  if (run.engine) items.push({ label: "engine", value: run.engine, variant: "info" });
  if (run.model) items.push({ label: "model", value: run.model });
  if (run.wall_seconds != null)
    items.push({ label: "wall", value: formatWall(run.wall_seconds) });
  if (run.cost_usd != null)
    items.push({ label: "cost", value: formatCost(run.cost_usd), variant: "accent" });
  if (run.input_tokens != null || run.output_tokens != null) {
    items.push({
      label: "tokens",
      value: `${run.input_tokens ?? 0}→${run.output_tokens ?? 0}`,
    });
  }
  if (!items.length) return null;
  return (
    <div className="flex flex-wrap items-center gap-1.5">
      {items.map((it) => (
        <span
          key={it.label}
          className={`chip ${it.variant === "info" ? "chip-info" : it.variant === "accent" ? "chip-accent" : ""}`}
        >
          <span className="chip-label">{it.label}</span>
          <span className="chip-value">{it.value}</span>
        </span>
      ))}
    </div>
  );
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      type="button"
      onClick={async () => {
        try {
          await navigator.clipboard.writeText(text);
          setCopied(true);
          window.setTimeout(() => setCopied(false), 1500);
        } catch {
          /* clipboard blocked — silently ignore */
        }
      }}
      className={`inline-flex items-center gap-1.5 rounded-lg border px-2.5 py-1 text-[11px] font-medium transition-colors ${
        copied
          ? "border-success/35 bg-success/10 text-success"
          : "border-base-300/70 bg-base-100/80 text-base-content/85 hover:bg-base-100 hover:border-primary/25 hover:text-base-content"
      }`}
      title="Copy answer"
    >
      <span className="font-mono-ui text-xs opacity-80">{copied ? "✓" : "⎘"}</span>
      <span>{copied ? "Copied" : "Copy"}</span>
    </button>
  );
}

function turnToDisplayed(turn: RunDetail): DisplayedRun {
  return {
    question: extractQuestion(turn),
    answer: turn.answer ?? "",
    engine: turn.engine,
    model: turn.model,
    wall_seconds: turn.total_seconds,
    cost_usd: turn.total_cost_usd,
    input_tokens: turn.input_tokens,
    output_tokens: turn.output_tokens,
    citations: (turn.citations ?? []) as Array<Record<string, unknown>>,
  };
}

type RawGrounding = {
  snippets?: Array<{
    repo?: string;
    path?: string;
    start_line?: number | null;
    end_line?: number | null;
    content?: string;
    url?: string | null;
    language?: string | null;
  }>;
  metrics?: {
    duration_ms?: number;
    snippet_count?: number;
    total_chars?: number;
    sources?: string[];
    sourcebot_files_seen?: number;
    error?: string | null;
  };
};

function extractGrounding(payload: Record<string, unknown> | undefined | null): RawGrounding | null {
  if (!payload) return null;
  const g = (payload as { grounding?: unknown }).grounding;
  if (!g || typeof g !== "object") return null;
  return g as RawGrounding;
}

function GroundingPanel({ grounding }: { grounding: RawGrounding }) {
  const m = grounding.metrics ?? {};
  const snippets = grounding.snippets ?? [];
  const ms = m.duration_ms ?? 0;
  const dur = ms < 1000 ? `${ms}ms` : `${(ms / 1000).toFixed(1)}s`;
  if (m.error) {
    return (
      <details className="mt-4 rounded-lg border border-warning/40 bg-warning/5">
        <summary className="cursor-pointer select-none px-4 py-2 text-xs font-medium text-warning flex items-center gap-2">
          Grounding failed: <code className="font-mono">{m.error}</code>
        </summary>
      </details>
    );
  }
  return (
    <details className="mt-4 rounded-lg border border-base-300 bg-base-100/70">
      <summary className="cursor-pointer select-none px-4 py-2 text-xs font-medium flex flex-wrap items-center gap-x-3 gap-y-1">
        <span className="badge badge-sm badge-outline">grounded</span>
        <span className="text-base-content/70">
          {snippets.length} snippet{snippets.length === 1 ? "" : "s"}
        </span>
        <span className="text-base-content/50">·</span>
        <span className="text-base-content/70">{(m.total_chars ?? 0).toLocaleString()} chars</span>
        <span className="text-base-content/50">·</span>
        <span className="text-base-content/70">+{dur}</span>
        {m.sources && m.sources.length > 0 && (
          <>
            <span className="text-base-content/50">·</span>
            <span className="font-mono text-base-content/65">{m.sources.join(", ")}</span>
          </>
        )}
      </summary>
      <div className="px-4 pb-4 pt-1 flex flex-col gap-3">
        {snippets.length === 0 ? (
          <p className="text-xs text-base-content/55 italic">No snippets returned.</p>
        ) : (
          snippets.map((s, i) => {
            const label = s.repo ? `${s.repo}/${s.path ?? ""}` : s.path ?? "";
            const lineSuffix =
              s.start_line && s.end_line && s.end_line !== s.start_line
                ? `:L${s.start_line}-L${s.end_line}`
                : s.start_line
                  ? `:L${s.start_line}`
                  : "";
            return (
              <div key={i} className="rounded border border-base-300/70 bg-base-200/30">
                <div className="px-3 py-2 flex items-center justify-between gap-2 text-[11px]">
                  {s.url ? (
                    <a
                      href={s.url}
                      target="_blank"
                      rel="noreferrer"
                      className="link link-hover font-mono truncate"
                    >
                      {label}{lineSuffix}
                    </a>
                  ) : (
                    <code className="font-mono truncate text-base-content/85">{label}{lineSuffix}</code>
                  )}
                  {s.language && (
                    <span className="badge badge-xs badge-ghost">{s.language}</span>
                  )}
                </div>
                <pre className="px-3 pb-3 pt-0 text-[11px] font-mono whitespace-pre overflow-x-auto leading-snug">
                  {s.content ?? ""}
                </pre>
              </div>
            );
          })
        )}
      </div>
    </details>
  );
}

/**
 * TelemetryPanel — debug-only inspector that decodes the run's metrics.extra
 * and payload.grounding into a structured "what actually happened" view.
 *
 * Surfaces (when present):
 *  - Operating mode (grounded/agent_only/hybrid/raw) — inferred from the
 *    request's grounded flag combined with whether the adapter emitted
 *    tool_use parts
 *  - Grounding metrics (extractor, terms, search_query, snippet count,
 *    sources, rerank stats, per-repo-cap drops, expansion stats)
 *  - Retrieved file paths from metrics.extra.grounding_paths
 *  - Pipeline routing decision (route tier, coverage, fallback path)
 *  - Tool calls + tool names (agentic adapters)
 *  - Raw metrics.extra dump for anything we don't render specially
 *
 * Lives under the answer body, collapsible. Driven by the global debug
 * toggle in the header.
 */
function TelemetryPanel({ turn }: { turn: RunDetail }) {
  const payload = (turn.payload || {}) as Record<string, unknown>;
  // Two shapes for "metrics.extra":
  //   - Live response (just after submit): payload.metrics.extra
  //   - Persisted (after /runs/{id} reload): payload.metrics_extra
  //     — api.py copies metrics.extra into that key so the runstore
  //     reload sees the same telemetry as the live response.
  const metricsExtra = (payload.metrics_extra as Record<string, unknown> | undefined) ?? undefined;
  const metrics = (payload.metrics as Record<string, unknown> | undefined) ?? {};
  const liveExtra = (metrics.extra as Record<string, unknown> | undefined) ?? undefined;
  const extra = metricsExtra ?? liveExtra ?? {};
  const grounding =
    (payload.grounding as Record<string, unknown> | undefined) ??
    ((extra.grounding as Record<string, unknown> | undefined) ?? undefined);
  const groundingMetrics =
    grounding && typeof grounding === "object"
      ? ((grounding.metrics as Record<string, unknown> | undefined) ?? grounding)
      : undefined;
  const groundingPaths = (extra.grounding_paths as string[] | undefined) ?? [];
  const toolCalls =
    typeof metrics.tool_calls === "number" ? metrics.tool_calls : undefined;
  const toolNames = (extra.opencode_tool_names as string[] | undefined) ?? [];

  // Pipeline-specific fields (set by adapters/_pipeline.py).
  const pipelineRoute = extra.pipeline_route as string | undefined;
  const pipelineReason = extra.pipeline_reason as string | undefined;
  const pipelinePath = extra.pipeline_path as string | undefined;
  const coverage = extra.coverage as
    | { score?: number; sufficient?: boolean; reason?: string }
    | undefined;
  const synthesisModel = extra.synthesis_model as string | undefined;
  const synthesisInsufficient = extra.synthesis_insufficient as boolean | undefined;
  const fallbackModel = extra.fallback_model as string | undefined;

  return (
    <details className="mt-4 group" open={false}>
      <summary className="cursor-pointer list-none inline-flex items-center gap-2 select-none">
        <span className="font-mono text-[10px] uppercase tracking-[0.16em] text-base-content/45 group-hover:text-base-content/65">
          ▸ Telemetry
        </span>
        <span className="text-base-content/25 text-[10px]">·</span>
        <span className="font-mono text-[10px] text-base-content/40">
          {grounding ? "grounded" : "no-grounding"}
          {pipelineRoute && ` · ${pipelineRoute}`}
          {toolCalls != null && ` · ${toolCalls} tool calls`}
        </span>
      </summary>

      <div className="mt-3 surface-2 p-4 space-y-4 text-[12px] font-mono leading-relaxed">
        {/* Pipeline routing decision — only when present */}
        {(pipelineRoute || pipelinePath || coverage) && (
          <Section title="Pipeline route">
            <KVRow k="tier" v={pipelineRoute} />
            <KVRow k="reason" v={pipelineReason} />
            <KVRow k="path" v={pipelinePath} />
            {coverage && (
              <>
                <KVRow
                  k="coverage"
                  v={
                    typeof coverage.score === "number"
                      ? `${coverage.score.toFixed(2)}  · ${
                          coverage.sufficient ? "sufficient" : "INSUFFICIENT"
                        }`
                      : undefined
                  }
                />
                <KVRow k="coverage_reason" v={coverage.reason} />
              </>
            )}
            <KVRow k="synthesis_model" v={synthesisModel} />
            {synthesisInsufficient !== undefined && (
              <KVRow
                k="synthesis_insufficient"
                v={synthesisInsufficient ? "yes (would trigger fallback)" : "no"}
              />
            )}
            {fallbackModel && <KVRow k="fallback_model" v={fallbackModel} />}
          </Section>
        )}

        {/* Grounding step */}
        {groundingMetrics && (
          <Section title="Grounding retrieval">
            <KVRow
              k="extractor"
              v={groundingMetrics.extractor as string | undefined}
            />
            <KVRow
              k="classifier"
              v={
                groundingMetrics.classifier_decision
                  ? `${groundingMetrics.classifier_decision}${
                      groundingMetrics.classifier_reason
                        ? ` — ${groundingMetrics.classifier_reason}`
                        : ""
                    }`
                  : undefined
              }
            />
            <KVRow
              k="terms"
              v={
                Array.isArray(groundingMetrics.extracted_terms)
                  ? (groundingMetrics.extracted_terms as unknown[]).join(", ")
                  : undefined
              }
            />
            <KVRow
              k="search_query"
              v={groundingMetrics.search_query as string | undefined}
            />
            <KVRow
              k="sources"
              v={
                Array.isArray(groundingMetrics.sources)
                  ? (groundingMetrics.sources as unknown[]).join(", ")
                  : undefined
              }
            />
            <KVRow
              k="snippets"
              v={String(groundingMetrics.snippet_count ?? 0)}
            />
            <KVRow
              k="total_chars"
              v={String(groundingMetrics.total_chars ?? 0)}
            />
            <KVRow
              k="sourcebot_files_seen"
              v={String(groundingMetrics.sourcebot_files_seen ?? 0)}
            />
            <KVRow
              k="serena_hits"
              v={String(groundingMetrics.serena_hits ?? 0)}
            />
            <KVRow
              k="rerank"
              v={
                groundingMetrics.rerank_model
                  ? `${groundingMetrics.rerank_model} · ${
                      groundingMetrics.rerank_candidates ?? "?"
                    } cands · ${groundingMetrics.rerank_ms ?? "?"}ms`
                  : "(not run)"
              }
            />
            <KVRow
              k="per_repo_cap"
              v={
                typeof groundingMetrics.per_repo_cap_applied === "number" &&
                groundingMetrics.per_repo_cap_applied > 0
                  ? `cap=${groundingMetrics.per_repo_cap_applied} · dropped=${groundingMetrics.per_repo_cap_dropped ?? 0}`
                  : "(disabled)"
              }
            />
            <KVRow
              k="window_expansion"
              v={
                typeof groundingMetrics.expanded_snippets === "number" &&
                groundingMetrics.expanded_snippets > 0
                  ? `expanded=${groundingMetrics.expanded_snippets} · added=${groundingMetrics.expansion_added_chars ?? 0} chars`
                  : "(disabled)"
              }
            />
            <KVRow
              k="serena_symbol_lookup"
              v={
                typeof groundingMetrics.serena_symbol_lookups === "number" &&
                groundingMetrics.serena_symbol_lookups > 0
                  ? `lookups=${groundingMetrics.serena_symbol_lookups} · hits=${groundingMetrics.serena_symbol_hits ?? 0} · ${groundingMetrics.serena_symbol_ms ?? "?"}ms`
                  : "(not run)"
              }
            />
            <KVRow
              k="duration_ms"
              v={String(groundingMetrics.duration_ms ?? 0)}
            />
            {Boolean(groundingMetrics.error) && (
              <KVRow k="error" v={String(groundingMetrics.error)} variant="danger" />
            )}
          </Section>
        )}

        {/* Tool calls — for agentic adapters */}
        {toolCalls != null && toolCalls > 0 && (
          <Section title="Agentic tools">
            <KVRow k="tool_calls" v={String(toolCalls)} />
            {toolNames.length > 0 && (
              <KVRow k="tool_names" v={toolNames.join(", ")} />
            )}
          </Section>
        )}

        {/* Retrieved files — the inputs that fed the model */}
        {groundingPaths.length > 0 && (
          <Section title={`Retrieved files (${groundingPaths.length})`}>
            <ul className="space-y-0.5 text-[11px] text-base-content/75">
              {groundingPaths.map((p) => (
                <li key={p} className="truncate" title={p}>
                  <span className="text-base-content/35">·</span> {p}
                </li>
              ))}
            </ul>
          </Section>
        )}

        {/* Raw extras — the catch-all */}
        <details className="group/raw">
          <summary className="cursor-pointer list-none inline-flex items-center gap-2 select-none">
            <span className="font-mono text-[10px] uppercase tracking-[0.16em] text-base-content/35 group-hover/raw:text-base-content/55">
              ▸ Raw metrics.extra
            </span>
          </summary>
          <pre className="mt-2 text-[10px] leading-snug text-base-content/55 whitespace-pre-wrap break-all max-h-72 overflow-y-auto">
            {JSON.stringify(extra, null, 2)}
          </pre>
        </details>
      </div>
    </details>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div>
      <h4 className="font-mono text-[10px] uppercase tracking-[0.16em] text-base-content/55 mb-2">
        {title}
      </h4>
      <div className="space-y-1">{children}</div>
    </div>
  );
}

function KVRow({
  k,
  v,
  variant,
}: {
  k: string;
  v: string | undefined;
  variant?: "danger";
}) {
  if (v == null || v === "") return null;
  return (
    <div className="grid grid-cols-[10rem_1fr] gap-3 items-start">
      <span className="text-base-content/45 text-[11px]">{k}</span>
      <span
        className={`text-[11px] break-all ${
          variant === "danger" ? "text-error" : "text-base-content/90"
        }`}
      >
        {v}
      </span>
    </div>
  );
}

function ChatTurn({
  turn,
  suppressRunningAssistant,
  showTelemetry,
}: {
  turn: RunDetail;
  /** While the user just submitted, ProgressTimeline shows progress — hide duplicate "Thinking…". */
  suppressRunningAssistant?: boolean;
  /** When true, render the TelemetryPanel under the answer. Driven by header toggle. */
  showTelemetry?: boolean;
}) {
  const q = extractQuestion(turn);
  const disp = turnToDisplayed(turn);
  const grounding = extractGrounding(turn.payload);
  return (
    <div className="flex flex-col gap-3">
      {q ? <QuestionBubble text={q} /> : null}
      {turn.status === "running" && !suppressRunningAssistant && (
        <div className="flex flex-col animate-msg-enter">
          <RoleLabel role="Assistant" align="left" />
          <p className="text-sm text-base-content/80 pl-1 flex items-center gap-2">
            <span className="loading loading-dots loading-sm text-primary" />
            Thinking…
          </p>
        </div>
      )}
      {turn.status === "failed" && (
        <div className="rounded-2xl border border-error/30 bg-error/5 p-4 text-sm shadow-sm ring-1 ring-error/10">
          {turn.error ? (
            <pre className="text-xs text-error font-mono-ui whitespace-pre-wrap">{turn.error}</pre>
          ) : (
            <p className="text-base-content/85">This turn failed.</p>
          )}
        </div>
      )}
      {turn.status === "completed" && (turn.answer?.length ?? 0) > 0 && (
        <div className="flex flex-col animate-msg-enter">
          <RoleLabel role="Assistant" align="left" />
          <article className="rounded-2xl border border-base-300/70 bg-gradient-to-b from-base-100 to-base-100/95 shadow-lg shadow-base-300/15 overflow-hidden ring-1 ring-base-content/[0.1]">
            <div className="h-1 bg-gradient-to-r from-primary/50 via-secondary/40 to-primary/30" aria-hidden />
            <div className="px-4 sm:px-5 pt-3.5 pb-3 flex flex-col sm:flex-row sm:items-start gap-3 sm:justify-between border-b border-base-200/80 bg-base-200/20">
              <MetaStrip run={disp} />
              <CopyButton text={disp.answer} />
            </div>
            <div className="px-4 sm:px-5 py-5">
              <AnswerBody text={disp.answer} />
              <Citations citations={disp.citations} />
              {grounding && <GroundingPanel grounding={grounding} />}
              {showTelemetry && <TelemetryPanel turn={turn} />}
            </div>
          </article>
        </div>
      )}
      {turn.status === "completed" && !(turn.answer?.length ?? 0) && (
        <p className="text-sm text-base-content/75 pl-1">No answer text for this turn.</p>
      )}
    </div>
  );
}

function EmptyState({ onPick }: { onPick: (q: string) => void }) {
  return (
    <div className="flex flex-col items-center justify-center text-center py-10 sm:py-14 px-4">
      <div className="mb-6 flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br from-primary/15 to-secondary/10 text-xl font-display font-semibold text-primary shadow-inner ring-1 ring-primary/15">
        ◈
      </div>
      <h2 className="font-display text-2xl sm:text-[1.7rem] font-semibold tracking-tight mb-2 text-base-content">
        Ask the codebase
      </h2>
      <p className="text-sm text-base-content/78 mb-8 max-w-md leading-relaxed">
        Each sidebar chat is its own session: follow-ups keep prior turns in context. Open a chat to
        continue, or start fresh below.
      </p>
      <div className="flex flex-col gap-2.5 w-full max-w-lg text-left rounded-2xl border border-base-300/70 bg-base-100/70 p-4 shadow-md ring-1 ring-base-content/[0.08] backdrop-blur-sm">
        <p className="text-[10px] uppercase tracking-wider text-base-content/65 font-semibold px-0.5">
          Try
        </p>
        {SUGGESTED_PROMPTS.map((p) => (
          <button
            key={p}
            type="button"
            onClick={() => onPick(p)}
            className="text-left text-sm px-4 py-3.5 rounded-xl border border-base-300/60 bg-base-200/30 hover:bg-base-100 hover:border-primary/30 hover:shadow-sm transition-all duration-200"
          >
            {p}
          </button>
        ))}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main app
// ---------------------------------------------------------------------------

export function App() {
  const [view, setView] = useState<View>("ask");
  // Persisted "show telemetry panels under answers" toggle. localStorage
  // so it survives page reloads. Default off so casual users aren't bombarded.
  const [debugEnabled, setDebugEnabled] = useState<boolean>(() => {
    if (typeof window === "undefined") return false;
    return window.localStorage.getItem("polycontext.debug") === "1";
  });
  useEffect(() => {
    if (typeof window === "undefined") return;
    window.localStorage.setItem("polycontext.debug", debugEnabled ? "1" : "0");
  }, [debugEnabled]);
  const [form, setForm] = useState<AskFormState>(INITIAL_FORM);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<RunListItem[]>([]);
  /** Shown at top of sidebar until the server lists the new thread (POST /ask is still in flight). */
  const [sidebarPlaceholder, setSidebarPlaceholder] = useState<{
    id: string;
    preview: string;
    adapter: string;
  } | null>(null);
  const [activeThreadId, setActiveThreadId] = useState<string | null>(null);
  const [threadMessages, setThreadMessages] = useState<RunDetail[]>([]);
  const [threadListLoading, setThreadListLoading] = useState(false);
  const [optimisticUser, setOptimisticUser] = useState<string | null>(null);
  const [progress, setProgress] = useState<LiveProgress | null>(null);
  const [adapters, setAdapters] = useState<AdapterInfo[]>([]);
  const [adapterListHydrated, setAdapterListHydrated] = useState(false);
  const [adapterListError, setAdapterListError] = useState<string | null>(null);
  /** Curated model catalog for the currently-selected adapter, fetched
   * lazily on adapter change. Empty list ⇒ model picker is hidden. */
  const [adapterModels, setAdapterModels] = useState<AdapterModelInfo[]>([]);
  /** Live OpenRouter model catalog (~350 models) for adapters that accept
   * OpenRouter-format ids (opencode, pipeline). Cached 1h server-side. */
  const [openrouterModels, setOpenrouterModels] = useState<AdapterModelInfo[]>([]);
  const progressTimer = useRef<number | null>(null);
  const threadRef = useRef<HTMLDivElement>(null);
  /** Invalidate stale thread / detail fetches when starting a new action. */
  const detailFetchGen = useRef(0);
  const composerRef = useRef<HTMLTextAreaElement>(null);
  /** Skip wiping transcript when re-opening the same thread from the sidebar. */
  const displayedThreadRef = useRef<string | null>(null);
  const loadHistory = useCallback(async () => {
    try {
      const rows = await api.listRuns({ mode: "ask", limit: 50, per_thread: true });
      setHistory(rows);
    } catch (e) {
      console.warn("history fetch failed", e);
    }
  }, []);

  useEffect(() => {
    loadHistory();
    api.listAdapters()
      .then((r) => {
        setAdapterListError(null);
        const askables = r.adapters
          .filter((a) => a.capabilities.includes("ask"))
          .sort(
            (a, b) =>
              (ADAPTER_PRIORITY[a.name] ?? 99) - (ADAPTER_PRIORITY[b.name] ?? 99),
          );
        setAdapters(askables);
        setForm((f) => {
          const current = askables.find((a) => a.name === f.adapter);
          if (current && current.health.ok) return f;
          const firstHealthy = askables.find((a) => a.health.ok);
          return firstHealthy ? { ...f, adapter: firstHealthy.name } : f;
        });
      })
      .catch((e) => {
        setAdapterListError((e as Error).message ?? String(e));
        console.warn("adapter list fetch failed", e);
      })
      .finally(() => setAdapterListHydrated(true));
  }, [loadHistory]);

  // Refresh the model catalog whenever the selected adapter changes.
  // Cancels stale responses if the user switches adapters before fetch returns.
  useEffect(() => {
    if (!form.adapter) {
      setAdapterModels([]);
      return;
    }
    let cancelled = false;
    api.listAdapterModels(form.adapter)
      .then((r) => {
        if (cancelled) return;
        setAdapterModels(r.models);
        setOpenrouterModels(r.openrouter_live || []);
        // If the currently-picked model isn't in either section, reset to
        // the adapter default (empty string).
        setForm((f) => {
          if (!f.model) return f;
          const stillValid =
            r.models.some((m) => m.id === f.model) ||
            (r.openrouter_live || []).some((m) => m.id === f.model);
          return stillValid ? f : { ...f, model: "" };
        });
      })
      .catch((e) => {
        if (cancelled) return;
        console.warn(`models fetch failed for ${form.adapter}`, e);
        setAdapterModels([]);
        setOpenrouterModels([]);
      });
    return () => { cancelled = true; };
  }, [form.adapter]);

  /** Auto-refresh sidebar history while the tab is visible; also refresh when returning to the tab. */
  useEffect(() => {
    const tick = () => {
      if (document.visibilityState === "visible") {
        void loadHistory();
      }
    };
    const intervalId = window.setInterval(tick, 15_000);
    const onVisibility = () => {
      if (document.visibilityState === "visible") tick();
    };
    document.addEventListener("visibilitychange", onVisibility);
    return () => {
      window.clearInterval(intervalId);
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }, [loadHistory]);

  const startNew = useCallback(() => {
    detailFetchGen.current += 1;
    if (progressTimer.current != null) {
      window.clearInterval(progressTimer.current);
      progressTimer.current = null;
    }
    setActiveThreadId(null);
    setThreadMessages([]);
    setError(null);
    setThreadListLoading(false);
    setProgress(null);
    setOptimisticUser(null);
    displayedThreadRef.current = null;
    setSidebarPlaceholder(null);
    setForm((f) => ({ ...f, question: "" }));
    window.requestAnimationFrame(() => {
      composerRef.current?.focus();
      threadRef.current?.scrollTo({ top: 0 });
    });
  }, []);

  const openThread = useCallback(async (tid: string) => {
    if (tid.startsWith(PENDING_SIDEBAR_PREFIX)) return;
    const gen = ++detailFetchGen.current;
    const prev = displayedThreadRef.current;
    const switching = prev !== null && prev !== tid;
    displayedThreadRef.current = tid;
    setActiveThreadId(tid);
    setError(null);
    setOptimisticUser(null);
    if (switching) {
      setThreadMessages([]);
    }
    setThreadListLoading(true);
    try {
      const rows = await api.listThreadRuns(tid);
      if (gen !== detailFetchGen.current) return;
      setThreadMessages(rows);
    } catch (err) {
      if (gen !== detailFetchGen.current) return;
      setError((err as Error).message);
    } finally {
      if (gen === detailFetchGen.current) setThreadListLoading(false);
    }
  }, []);

  const pickedAdapter = useMemo(
    () => adapters.find((a) => a.name === form.adapter),
    [adapters, form.adapter],
  );

  const handleSubmit = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault();
      const question = form.question.trim();
      if (!question || submitting) return;

      const adapterMeta = adapters.find((a) => a.name === form.adapter);
      if (!adapterListHydrated) {
        setError("Still loading adapters — wait a moment and try again.");
        return;
      }
      if (adapters.length > 0 && adapterMeta && !adapterMeta.health.ok) {
        setError(
          `Adapter "${form.adapter}" is not ready: ${adapterMeta.health.reason ?? "unhealthy"}`,
        );
        return;
      }

      setSubmitting(true);
      setError(null);
      detailFetchGen.current += 1;
      const submitGen = detailFetchGen.current;
      setThreadListLoading(false);
      setOptimisticUser(question);
      setForm((f) => ({ ...f, question: "" }));
      if (!activeThreadId) {
        setSidebarPlaceholder({
          id: `${PENDING_SIDEBAR_PREFIX}${submitGen}`,
          preview: question,
          adapter: form.adapter,
        });
      }

      const startedAt = Date.now();
      const sinceIso = new Date(startedAt - 1000).toISOString();
      setProgress({
        startedAt,
        elapsed: 0,
        currentStage: "engine:structured",
        stagesDone: ["input:blocking", "enrich:(none)"],
      });
      const tick = async () => {
        void loadHistory();
        try {
          const rows = await api.listRuns({
            mode: "ask",
            status: "running",
            limit: 1,
            since: sinceIso,
          });
          const row = rows[0];
          setProgress((prev) =>
            prev
              ? {
                  ...prev,
                  elapsed: (Date.now() - prev.startedAt) / 1000,
                  currentStage: row?.current_stage ?? prev.currentStage,
                  stagesDone: row?.stages_done ?? prev.stagesDone ?? [],
                }
              : prev,
          );
        } catch {
          setProgress((prev) =>
            prev ? { ...prev, elapsed: (Date.now() - prev.startedAt) / 1000 } : prev,
          );
        }
      };
      progressTimer.current = window.setInterval(tick, 1000);
      void tick();

      let askCompleted = false;
      try {
        if (!form.adapter) {
          throw new Error("pick an adapter from the dropdown before submitting");
        }
        const wireMode = askModeToWire(form.mode);
        const askPromise = api.adapterAsk(form.adapter, {
          query: question,
          grounded: wireMode.grounded,
          tools_enabled: wireMode.tools_enabled,
          ...(form.model ? { model: form.model } : {}),
          ...(activeThreadId ? { thread_id: activeThreadId } : {}),
        });
        void loadHistory();
        queueMicrotask(() => void loadHistory());
        window.setTimeout(() => void loadHistory(), 280);
        window.setTimeout(() => void loadHistory(), 750);
        const r = await askPromise;
        askCompleted = true;
        if (submitGen !== detailFetchGen.current) return;
        setActiveThreadId(r.thread_id);
        displayedThreadRef.current = r.thread_id;
        await loadHistory();
        const rows = await api.listThreadRuns(r.thread_id);
        if (submitGen !== detailFetchGen.current) return;
        setThreadMessages(rows);
      } catch (err) {
        if (submitGen === detailFetchGen.current) {
          setError((err as Error).message);
          if (!askCompleted) {
            setForm((f) => ({ ...f, question: question }));
          }
        }
      } finally {
        if (progressTimer.current != null) {
          window.clearInterval(progressTimer.current);
          progressTimer.current = null;
        }
        setProgress(null);
        setOptimisticUser(null);
        setSubmitting(false);
        setSidebarPlaceholder(null);
      }
    },
    [form, submitting, loadHistory, adapters, adapterListHydrated, activeThreadId],
  );

  useEffect(() => {
    return () => {
      if (progressTimer.current != null) {
        window.clearInterval(progressTimer.current);
      }
    };
  }, []);

  const handleReplay = useCallback(
    async (id: string) => {
      setError(null);
      try {
        await api.replayRun(id);
      } catch (err) {
        setError((err as Error).message);
        return;
      }
      await loadHistory();
    },
    [loadHistory],
  );

  const historyWithPlaceholder = useMemo(() => {
    if (!sidebarPlaceholder) return history;
    const preview = sidebarPlaceholder.preview.trim();
    const serverCaughtUp = history.some(
      (h) =>
        h.status === "running" && (h.input_preview ?? "").trim() === preview,
    );
    if (serverCaughtUp) return history;
    return [phantomHistoryRow(sidebarPlaceholder), ...history];
  }, [history, sidebarPlaceholder]);

  const historyGroups = useMemo(
    () => groupHistoryByDay(historyWithPlaceholder),
    [historyWithPlaceholder],
  );

  useEffect(() => {
    const el = threadRef.current;
    if (!el) return;
    el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
  }, [threadMessages, submitting, optimisticUser, threadListLoading]);

  return (
    <div className="app-surface min-h-screen text-base-content">
      <div className="app-grain" aria-hidden />
      <div className="app-shell-content grid grid-cols-1 md:grid-cols-[minmax(17rem,20rem)_1fr] h-screen min-h-0">
        <aside className="min-h-0 flex flex-col hairline-r bg-base-100/60 backdrop-blur-md">
          {/* Brand block — tight 2-line ID, monospace, no gradient blob. */}
          <div className="px-4 pt-5 pb-4 hairline-b">
            <div className="flex items-center gap-2.5">
              <div className="relative flex h-7 w-7 shrink-0 items-center justify-center rounded-md border border-primary/30 bg-primary/8">
                <span className="font-display text-[14px] font-semibold text-primary leading-none">▰</span>
                <span className="absolute -bottom-px left-1/2 -translate-x-1/2 h-px w-3.5 bg-gradient-to-r from-transparent via-primary/70 to-transparent" />
              </div>
              <div className="min-w-0">
                <p className="font-display text-[14px] font-medium tracking-tight leading-tight text-base-content/95">
                  polycontext
                </p>
                <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-base-content/40 leading-tight mt-0.5">
                  code Q&amp;A
                </p>
              </div>
            </div>
          </div>

          {/* New question — ghost-style button; primary action lives in the composer. */}
          <div className="px-3 py-3 hairline-b">
            <button
              type="button"
              onClick={startNew}
              className="w-full inline-flex items-center justify-center gap-2 rounded-lg border border-base-300 bg-base-200/50 hover:bg-base-200 hover:border-primary/30 transition-colors px-3 py-1.5 font-mono text-[11px] tracking-wide text-base-content/85"
              title="Clear the composer and start a new thread"
            >
              <span className="text-primary text-sm leading-none">＋</span>
              <span>New thread</span>
            </button>
          </div>

          {/* History header — tight tracking, refresh tucked at the end. */}
          <div className="px-4 pt-4 pb-2 flex items-center justify-between">
            <h2 className="font-mono text-[10px] font-medium uppercase tracking-[0.18em] text-base-content/45">
              History
            </h2>
            <button
              type="button"
              onClick={loadHistory}
              className="inline-flex h-5 w-5 items-center justify-center rounded text-base-content/45 hover:text-base-content hover:bg-base-200/50 transition-colors"
              aria-label="Refresh history"
              title="Refresh now (also updates every 15s while this tab is visible)"
            >
              <span className="text-[11px]">↻</span>
            </button>
          </div>

          <div className="flex-1 min-h-0 overflow-y-auto px-2 pb-4">
            {historyWithPlaceholder.length === 0 ? (
              <p className="text-[11px] text-base-content/50 px-3 py-6 leading-relaxed font-mono">
                no runs yet. ask in the composer.
              </p>
            ) : (
              <div className="flex flex-col gap-3">
                {historyGroups.map((g) => (
                  <div key={g.label}>
                    <p className="font-mono text-[10px] uppercase tracking-[0.16em] text-base-content/35 px-3 mb-1">
                      {g.label}
                    </p>
                    <ul className="flex flex-col gap-0.5">
                      {g.items.map((r) => (
                        <li key={r.id}>
                          <HistoryItem
                            run={r}
                            selected={
                              threadKey(r) === activeThreadId ||
                              (sidebarPlaceholder != null &&
                                activeThreadId == null &&
                                threadKey(r) === sidebarPlaceholder.id)
                            }
                            onSelect={openThread}
                            onReplay={handleReplay}
                          />
                        </li>
                      ))}
                    </ul>
                  </div>
                ))}
              </div>
            )}
          </div>
        </aside>

        <main className="min-h-0 flex flex-col h-full min-w-0">
          <header className="shrink-0 hairline-b bg-base-100/60 backdrop-blur-md px-4 sm:px-6 py-3">
            <div className="flex items-center justify-between gap-4">
              <div className="flex items-center gap-3 min-w-0">
                <h1 className="font-display text-[15px] font-medium tracking-tight text-base-content/95">
                  {view === "ask" ? "Ask" : "Decompose"}
                </h1>
                <span className="font-mono text-[10px] uppercase tracking-[0.18em] text-base-content/35 hidden md:inline">
                  {view === "ask"
                    ? activeThreadId
                      ? "thread · follow-up context"
                      : pickedAdapter
                        ? `route ${pickedAdapter.name}`
                        : "no adapter"
                    : "ticket → subtasks"}
                </span>
              </div>
              <div className="flex items-center gap-3 shrink-0">
                {/* Debug toggle — shows the Telemetry panel under each answer. */}
                <label
                  className="inline-flex items-center gap-2 cursor-pointer select-none"
                  title="Show retrieval + tool-call telemetry under each answer"
                >
                  <span className="font-mono text-[10px] uppercase tracking-[0.14em] text-base-content/45 hidden sm:inline">
                    Debug
                  </span>
                  <button
                    type="button"
                    role="switch"
                    aria-checked={debugEnabled}
                    onClick={() => setDebugEnabled((v) => !v)}
                    className={`relative inline-flex h-[18px] w-[30px] items-center rounded-full transition-colors ${
                      debugEnabled
                        ? "bg-primary/70"
                        : "bg-base-300 hover:bg-base-200"
                    }`}
                  >
                    <span
                      className={`inline-block h-3 w-3 rounded-full bg-base-100 shadow-sm transition-transform ${
                        debugEnabled ? "translate-x-[14px]" : "translate-x-[2px]"
                      }`}
                    />
                  </button>
                </label>

                {/* Mode tabs styled as a segmented control matching the composer. */}
                <div role="tablist" className="segmented" aria-label="View">
                  <button
                    type="button"
                    role="tab"
                    aria-pressed={view === "ask"}
                    onClick={() => setView("ask")}
                  >
                    Ask
                  </button>
                  <button
                    type="button"
                    role="tab"
                    aria-pressed={view === "decompose"}
                    onClick={() => setView("decompose")}
                  >
                    Decompose
                  </button>
                </div>
              </div>
            </div>
          </header>

          {view === "decompose" ? (
            <Decompose />
          ) : (
          <>
          <div
            ref={threadRef}
            className="flex-1 min-h-0 overflow-y-auto px-4 sm:px-6 py-6"
          >
            <div className="max-w-3xl mx-auto flex flex-col gap-6">
              {adapterListHydrated && !adapterListError && adapters.length === 0 && (
                <div className="alert alert-warning text-xs shadow-sm">
                  No adapters with <code>ask</code> capability. Check{" "}
                  <code className="text-[10px]">ENABLED_ADAPTERS</code> and server logs.
                </div>
              )}

              {adapterListError && (
                <div className="alert alert-warning text-xs font-mono whitespace-pre-wrap shadow-sm">
                  Could not load /v1/adapters: {adapterListError}
                </div>
              )}

              {!adapterListError && adapterListHydrated && pickedAdapter && !pickedAdapter.health.ok && (
                <div className="alert alert-warning text-xs shadow-sm">
                  Adapter <strong>{pickedAdapter.name}</strong> is unhealthy —{" "}
                  {pickedAdapter.health.reason ?? "waiting for readiness"}.
                </div>
              )}

              {error && (
                <div className="alert alert-error text-xs font-mono whitespace-pre-wrap shadow-lg">
                  <div className="flex-1">{error}</div>
                  <button
                    type="button"
                    onClick={() => setError(null)}
                    className="btn btn-ghost btn-xs"
                    aria-label="Dismiss error"
                  >
                    ✕
                  </button>
                </div>
              )}

              <section className="flex flex-col gap-6">
                {threadListLoading && threadMessages.length === 0 && !submitting && (
                  <div className="rounded-2xl border border-base-300 bg-base-100/90 p-6 flex items-center gap-3 text-sm text-base-content/85 shadow-sm">
                    <span className="loading loading-spinner loading-sm text-primary" />
                    Loading chat…
                  </div>
                )}

                {threadMessages.map((turn) => (
                  <ChatTurn
                    key={turn.id}
                    turn={turn}
                    suppressRunningAssistant={submitting && turn.status === "running"}
                    showTelemetry={debugEnabled}
                  />
                ))}

                {submitting && progress && (
                  <div className="flex flex-col gap-3">
                    {optimisticUser &&
                      !(
                        threadMessages.length > 0 &&
                        threadMessages[threadMessages.length - 1]?.status === "running" &&
                        extractQuestion(threadMessages[threadMessages.length - 1]).trim() === optimisticUser.trim()
                      ) && <QuestionBubble text={optimisticUser} />}
                    <div className="flex flex-col">
                      <RoleLabel role="Assistant" align="left" />
                      <ProgressTimeline progress={progress} />
                    </div>
                  </div>
                )}

                {!submitting &&
                  !threadListLoading &&
                  threadMessages.length === 0 &&
                  activeThreadId === null &&
                  !optimisticUser && (
                    <EmptyState onPick={(q) => setForm((f) => ({ ...f, question: q }))} />
                  )}
              </section>
            </div>
          </div>

          <div className="shrink-0 hairline-t bg-base-100/95 backdrop-blur-md px-4 sm:px-6 py-4">
            <form
              onSubmit={handleSubmit}
              className="max-w-4xl mx-auto w-full"
            >
              {/* Composer card — textarea above; controls strip below in a hairline-separated row. */}
              <div className="surface overflow-hidden focus-within:border-base-300 transition-colors">
                <textarea
                  ref={composerRef}
                  placeholder="Ask about architecture, flows, or where something lives…"
                  value={form.question}
                  onChange={(e) => setForm((f) => ({ ...f, question: e.target.value }))}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
                      e.preventDefault();
                      void handleSubmit(e as unknown as React.FormEvent);
                    }
                  }}
                  rows={3}
                  className="w-full px-5 py-4 bg-transparent text-[15px] leading-relaxed resize-y min-h-20 focus:outline-none placeholder:text-base-content/40 font-sans"
                />
                {/* Three labeled pickers + Ask button, separated by hairlines. */}
                <div className="hairline-t flex items-end gap-4 flex-wrap px-4 py-3 bg-base-100/40">
                  <div className="picker">
                    <span className="picker-label">SDK</span>
                    <select
                      value={form.adapter}
                      onChange={(e) => setForm((f) => ({ ...f, adapter: e.target.value }))}
                      disabled={!adapterListHydrated || adapters.length === 0}
                      title="POST /v1/adapters/{name}/ask"
                      className="min-w-[10rem]"
                    >
                      {!adapterListHydrated ? (
                        <option value={form.adapter}>Loading…</option>
                      ) : adapters.length === 0 ? (
                        <option value="">No SDKs</option>
                      ) : (
                        adapters.map((a) => (
                          <option
                            key={a.name}
                            value={a.name}
                            title={a.health.ok ? a.description : (a.health.reason ?? "unhealthy")}
                          >
                            {displayAdapterName(a.name)}
                            {a.health.ok ? "" : " (unhealthy)"}
                          </option>
                        ))
                      )}
                    </select>
                  </div>

                  {(adapterModels.length > 0 || openrouterModels.length > 0) && (
                    <div className="picker">
                      <span className="picker-label">Model</span>
                      <select
                        value={form.model}
                        onChange={(e) => setForm((f) => ({ ...f, model: e.target.value }))}
                        title="Per-request model override. Provider credentials must be set in .env."
                        className="min-w-[14rem] max-w-[18rem]"
                      >
                        <option value="">(adapter default)</option>
                        {adapterModels.length > 0 && (
                          <optgroup label="Recommended">
                            {adapterModels.map((m) => (
                              <option key={m.id} value={m.id} title={m.note ?? ""}>
                                {m.name}
                                {typeof m.in_per_m_usd === "number"
                                  ? `  · $${m.in_per_m_usd}/${m.out_per_m_usd}/M`
                                  : ""}
                              </option>
                            ))}
                          </optgroup>
                        )}
                        {openrouterModels.length > 0 && (
                          <optgroup label={`OpenRouter live (${openrouterModels.length})`}>
                            {openrouterModels.map((m) => (
                              <option key={m.id} value={m.id} title={m.note ?? ""}>
                                {m.name}
                                {typeof m.in_per_m_usd === "number"
                                  ? `  · $${m.in_per_m_usd}/${m.out_per_m_usd}/M`
                                  : ""}
                              </option>
                            ))}
                          </optgroup>
                        )}
                      </select>
                    </div>
                  )}

                  {/* Mode — segmented control. 4 first-class pills, far better
                      than a <select> for the small finite set of options. */}
                  <div className="picker">
                    <span className="picker-label" title={ASK_MODE_DESCRIPTIONS[form.mode]}>Mode</span>
                    <div className="segmented" role="tablist" aria-label="Operating mode">
                      {(Object.keys(ASK_MODE_LABELS) as AskMode[]).map((m) => (
                        <button
                          key={m}
                          type="button"
                          role="tab"
                          aria-pressed={form.mode === m}
                          aria-selected={form.mode === m}
                          onClick={() => setForm((f) => ({ ...f, mode: m }))}
                          title={ASK_MODE_DESCRIPTIONS[m]}
                        >
                          {askModeShortLabel(m)}
                        </button>
                      ))}
                    </div>
                  </div>

                  <div className="flex-1" />

                  <div className="flex items-center gap-3">
                    <span className="font-mono text-[10px] text-base-content/45 tracking-[0.12em] uppercase hidden sm:inline">
                      ⌘↵ Send
                    </span>
                    <button
                      type="submit"
                      className="btn btn-primary btn-sm rounded-lg gap-2 px-4 font-mono text-[12px] font-medium tracking-wide normal-case"
                      disabled={
                        submitting ||
                        !adapterListHydrated ||
                        !form.question.trim() ||
                        !pickedAdapter?.health.ok
                      }
                    >
                      {submitting && <span className="loading loading-spinner loading-xs" />}
                      <span>{submitting ? "asking…" : "ask"}</span>
                    </button>
                  </div>
                </div>
              </div>
              {/* Helper line below — keeps mode context legible without
                  cramping the controls strip itself. */}
              <p className="text-[11px] text-base-content/45 mt-2 font-mono leading-relaxed">
                <span className="text-base-content/65">{ASK_MODE_LABELS[form.mode].toLowerCase()}</span>
                <span className="text-base-content/30"> · </span>
                {ASK_MODE_DESCRIPTIONS[form.mode]}
              </p>
            </form>
          </div>
          </>
          )}
        </main>
      </div>
    </div>
  );
}
