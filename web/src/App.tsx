import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { api } from "./api";
import type { AdapterInfo, RunDetail, RunListItem } from "./api";

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
};

// Initial dropdown selection. Overridden at mount if this adapter isn't
// installed or isn't healthy — see the adapter list useEffect.
const DEFAULT_ADAPTER = "cursor";

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

function statusDotClass(status: string): string {
  switch (status) {
    case "completed":
      return "bg-success";
    case "failed":
      return "bg-error";
    case "running":
      return "bg-warning animate-pulse";
    default:
      return "bg-base-content/30";
  }
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
      className={`text-[10px] font-semibold uppercase tracking-[0.16em] text-base-content/40 mb-1 ${
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
  return (
    <button
      type="button"
      onClick={() => onSelect(threadKey(run))}
      className={`group relative w-full text-left rounded-2xl px-3 py-3 transition-all duration-200 outline-none focus-visible:ring-2 focus-visible:ring-primary/35 focus-visible:ring-offset-2 focus-visible:ring-offset-base-100 ${
        selected
          ? "bg-base-100 shadow-md ring-1 ring-primary/25 border-l-[3px] border-l-primary"
          : "bg-base-100/50 hover:bg-base-100 hover:shadow-sm border border-base-300/50 border-l-[3px] border-l-transparent"
      }`}
    >
      <div className="flex items-start gap-2.5">
        <span
          className={`mt-1.5 inline-block w-2 h-2 rounded-full shrink-0 ring-2 ring-base-100 ${statusDotClass(
            run.status,
          )}`}
          aria-hidden
        />
        <div className="min-w-0 flex-1">
          <p className="text-[13px] leading-snug line-clamp-2 text-base-content font-medium tracking-tight">
            {historyTitle(run)}
          </p>
          <div className="mt-1.5 flex flex-wrap items-center gap-x-2 gap-y-0.5 text-[11px] text-base-content/45">
            <span className="tabular-nums">{formatTimeAgo(run.created_at)}</span>
            {adapter && (
              <>
                <span aria-hidden className="text-base-content/30">
                  ·
                </span>
                <span className="font-mono-ui text-[10px] uppercase tracking-wider text-base-content/40">
                  {adapter}
                </span>
              </>
            )}
          </div>
          {(run.total_seconds != null || run.total_cost_usd != null) && (
            <div className="mt-1 text-[10px] text-base-content/45 tabular-nums">
              {run.total_seconds != null && <span>{formatWall(run.total_seconds)}</span>}
              {run.total_seconds != null && run.total_cost_usd != null && (
                <span className="mx-1 text-base-content/30">·</span>
              )}
              {run.total_cost_usd != null && <span>{formatCost(run.total_cost_usd)}</span>}
            </div>
          )}
        </div>
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
      </div>
    </button>
  );
}

function ProgressTimeline({ progress }: { progress: LiveProgress }) {
  return (
    <div className="rounded-2xl border border-base-300/80 bg-gradient-to-br from-base-100 to-base-200/40 p-5 shadow-inner ring-1 ring-base-content/[0.04] animate-msg-enter">
      <div className="flex items-center gap-3 mb-5">
        <span className="loading loading-spinner loading-sm text-primary" />
        <span className="text-sm font-medium tracking-tight">{prettyStage(progress.currentStage)}…</span>
        <span className="ml-auto font-mono-ui text-xs text-base-content/50 tabular-nums">
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
                      : "bg-base-300/50 text-base-content/25"
                }`}
              >
                {st === "done" ? "✓" : st === "active" ? "●" : ""}
              </span>
              <span
                className={
                  st === "pending"
                    ? "text-base-content/38"
                    : st === "active"
                      ? "text-base-content font-medium"
                      : "text-base-content/65"
                }
              >
                {s.label}
              </span>
            </li>
          );
        })}
      </ol>
      <p className="mt-5 text-[11px] leading-relaxed text-base-content/45 border-t border-base-300/50 pt-4">
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
    <details className="mt-6 rounded-xl border border-base-300/80 bg-base-200/25 shadow-inner ring-1 ring-base-content/[0.03] group open:bg-base-200/35">
      <summary className="cursor-pointer select-none list-none px-4 py-3 text-sm font-medium text-base-content/85 flex items-center gap-3 hover:bg-base-200/40 rounded-xl transition-colors [&::-webkit-details-marker]:hidden">
        <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-base-100/80 text-[11px] font-semibold tabular-nums text-base-content/55 ring-1 ring-base-300/60">
          {citations.length}
        </span>
        <span className="flex-1 min-w-0">
          <span className="block tracking-tight">Sources</span>
          <span className="block text-[11px] font-normal text-base-content/45 mt-0.5">
            Citations from the indexed codebase
          </span>
        </span>
        <span className="text-base-content/35 text-xs transition-transform group-open:rotate-90">›</span>
      </summary>
      <ul className="px-3 pb-3 pt-0 flex flex-col gap-2.5 text-sm border-t border-base-300/50">
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
              className="mt-3 first:mt-3 rounded-lg border border-base-300/55 bg-base-100/70 pl-3 pr-3 py-2.5"
            >
              <code className="font-mono-ui text-[11px] leading-relaxed text-base-content/80 block break-all">
                {repo ? `${repo} / ` : ""}
                {path}
                {range}
              </code>
              {snippet && (
                <pre className="mt-2 whitespace-pre-wrap text-[11px] leading-relaxed text-base-content/72 font-mono-ui bg-base-300/35 rounded-md px-2.5 py-2 overflow-x-auto border border-base-300/40">
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
  const items: Array<{ label: string; value: string }> = [];
  if (run.engine) items.push({ label: "Engine", value: run.engine });
  if (run.model) items.push({ label: "Model", value: run.model });
  if (run.wall_seconds != null)
    items.push({ label: "Time", value: formatWall(run.wall_seconds) });
  if (run.cost_usd != null) items.push({ label: "Cost", value: formatCost(run.cost_usd) });
  if (run.input_tokens != null || run.output_tokens != null) {
    items.push({
      label: "Tokens",
      value: `${run.input_tokens ?? 0} → ${run.output_tokens ?? 0}`,
    });
  }
  if (!items.length) return null;
  return (
    <div className="flex flex-wrap gap-1.5">
      {items.map((it) => (
        <span
          key={it.label}
          className="inline-flex items-center gap-1.5 rounded-full bg-base-200/90 px-3 py-1 text-[11px] text-base-content/80 border border-base-300/60 shadow-sm"
        >
          <span className="text-base-content/40 font-semibold uppercase tracking-wide text-[10px]">
            {it.label}
          </span>
          <span className="font-mono-ui tabular-nums text-[11px] text-base-content/90">{it.value}</span>
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
          : "border-base-300/70 bg-base-100/80 text-base-content/70 hover:bg-base-100 hover:border-primary/25 hover:text-base-content"
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

function ChatTurn({
  turn,
  suppressRunningAssistant,
}: {
  turn: RunDetail;
  /** While the user just submitted, ProgressTimeline shows progress — hide duplicate "Thinking…". */
  suppressRunningAssistant?: boolean;
}) {
  const q = extractQuestion(turn);
  const disp = turnToDisplayed(turn);
  return (
    <div className="flex flex-col gap-3">
      {q ? <QuestionBubble text={q} /> : null}
      {turn.status === "running" && !suppressRunningAssistant && (
        <div className="flex flex-col animate-msg-enter">
          <RoleLabel role="Assistant" align="left" />
          <p className="text-sm text-base-content/60 pl-1 flex items-center gap-2">
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
            <p className="text-base-content/70">This turn failed.</p>
          )}
        </div>
      )}
      {turn.status === "completed" && (turn.answer?.length ?? 0) > 0 && (
        <div className="flex flex-col animate-msg-enter">
          <RoleLabel role="Assistant" align="left" />
          <article className="rounded-2xl border border-base-300/70 bg-gradient-to-b from-base-100 to-base-100/95 shadow-lg shadow-base-300/15 overflow-hidden ring-1 ring-base-content/[0.04]">
            <div className="h-1 bg-gradient-to-r from-primary/50 via-secondary/40 to-primary/30" aria-hidden />
            <div className="px-4 sm:px-5 pt-3.5 pb-3 flex flex-col sm:flex-row sm:items-start gap-3 sm:justify-between border-b border-base-200/80 bg-base-200/20">
              <MetaStrip run={disp} />
              <CopyButton text={disp.answer} />
            </div>
            <div className="px-4 sm:px-5 py-5">
              <AnswerBody text={disp.answer} />
              <Citations citations={disp.citations} />
            </div>
          </article>
        </div>
      )}
      {turn.status === "completed" && !(turn.answer?.length ?? 0) && (
        <p className="text-sm text-base-content/50 pl-1">No answer text for this turn.</p>
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
      <p className="text-sm text-base-content/55 mb-8 max-w-md leading-relaxed">
        Each sidebar chat is its own session: follow-ups keep prior turns in context. Open a chat to
        continue, or start fresh below.
      </p>
      <div className="flex flex-col gap-2.5 w-full max-w-lg text-left rounded-2xl border border-base-300/70 bg-base-100/70 p-4 shadow-md ring-1 ring-base-content/[0.03] backdrop-blur-sm">
        <p className="text-[10px] uppercase tracking-wider text-base-content/40 font-semibold px-0.5">
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
  const [form, setForm] = useState<AskFormState>(INITIAL_FORM);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<RunListItem[]>([]);
  const [activeThreadId, setActiveThreadId] = useState<string | null>(null);
  const [threadMessages, setThreadMessages] = useState<RunDetail[]>([]);
  const [threadListLoading, setThreadListLoading] = useState(false);
  const [optimisticUser, setOptimisticUser] = useState<string | null>(null);
  const [progress, setProgress] = useState<LiveProgress | null>(null);
  const [adapters, setAdapters] = useState<AdapterInfo[]>([]);
  const [adapterListHydrated, setAdapterListHydrated] = useState(false);
  const [adapterListError, setAdapterListError] = useState<string | null>(null);
  const progressTimer = useRef<number | null>(null);
  const threadRef = useRef<HTMLDivElement>(null);
  /** Invalidate stale thread / detail fetches when starting a new action. */
  const detailFetchGen = useRef(0);
  const composerRef = useRef<HTMLTextAreaElement>(null);
  /** Skip wiping transcript when re-opening the same thread from the sidebar. */
  const displayedThreadRef = useRef<string | null>(null);
  /** Count progress polls during an in-flight ask; refresh sidebar every N ticks. */
  const historyPollDuringProgressRef = useRef(0);

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
        const askables = r.adapters.filter((a) => a.capabilities.includes("ask"));
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
    setForm((f) => ({ ...f, question: "" }));
    window.requestAnimationFrame(() => {
      composerRef.current?.focus();
      threadRef.current?.scrollTo({ top: 0 });
    });
  }, []);

  const openThread = useCallback(async (tid: string) => {
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

      const startedAt = Date.now();
      const sinceIso = new Date(startedAt - 1000).toISOString();
      setProgress({
        startedAt,
        elapsed: 0,
        currentStage: "engine:structured",
        stagesDone: ["input:blocking", "enrich:(none)"],
      });
      historyPollDuringProgressRef.current = 0;
      const tick = async () => {
        historyPollDuringProgressRef.current += 1;
        if (historyPollDuringProgressRef.current % 6 === 0) {
          void loadHistory();
        }
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
        const r = await api.adapterAsk(form.adapter, {
          query: question,
          ...(activeThreadId ? { thread_id: activeThreadId } : {}),
        });
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

  const historyGroups = useMemo(() => groupHistoryByDay(history), [history]);

  useEffect(() => {
    const el = threadRef.current;
    if (!el) return;
    el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
  }, [threadMessages, submitting, optimisticUser, threadListLoading]);

  return (
    <div className="app-surface min-h-screen text-base-content">
      <div className="app-grain" aria-hidden />
      <div className="app-shell-content grid grid-cols-1 md:grid-cols-[minmax(17rem,20rem)_1fr] h-screen min-h-0">
        <aside className="min-h-0 flex flex-col border-b md:border-b-0 md:border-r border-base-300/90 bg-base-100/80 backdrop-blur-md">
          <div className="px-4 pt-4 pb-3 border-b border-base-300/80 bg-base-200/15">
            <div className="flex items-start gap-3">
              <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-gradient-to-br from-primary/25 to-secondary/10 text-base font-display font-semibold text-primary shadow-sm ring-1 ring-primary/20">
                ◈
              </div>
              <div className="min-w-0 pt-0.5">
                <p className="font-display text-lg font-semibold tracking-tight leading-tight">decomp</p>
                <p className="text-[11px] text-base-content/50 mt-0.5">Multi-repo Q&amp;A</p>
              </div>
            </div>
          </div>
          <div className="px-3 py-3 border-b border-base-300/60">
            <button
              type="button"
              onClick={startNew}
              className="btn btn-primary btn-sm w-full gap-2 rounded-xl"
            >
              <span className="text-lg leading-none">+</span>
              <span>New question</span>
            </button>
          </div>

          <div className="px-3 pt-3 pb-1 flex items-center justify-between">
            <h2 className="text-[10px] font-bold uppercase tracking-widest text-base-content/45">
              History
            </h2>
            <button
              type="button"
              onClick={loadHistory}
              className="btn btn-ghost btn-xs btn-square text-base-content/50"
              aria-label="Refresh history"
              title="Refresh now (also updates every 15s while this tab is visible)"
            >
              ↻
            </button>
          </div>

          <div className="flex-1 min-h-0 overflow-y-auto px-2 pb-4">
            {history.length === 0 ? (
              <p className="text-xs text-base-content/50 px-2 py-6 leading-relaxed">
                No runs yet. Ask a question in the composer — it will show up here.
              </p>
            ) : (
              <div className="flex flex-col gap-4">
                {historyGroups.map((g) => (
                  <div key={g.label}>
                    <p className="text-[10px] font-semibold uppercase tracking-wider text-base-content/40 px-1.5 mb-1.5">
                      {g.label}
                    </p>
                    <ul className="flex flex-col gap-1.5">
                      {g.items.map((r) => (
                        <li key={r.id}>
                          <HistoryItem
                            run={r}
                            selected={threadKey(r) === activeThreadId}
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

        <main className="min-h-0 flex flex-col h-full min-w-0 bg-base-200/25">
          <header className="shrink-0 border-b border-base-300/70 bg-base-100/45 backdrop-blur-md shadow-sm shadow-base-300/10 px-4 sm:px-6 py-3 sm:py-4">
            <h1 className="font-display text-xl sm:text-2xl font-semibold tracking-tight">
              Workspace
            </h1>
            <p className="text-xs text-base-content/50 mt-1 leading-relaxed max-w-2xl">
              Ask with follow-ups in one thread. History on the left keeps prior sessions.
            </p>
            <p className="text-[11px] text-base-content/45 mt-2 border-t border-base-300/50 pt-2">
              {activeThreadId
                ? "Follow-ups use this thread — prior turns are sent as context."
                : pickedAdapter
                  ? `Routing through ${pickedAdapter.name}.`
                  : "Pick an adapter before sending."}
            </p>
          </header>

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
                  <div className="rounded-2xl border border-base-300 bg-base-100/90 p-6 flex items-center gap-3 text-sm text-base-content/70 shadow-sm">
                    <span className="loading loading-spinner loading-sm text-primary" />
                    Loading chat…
                  </div>
                )}

                {threadMessages.map((turn) => (
                  <ChatTurn
                    key={turn.id}
                    turn={turn}
                    suppressRunningAssistant={submitting && turn.status === "running"}
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

          <div className="shrink-0 border-t border-base-300/80 bg-base-100/90 backdrop-blur-md px-4 sm:px-6 py-4">
            <form
              onSubmit={handleSubmit}
              className="max-w-3xl mx-auto w-full rounded-2xl bg-base-100 border border-base-300 shadow-md focus-within:border-primary/35 focus-within:shadow-md focus-within:ring-1 focus-within:ring-primary/15 transition-all"
            >
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
                className="w-full p-4 bg-transparent text-sm resize-y min-h-18 focus:outline-none placeholder:text-base-content/35"
              />
              <div className="flex items-center gap-3 flex-wrap px-3 py-2.5 border-t border-base-200/90 bg-base-200/20 rounded-b-2xl">
                {adapters.length > 1 && (
                  <label className="flex items-center gap-2 text-[11px] text-base-content/55 font-medium">
                    <span>Adapter</span>
                    <select
                      className="select select-bordered select-xs rounded-lg"
                      value={form.adapter}
                      onChange={(e) => setForm((f) => ({ ...f, adapter: e.target.value }))}
                      title="POST /v1/adapters/{name}/ask"
                    >
                      {adapters.map((a) => (
                        <option
                          key={a.name}
                          value={a.name}
                          title={a.health.ok ? a.description : (a.health.reason ?? "unhealthy")}
                        >
                          {a.name}
                          {a.health.ok ? "" : " ⚠"}
                        </option>
                      ))}
                    </select>
                  </label>
                )}
                <span className="text-[10px] text-base-content/40 hidden sm:inline ml-auto sm:ml-0">
                  ⌘↵ send
                </span>
                <button
                  type="submit"
                  className="btn btn-primary btn-sm rounded-xl gap-2 sm:ml-auto"
                  disabled={
                    submitting ||
                    !adapterListHydrated ||
                    !form.question.trim() ||
                    !pickedAdapter?.health.ok
                  }
                >
                  {submitting && <span className="loading loading-spinner loading-xs" />}
                  <span>{submitting ? "Asking…" : "Ask"}</span>
                </button>
              </div>
            </form>
          </div>
        </main>
      </div>
    </div>
  );
}
