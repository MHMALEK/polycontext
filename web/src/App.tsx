import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { api } from "./api";
import type {
  AdapterInfo,
  AskResponse,
  RunDetail,
  RunListItem,
} from "./api";
import { Bakeoff } from "./Bakeoff";

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

type View = "ask" | "bakeoff";

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

// ---------------------------------------------------------------------------
// Small components
// ---------------------------------------------------------------------------

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
      onClick={() => onSelect(run.id)}
      className={`group w-full text-left rounded-xl px-3 py-2.5 transition-all border outline-none focus-visible:ring-2 focus-visible:ring-primary/30 focus-visible:ring-offset-2 focus-visible:ring-offset-base-100 ${
        selected
          ? "bg-primary/8 border-primary/35 shadow-[inset_0_0_0_1px_rgba(0,0,0,0.04)]"
          : "bg-base-100/40 border-base-300/60 hover:bg-base-200/80 hover:border-base-300"
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
          <p className="text-[13px] leading-snug line-clamp-2 text-base-content font-medium">
            {historyTitle(run)}
          </p>
          <div className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-0.5 text-[11px] text-base-content/50">
            <span className="tabular-nums">{formatTimeAgo(run.created_at)}</span>
            {adapter && (
              <>
                <span aria-hidden className="text-base-content/30">
                  ·
                </span>
                <span className="font-mono text-[10px] uppercase tracking-wide text-base-content/45">
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
    <div className="rounded-xl border border-base-300 bg-base-100 p-5">
      <div className="flex items-center gap-3 mb-4">
        <span className="loading loading-spinner loading-sm text-primary" />
        <span className="text-sm font-medium">
          {prettyStage(progress.currentStage)}…
        </span>
        <span className="ml-auto text-xs text-base-content/50 font-mono tabular-nums">
          {progress.elapsed.toFixed(1)}s
        </span>
      </div>
      <ol className="flex flex-col gap-2">
        {PIPELINE_STAGES.map((s) => {
          const st = stageState(s.key, progress.stagesDone, progress.currentStage);
          return (
            <li key={s.key} className="flex items-center gap-3 text-sm">
              <span
                className={`inline-flex items-center justify-center w-5 h-5 rounded-full shrink-0 text-[10px] font-bold ${
                  st === "done"
                    ? "bg-success/15 text-success"
                    : st === "active"
                      ? "bg-primary/15 text-primary"
                      : "bg-base-200 text-base-content/30"
                }`}
              >
                {st === "done" ? "✓" : st === "active" ? "•" : ""}
              </span>
              <span
                className={
                  st === "pending"
                    ? "text-base-content/40"
                    : st === "active"
                      ? "text-base-content font-medium"
                      : "text-base-content/70"
                }
              >
                {s.label}
              </span>
            </li>
          );
        })}
      </ol>
      <p className="mt-4 text-xs text-base-content/50">
        Sourcebot questions usually take 10–60s while the model searches indexed
        repos and drafts a structured answer.
      </p>
    </div>
  );
}

function QuestionBubble({ text }: { text: string }) {
  if (!text) return null;
  return (
    <div className="flex justify-end">
      <div className="max-w-[min(100%,38rem)] rounded-2xl rounded-tr-md bg-primary text-primary-content px-4 py-3 text-sm whitespace-pre-wrap shadow-md shadow-primary/10">
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
    <details className="mt-6 rounded-lg border border-base-300 bg-base-100/60 group">
      <summary className="cursor-pointer select-none px-4 py-2.5 text-sm font-medium text-base-content/80 flex items-center gap-2 hover:bg-base-200/50 rounded-lg">
        <span className="text-base-content/60">📎</span>
        Sources <span className="text-xs text-base-content/50">({citations.length})</span>
      </summary>
      <ul className="px-4 pb-3 pt-1 flex flex-col gap-2 text-sm">
        {citations.map((c, i) => {
          const repo = (c.repo as string) || "";
          const path = (c.path as string) || "";
          const ls = c.line_start as number | undefined;
          const le = c.line_end as number | undefined;
          const snippet = (c.snippet as string) || "";
          const range = ls != null ? (le != null && le !== ls ? `:${ls}-${le}` : `:${ls}`) : "";
          return (
            <li key={i} className="border-l-2 border-base-300 pl-3">
              <code className="text-xs text-base-content/80">
                {repo ? `${repo} / ` : ""}
                {path}
                {range}
              </code>
              {snippet && (
                <pre className="mt-1 whitespace-pre-wrap text-[12px] leading-snug text-base-content/70 font-mono bg-base-200/60 rounded p-2 overflow-x-auto">
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
          className="inline-flex items-center gap-1 rounded-full bg-base-200/80 px-2.5 py-0.5 text-[11px] text-base-content/75 border border-base-300/50"
        >
          <span className="text-base-content/45 font-medium">{it.label}</span>
          <span className="font-mono tabular-nums text-[11px]">{it.value}</span>
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
      className="btn btn-ghost btn-xs gap-1.5"
      title="Copy answer"
    >
      <span>{copied ? "✓" : "📋"}</span>
      <span>{copied ? "Copied" : "Copy"}</span>
    </button>
  );
}

function EmptyState({ onPick }: { onPick: (q: string) => void }) {
  return (
    <div className="flex flex-col items-center justify-center text-center py-12 px-4">
      <div className="mb-5 flex h-14 w-14 items-center justify-center rounded-2xl bg-primary/10 text-2xl">
        ◈
      </div>
      <h2 className="font-display text-2xl sm:text-[1.65rem] font-semibold tracking-tight mb-2 text-base-content">
        Ask the codebase
      </h2>
      <p className="text-sm text-base-content/55 mb-8 max-w-md leading-relaxed">
        Questions route through your chosen adapter. You get grounded answers, citations when
        available, and full history on the left — click any past run to reopen it.
      </p>
      <div className="flex flex-col gap-2 w-full max-w-lg text-left">
        <p className="text-[10px] uppercase tracking-wider text-base-content/40 font-semibold px-1">
          Try
        </p>
        {SUGGESTED_PROMPTS.map((p) => (
          <button
            key={p}
            type="button"
            onClick={() => onPick(p)}
            className="text-left text-sm px-4 py-3 rounded-xl border border-base-300/80 bg-base-100/80 hover:bg-base-100 hover:border-primary/25 transition-colors shadow-sm"
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
  const [form, setForm] = useState<AskFormState>(INITIAL_FORM);
  const [submitting, setSubmitting] = useState(false);
  const [answer, setAnswer] = useState<AskResponse | null>(null);
  const [answeredQuestion, setAnsweredQuestion] = useState<string>("");
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<RunListItem[]>([]);
  const [selectedRun, setSelectedRun] = useState<string | null>(null);
  const [selectedDetail, setSelectedDetail] = useState<RunDetail | null>(null);
  const [selectedLoading, setSelectedLoading] = useState(false);
  const [progress, setProgress] = useState<LiveProgress | null>(null);
  const [adapters, setAdapters] = useState<AdapterInfo[]>([]);
  const [adapterListHydrated, setAdapterListHydrated] = useState(false);
  const [adapterListError, setAdapterListError] = useState<string | null>(null);
  const progressTimer = useRef<number | null>(null);
  const threadRef = useRef<HTMLDivElement>(null);

  const loadHistory = useCallback(async () => {
    try {
      const rows = await api.listRuns({ mode: "ask", limit: 50 });
      setHistory(rows);
    } catch (e) {
      console.warn("history fetch failed", e);
    }
  }, []);

  useEffect(() => {
    loadHistory();
    // Adapter list is fetched once at mount; status changes (e.g. starting
    // the cline-sdk-bridge sidecar) require a refresh anyway.
    api.listAdapters()
      .then((r) => {
        setAdapterListError(null);
        const askables = r.adapters.filter((a) => a.capabilities.includes("ask"));
        setAdapters(askables);
        // If the default adapter is missing or unhealthy, pick the first
        // healthy one so the user doesn't land on a broken selection.
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

  const startNew = useCallback(() => {
    setSelectedRun(null);
    setSelectedDetail(null);
    setAnswer(null);
    setAnsweredQuestion("");
    setError(null);
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
      setAnswer(null);
      setSelectedRun(null);
      setSelectedDetail(null);
      setAnsweredQuestion(question);

      const startedAt = Date.now();
      const sinceIso = new Date(startedAt - 1000).toISOString();
      setProgress({ startedAt, elapsed: 0, currentStage: null, stagesDone: [] });

      const tick = async () => {
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

      try {
        if (!form.adapter) {
          throw new Error("pick an adapter from the dropdown before submitting");
        }
        // Route every question through POST /v1/adapters/{name}/ask. The
        // adapter result is flattened into the AskResponse shape the
        // answer view already binds to.
        const r = await api.adapterAsk(form.adapter, { query: question });
        const ar = r.result;
        const res: AskResponse = {
          engine: ar.adapter,
          answer: ar.answer ?? "",
          citations: (ar.citations ?? []) as Array<Record<string, unknown>>,
          model: ar.metrics?.model ?? null,
          transport: null,
          wall_seconds: ar.metrics?.duration_ms != null
            ? ar.metrics.duration_ms / 1000
            : null,
          input_tokens: ar.metrics?.tokens_in ?? null,
          output_tokens: ar.metrics?.tokens_out ?? null,
          cost_usd: ar.metrics?.cost_usd ?? null,
          markdown_path: null,
          run_id: r.run_id,
        };
        setAnswer(res);
        setSelectedRun(res.run_id);
        setForm((f) => ({ ...f, question: "" }));
        await loadHistory();
      } catch (err) {
        setError((err as Error).message);
      } finally {
        if (progressTimer.current != null) {
          window.clearInterval(progressTimer.current);
          progressTimer.current = null;
        }
        setProgress(null);
        setSubmitting(false);
      }
    },
    [form, submitting, loadHistory, adapters, adapterListHydrated],
  );

  useEffect(() => {
    return () => {
      if (progressTimer.current != null) {
        window.clearInterval(progressTimer.current);
      }
    };
  }, []);

  const handleSelectRun = useCallback(async (id: string) => {
    setSelectedRun(id);
    setSelectedDetail(null);
    setSelectedLoading(true);
    setAnswer(null);
    setAnsweredQuestion("");
    setError(null);
    try {
      const detail = await api.getRun(id);
      setSelectedDetail(detail);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setSelectedLoading(false);
    }
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

  const displayed: DisplayedRun | null = useMemo(() => {
    if (answer) {
      return {
        question: answeredQuestion,
        answer: answer.answer,
        engine: answer.engine,
        model: answer.model,
        wall_seconds: answer.wall_seconds,
        cost_usd: answer.cost_usd,
        input_tokens: answer.input_tokens,
        output_tokens: answer.output_tokens,
        citations: answer.citations ?? [],
      };
    }
    if (selectedDetail && selectedDetail.answer != null) {
      return {
        question: extractQuestion(selectedDetail),
        answer: selectedDetail.answer,
        engine: selectedDetail.engine,
        model: selectedDetail.model,
        wall_seconds: selectedDetail.total_seconds,
        cost_usd: selectedDetail.total_cost_usd,
        input_tokens: selectedDetail.input_tokens,
        output_tokens: selectedDetail.output_tokens,
        citations: selectedDetail.citations ?? [],
      };
    }
    return null;
  }, [answer, answeredQuestion, selectedDetail]);

  const historyGroups = useMemo(() => groupHistoryByDay(history), [history]);

  useEffect(() => {
    threadRef.current?.scrollTo({ top: 0, behavior: "smooth" });
  }, [selectedRun, answer?.run_id]);

  if (view === "bakeoff") {
    return (
      <div className="min-h-screen bg-base-200 text-base-content">
        <div className="border-b border-base-300 bg-base-100/90 backdrop-blur-md px-4 py-3 flex items-center gap-4">
          <h1 className="font-display text-lg font-semibold tracking-tight flex-1">Bake-off</h1>
          <button type="button" className="btn btn-ghost btn-sm gap-1" onClick={() => setView("ask")}>
            ← Ask
          </button>
        </div>
        <Bakeoff />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-base-200 text-base-content relative">
      <div
        className="pointer-events-none absolute inset-0 opacity-[0.35] dark:opacity-25"
        style={{
          background:
            "radial-gradient(ellipse 90% 70% at 100% -10%, oklch(var(--p) / 0.14), transparent 55%), radial-gradient(ellipse 70% 50% at -10% 110%, oklch(var(--in) / 0.1), transparent 50%)",
        }}
      />
      <div className="relative grid grid-cols-1 md:grid-cols-[minmax(17rem,20rem)_1fr] h-screen min-h-0">
        <aside className="min-h-0 flex flex-col border-b md:border-b-0 md:border-r border-base-300 bg-base-100/85 backdrop-blur-md z-10">
          <div className="px-4 pt-4 pb-3 border-b border-base-300/80">
            <p className="font-display text-lg font-semibold tracking-tight leading-tight">decomp</p>
            <p className="text-[11px] text-base-content/50 mt-0.5">Multi-repo Q&amp;A</p>
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
              title="Refresh"
            >
              ↻
            </button>
          </div>

          <div className="flex-1 min-h-0 overflow-y-auto px-2 pb-4">
            {history.length === 0 ? (
              <p className="text-xs text-base-content/50 px-2 py-6 leading-relaxed">
                No runs yet. Use the composer below to ask your first question — it will appear here.
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
                            selected={r.id === selectedRun}
                            onSelect={handleSelectRun}
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

          <div className="shrink-0 p-3 border-t border-base-300/70">
            <button
              type="button"
              className="btn btn-outline btn-sm w-full rounded-xl border-base-300"
              onClick={() => setView("bakeoff")}
            >
              Adapter bake-off
            </button>
          </div>
        </aside>

        <main className="min-h-0 flex flex-col h-full min-w-0 bg-base-200/40">
          <header className="shrink-0 flex items-center justify-between gap-3 px-5 py-4 border-b border-base-300/70 bg-base-100/50 backdrop-blur-sm">
            <div>
              <h1 className="font-display text-xl sm:text-2xl font-semibold tracking-tight">
                Workspace
              </h1>
              <p className="text-xs text-base-content/50 mt-0.5">
                {pickedAdapter
                  ? `Routing questions through ${pickedAdapter.name}`
                  : "Pick an adapter when you send"}
              </p>
            </div>
          </header>

          <div
            ref={threadRef}
            className="flex-1 min-h-0 overflow-y-auto px-4 sm:px-6 py-6"
          >
            <div className="max-w-3xl mx-auto flex flex-col gap-5">
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

              <section className="flex flex-col gap-5">
                {submitting && progress && (
                  <>
                    {answeredQuestion && <QuestionBubble text={answeredQuestion} />}
                    <ProgressTimeline progress={progress} />
                  </>
                )}

                {selectedLoading && !submitting && (
                  <div className="rounded-2xl border border-base-300 bg-base-100/90 p-6 flex items-center gap-3 text-sm text-base-content/70 shadow-sm">
                    <span className="loading loading-spinner loading-sm text-primary" />
                    Opening conversation…
                  </div>
                )}

                {!submitting && displayed && (
                  <>
                    {displayed.question && <QuestionBubble text={displayed.question} />}
                    <article className="rounded-2xl border border-base-300/90 bg-base-100/95 shadow-lg shadow-base-300/20 overflow-hidden">
                      <div className="px-4 sm:px-5 pt-4 pb-3 flex flex-col sm:flex-row sm:items-start gap-3 sm:justify-between border-b border-base-200/90">
                        <MetaStrip run={displayed} />
                        <CopyButton text={displayed.answer} />
                      </div>
                      <div className="px-4 sm:px-5 py-5">
                        <AnswerBody text={displayed.answer} />
                        <Citations citations={displayed.citations} />
                      </div>
                    </article>
                  </>
                )}

                {!submitting && !displayed && selectedDetail && !selectedLoading && (
                  <div className="rounded-2xl border border-base-300 bg-base-100/95 p-5 text-sm shadow-sm">
                    <div className="flex items-center gap-2 mb-3">
                      <span
                        className={`inline-block w-2 h-2 rounded-full ${statusDotClass(
                          selectedDetail.status,
                        )}`}
                      />
                      <span className="text-base-content/70">
                        {selectedDetail.status === "failed" ? "Run failed" : "No answer on file"}
                      </span>
                    </div>
                    {extractQuestion(selectedDetail) && (
                      <QuestionBubble text={extractQuestion(selectedDetail)} />
                    )}
                    {selectedDetail.error ? (
                      <pre className="mt-3 text-xs text-error font-mono whitespace-pre-wrap bg-error/5 rounded-xl p-3 border border-error/20">
                        {selectedDetail.error}
                      </pre>
                    ) : (
                      <p className="text-base-content/55 text-sm">
                        This run has no saved response
                        {selectedDetail.status === "completed" ? " (legacy run or empty reply)." : "."}
                      </p>
                    )}
                  </div>
                )}

                {!submitting && !displayed && !selectedDetail && !selectedLoading && (
                  <EmptyState onPick={(q) => setForm((f) => ({ ...f, question: q }))} />
                )}
              </section>
            </div>
          </div>

          <div className="shrink-0 border-t border-base-300/80 bg-base-100/90 backdrop-blur-md px-4 sm:px-6 py-4">
            <form
              onSubmit={handleSubmit}
              className="max-w-3xl mx-auto rounded-2xl bg-base-100 border border-base-300 shadow-md focus-within:border-primary/35 focus-within:shadow-md focus-within:ring-1 focus-within:ring-primary/15 transition-all"
            >
              <textarea
                placeholder="Ask about architecture, flows, or where something lives…"
                value={form.question}
                onChange={(e) => setForm({ ...form, question: e.target.value })}
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
                      onChange={(e) => setForm({ ...form, adapter: e.target.value })}
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
