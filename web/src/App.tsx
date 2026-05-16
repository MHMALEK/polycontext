import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { api } from "./api";
import type {
  AdapterInfo,
  AskResponse,
  OutputFormat,
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
  format: OutputFormat;
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
  format: OutputFormat;
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
  format: "markdown",
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
      if (s.includes("local")) return "Running local agent";
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
  return detail.input_preview ?? "";
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
  return (
    <button
      type="button"
      onClick={() => onSelect(run.id)}
      className={`group w-full text-left rounded-lg px-3 py-2.5 transition-colors border ${
        selected
          ? "bg-primary/10 border-primary/30"
          : "bg-transparent border-transparent hover:bg-base-200 hover:border-base-300"
      }`}
    >
      <div className="flex items-center gap-2 mb-1">
        <span
          className={`inline-block w-2 h-2 rounded-full shrink-0 ${statusDotClass(run.status)}`}
          aria-label={run.status}
        />
        <span className="text-[11px] uppercase tracking-wide text-base-content/50 font-medium">
          {run.status}
        </span>
        <span className="ml-auto text-[11px] text-base-content/50">
          {formatTimeAgo(run.created_at)}
        </span>
      </div>
      <p className="text-sm leading-snug line-clamp-2 text-base-content/90">
        {run.input_preview || run.id}
      </p>
      <div className="mt-1.5 flex items-center gap-2 text-[11px] text-base-content/55">
        {run.total_seconds != null && <span>{formatWall(run.total_seconds)}</span>}
        {run.total_cost_usd != null && <span>· {formatCost(run.total_cost_usd)}</span>}
        <span
          role="button"
          tabIndex={0}
          onClick={(e) => {
            e.stopPropagation();
            onReplay(run.id);
          }}
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === " ") {
              e.preventDefault();
              e.stopPropagation();
              onReplay(run.id);
            }
          }}
          className="ml-auto opacity-0 group-hover:opacity-100 transition-opacity text-primary hover:underline cursor-pointer"
          title="Re-run this question"
        >
          ↻ replay
        </span>
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
      <div className="max-w-[80%] rounded-2xl rounded-tr-sm bg-primary text-primary-content px-4 py-3 text-sm whitespace-pre-wrap shadow-sm">
        {text}
      </div>
    </div>
  );
}

function AnswerBody({ text, format }: { text: string; format: OutputFormat }) {
  if (format === "html") {
    return (
      <div
        className="prose prose-sm sm:prose-base max-w-none"
        dangerouslySetInnerHTML={{ __html: text }}
      />
    );
  }
  if (format === "text") {
    return (
      <pre className="whitespace-pre-wrap font-mono text-sm leading-relaxed">{text}</pre>
    );
  }
  return (
    <div className="prose prose-sm sm:prose-base max-w-none prose-pre:bg-base-200 prose-pre:text-base-content prose-code:before:content-none prose-code:after:content-none prose-code:bg-base-200 prose-code:px-1 prose-code:py-0.5 prose-code:rounded">
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
  if (run.engine) items.push({ label: "engine", value: run.engine });
  if (run.model) items.push({ label: "model", value: run.model });
  if (run.wall_seconds != null)
    items.push({ label: "wall", value: formatWall(run.wall_seconds) });
  if (run.cost_usd != null) items.push({ label: "cost", value: formatCost(run.cost_usd) });
  if (run.input_tokens != null || run.output_tokens != null) {
    items.push({
      label: "tokens",
      value: `${run.input_tokens ?? 0} in / ${run.output_tokens ?? 0} out`,
    });
  }
  if (!items.length) return null;
  return (
    <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-base-content/55">
      {items.map((it) => (
        <span key={it.label}>
          <span className="text-base-content/40">{it.label}:</span>{" "}
          <span className="font-mono text-base-content/70">{it.value}</span>
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
    <div className="flex flex-col items-center justify-center text-center py-16 px-4">
      <div className="text-5xl mb-4 opacity-70">💬</div>
      <h2 className="text-xl font-semibold mb-2">Ask anything about the codebase</h2>
      <p className="text-sm text-base-content/60 mb-6 max-w-md">
        Pose a question and the agent will search the indexed repositories, read
        the relevant code, and draft a grounded answer with sources.
      </p>
      <div className="flex flex-col gap-2 w-full max-w-md">
        {SUGGESTED_PROMPTS.map((p) => (
          <button
            key={p}
            type="button"
            onClick={() => onPick(p)}
            className="text-left text-sm px-4 py-2.5 rounded-lg border border-base-300 bg-base-100 hover:bg-base-200 transition-colors"
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
        format: form.format,
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
      const fmt = (selectedDetail.output_format as OutputFormat) || "markdown";
      return {
        question: extractQuestion(selectedDetail),
        answer: selectedDetail.answer,
        format: fmt,
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
  }, [answer, answeredQuestion, form.format, selectedDetail]);

  const historyGroups = useMemo(() => groupHistoryByDay(history), [history]);

  if (view === "bakeoff") {
    return (
      <div className="min-h-screen bg-base-200 text-base-content">
        <div className="navbar bg-base-100 border-b border-base-300 px-4">
          <h1 className="text-lg font-bold flex-1">tech-decomposition</h1>
          <button
            type="button"
            className="btn btn-ghost btn-sm"
            onClick={() => setView("ask")}
          >
            ← Ask
          </button>
        </div>
        <Bakeoff />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-base-200 text-base-content">
      <div className="grid grid-cols-1 md:grid-cols-[300px_1fr] h-screen">
        {/* ── Sidebar ─────────────────────────────────────────────────────── */}
        <aside className="bg-base-100 border-r border-base-300 flex flex-col overflow-hidden">
          <div className="px-4 py-4 border-b border-base-300">
            <button
              type="button"
              onClick={startNew}
              className="btn btn-primary btn-sm w-full gap-2"
            >
              <span>＋</span>
              <span>New question</span>
            </button>
          </div>

          <div className="px-4 pt-4 pb-2 flex items-center justify-between">
            <h2 className="text-[11px] font-semibold uppercase tracking-wider text-base-content/50">
              History
            </h2>
            <button
              type="button"
              onClick={loadHistory}
              className="btn btn-ghost btn-xs text-base-content/60"
              aria-label="refresh history"
              title="Refresh"
            >
              ↻
            </button>
          </div>

          <div className="flex-1 overflow-y-auto px-2 pb-4">
            {history.length === 0 ? (
              <p className="text-sm text-base-content/50 px-2 py-4">
                No prior runs yet. Ask a question to get started.
              </p>
            ) : (
              <div className="flex flex-col gap-4">
                {historyGroups.map((g) => (
                  <div key={g.label}>
                    <p className="text-[10px] uppercase tracking-wider text-base-content/40 px-2 mb-1">
                      {g.label}
                    </p>
                    <ul className="flex flex-col gap-0.5">
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
        </aside>

        {/* ── Main ────────────────────────────────────────────────────────── */}
        <main className="overflow-y-auto">
          <div className="max-w-3xl mx-auto px-6 py-8">
            <header className="mb-6 flex items-baseline justify-between gap-4">
              <div>
                <h1 className="text-xl font-semibold">tech-decomposition</h1>
                <p className="text-sm text-base-content/60">multi-repo code Q&amp;A</p>
              </div>
            </header>

            {adapterListHydrated &&
              !adapterListError &&
              adapters.length === 0 && (
                <div className="alert alert-warning mb-4 text-xs">
                  No adapters with <code>ask</code> capability were returned. Check API configuration
                  (e.g. <code>ENABLED_ADAPTERS</code>) or server logs.
                </div>
              )}

            {adapterListError && (
              <div className="alert alert-warning mb-4 text-xs font-mono whitespace-pre-wrap">
                Could not load adapter list (/v1/adapters): {adapterListError}
              </div>
            )}
            {!adapterListError &&
              adapterListHydrated &&
              pickedAdapter &&
              !pickedAdapter.health.ok && (
                <div className="alert alert-warning mb-4 text-xs">
                  Selected adapter <strong>{pickedAdapter.name}</strong> reports unhealthy —
                  Ask stays disabled until it passes health ({pickedAdapter.health.reason ?? "reason unknown"}
                  ).
                </div>
              )}

            {/* Ask form */}
            <form
              onSubmit={handleSubmit}
              className="rounded-2xl bg-base-100 border border-base-300 shadow-sm mb-6 focus-within:border-primary/40 focus-within:shadow transition-all"
            >
              <textarea
                placeholder="Ask a question about the codebase…"
                value={form.question}
                onChange={(e) => setForm({ ...form, question: e.target.value })}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
                    e.preventDefault();
                    void handleSubmit(e as unknown as React.FormEvent);
                  }
                }}
                rows={3}
                className="w-full p-4 bg-transparent text-sm resize-y focus:outline-none placeholder:text-base-content/40"
              />
              <div className="flex items-center gap-3 flex-wrap px-3 py-2 border-t border-base-200">
                {adapters.length > 1 && (
                  <label className="flex items-center gap-1.5 text-xs text-base-content/60">
                    <span>adapter</span>
                    <select
                      className="select select-bordered select-xs"
                      value={form.adapter}
                      onChange={(e) => setForm({ ...form, adapter: e.target.value })}
                      title="Routes the question through POST /v1/adapters/{name}/ask."
                    >
                      {adapters.map((a) => (
                        <option
                          key={a.name}
                          value={a.name}
                          title={a.health.ok ? a.description : (a.health.reason ?? "unhealthy")}
                        >
                          {a.name}
                          {a.health.ok ? "" : " (unhealthy)"}
                        </option>
                      ))}
                    </select>
                  </label>
                )}
                <span className="text-[10px] text-base-content/40 hidden sm:inline">
                  ⌘+↵ to send
                </span>
                <button
                  type="submit"
                  className="btn btn-primary btn-sm ml-auto gap-2"
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

            {/* Error */}
            {error && (
              <div className="alert alert-error mb-4 text-xs font-mono whitespace-pre-wrap">
                <div className="flex-1">{error}</div>
                <button
                  type="button"
                  onClick={() => setError(null)}
                  className="btn btn-ghost btn-xs"
                  aria-label="dismiss"
                >
                  ✕
                </button>
              </div>
            )}

            {/* Conversation area */}
            <section className="flex flex-col gap-4">
              {/* While submitting a NEW question */}
              {submitting && progress && (
                <>
                  {answeredQuestion && <QuestionBubble text={answeredQuestion} />}
                  <ProgressTimeline progress={progress} />
                </>
              )}

              {/* While loading a clicked history item */}
              {selectedLoading && !submitting && (
                <div className="rounded-xl border border-base-300 bg-base-100 p-5 flex items-center gap-3 text-sm text-base-content/70">
                  <span className="loading loading-spinner loading-sm" />
                  Loading run…
                </div>
              )}

              {/* Displayed answer (fresh or historical) */}
              {!submitting && displayed && (
                <>
                  {displayed.question && <QuestionBubble text={displayed.question} />}
                  <article className="rounded-2xl border border-base-300 bg-base-100 shadow-sm">
                    <div className="px-5 pt-4 pb-3 flex items-center justify-between gap-3 border-b border-base-200">
                      <MetaStrip run={displayed} />
                      <CopyButton text={displayed.answer} />
                    </div>
                    <div className="px-5 py-5">
                      <AnswerBody text={displayed.answer} format={displayed.format} />
                      <Citations citations={displayed.citations} />
                    </div>
                  </article>
                </>
              )}

              {/* Error on selected run with no answer (e.g., failed run) */}
              {!submitting && !displayed && selectedDetail && !selectedLoading && (
                <div className="rounded-xl border border-base-300 bg-base-100 p-5 text-sm">
                  <div className="flex items-center gap-2 mb-2">
                    <span
                      className={`inline-block w-2 h-2 rounded-full ${statusDotClass(selectedDetail.status)}`}
                    />
                    <span className="text-base-content/60">
                      Run {selectedDetail.id.slice(0, 8)} · {selectedDetail.status}
                    </span>
                  </div>
                  {selectedDetail.error ? (
                    <pre className="text-xs text-error font-mono whitespace-pre-wrap">
                      {selectedDetail.error}
                    </pre>
                  ) : (
                    <p className="text-base-content/60">
                      No answer recorded for this run.
                    </p>
                  )}
                </div>
              )}

              {/* Empty state */}
              {!submitting && !displayed && !selectedDetail && !selectedLoading && (
                <EmptyState onPick={(q) => setForm((f) => ({ ...f, question: q }))} />
              )}
            </section>
          </div>
        </main>
      </div>
    </div>
  );
}
