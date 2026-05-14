import { useCallback, useEffect, useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { api } from "./api";
import type { AdapterInfo, AskEngine, AskResponse, OutputFormat, RunListItem } from "./api";
import { Bakeoff } from "./Bakeoff";

type AskFormState = {
  question: string;
  engine: AskEngine;
  format: OutputFormat;
  // Empty string = use the legacy /ask endpoint (which is baseline-Sourcebot
  // with the structurer wrapper). Any other value routes to
  // POST /v1/adapters/{adapter}/ask so we can drive a single named adapter
  // from the main view without leaving the page.
  adapter: string;
};

const INITIAL: AskFormState = {
  question: "",
  engine: "sourcebot",
  format: "markdown",
  adapter: "",
};

function statusBadgeClass(status: string): string {
  if (status === "completed") return "badge badge-success badge-sm";
  if (status === "failed") return "badge badge-error badge-sm";
  if (status === "running") return "badge badge-warning badge-sm";
  return "badge badge-ghost badge-sm";
}

type View = "ask" | "bakeoff";

export function App() {
  const [view, setView] = useState<View>("ask");
  const [form, setForm] = useState<AskFormState>(INITIAL);
  const [submitting, setSubmitting] = useState(false);
  const [answer, setAnswer] = useState<AskResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<RunListItem[]>([]);
  const [selectedRun, setSelectedRun] = useState<string | null>(null);
  const [historyAnswer, setHistoryAnswer] = useState<string | null>(null);
  const [adapters, setAdapters] = useState<AdapterInfo[]>([]);

  const loadHistory = useCallback(async () => {
    try {
      const rows = await api.listRuns({ mode: "ask", limit: 25 });
      setHistory(rows);
    } catch (e) {
      console.warn("history fetch failed", e);
    }
  }, []);

  useEffect(() => {
    loadHistory();
    // Adapter list is fetched once at mount; status changes (e.g. starting
    // the OpenHands container) require a refresh anyway.
    api.listAdapters()
      .then((r) => setAdapters(r.adapters.filter((a) => a.capabilities.includes("ask"))))
      .catch((e) => console.warn("adapter list fetch failed", e));
  }, [loadHistory]);

  const handleSubmit = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault();
      if (!form.question.trim() || submitting) return;
      setSubmitting(true);
      setError(null);
      setAnswer(null);
      setHistoryAnswer(null);
      setSelectedRun(null);
      try {
        if (form.adapter) {
          // Adapter path: hit /v1/adapters/{name}/ask. Adapter results have
          // a different shape than legacy /ask, so we map back into the
          // AskResponse the rest of this view consumes.
          const r = await api.bakeoff("ask", {
            adapters: [form.adapter],
            ask: { query: form.question.trim() },
          });
          const item = r.results[0];
          if (!item.ok) throw new Error(item.error || `adapter ${form.adapter} failed`);
          const res = item.result!;
          setAnswer({
            engine: res.adapter,
            answer: res.answer ?? "",
            citations: (res.citations ?? []) as Array<Record<string, unknown>>,
            model: res.metrics?.model ?? null,
            transport: null,
            wall_seconds: res.metrics?.duration_ms != null
              ? res.metrics.duration_ms / 1000
              : null,
            input_tokens: res.metrics?.tokens_in ?? null,
            output_tokens: res.metrics?.tokens_out ?? null,
            cost_usd: res.metrics?.cost_usd ?? null,
            markdown_path: null,
            run_id: "(adapter-call)",
          });
        } else {
          // Legacy /ask path — baseline-Sourcebot with the structurer wrapper.
          const res = await api.ask({
            question: form.question.trim(),
            engine: form.engine,
            format: form.format,
          });
          setAnswer(res);
        }
        await loadHistory();
      } catch (err) {
        setError((err as Error).message);
      } finally {
        setSubmitting(false);
      }
    },
    [form, submitting, loadHistory],
  );

  const handleSelectRun = useCallback(async (id: string) => {
    setSelectedRun(id);
    setAnswer(null);
    setError(null);
    try {
      const run = await api.getRun(id);
      setHistoryAnswer(run.answer ?? "(empty)");
    } catch (err) {
      setError((err as Error).message);
    }
  }, []);

  const handleReplay = useCallback(
    async (id: string) => {
      try {
        await api.replayRun(id);
      } catch (err) {
        setError((err as Error).message);
        return;
      }
      // Replay triggers a new run on the backend. Reload history a moment
      // later so the new entry shows up at the top.
      setTimeout(loadHistory, 500);
    },
    [loadHistory],
  );

  const activeAnswer = useMemo(() => {
    if (answer) return { body: answer.answer, format: form.format };
    if (historyAnswer) return { body: historyAnswer, format: "markdown" as OutputFormat };
    return null;
  }, [answer, historyAnswer, form.format]);

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
      <div className="grid grid-cols-1 md:grid-cols-[320px_1fr] h-screen">
        {/* Sidebar */}
        <aside className="bg-base-100 border-r border-base-300 overflow-y-auto p-4 flex flex-col gap-3">
          <div className="flex items-center justify-between">
            <h2 className="text-xs font-semibold uppercase tracking-wider text-base-content/60">
              History
            </h2>
            <button
              type="button"
              onClick={loadHistory}
              className="btn btn-ghost btn-xs"
              aria-label="refresh history"
            >
              ↻
            </button>
          </div>

          {history.length === 0 ? (
            <p className="text-sm text-base-content/50">No prior runs.</p>
          ) : (
            <ul className="flex flex-col gap-2">
              {history.map((r) => (
                <li
                  key={r.id}
                  className={`card card-compact bg-base-100 border ${
                    r.id === selectedRun ? "border-primary" : "border-base-300"
                  } cursor-pointer hover:bg-base-200 transition-colors group relative`}
                  onClick={() => handleSelectRun(r.id)}
                >
                  <div className="card-body">
                    <div className="flex items-center gap-2 text-xs">
                      <span className={statusBadgeClass(r.status)}>{r.status}</span>
                      <span className="text-base-content/60 font-mono">
                        {r.engine ?? "?"}
                      </span>
                      {r.total_cost_usd != null && (
                        <span className="ml-auto text-base-content/60 font-mono">
                          ${r.total_cost_usd.toFixed(4)}
                        </span>
                      )}
                    </div>
                    <p className="text-sm font-medium line-clamp-2">
                      {r.input_preview || r.id}
                    </p>
                    <p className="text-xs text-base-content/50">
                      {r.created_at.slice(0, 19).replace("T", " ")}
                      {r.total_seconds != null && ` · ${r.total_seconds.toFixed(1)}s`}
                    </p>
                  </div>
                  <button
                    type="button"
                    className="btn btn-xs btn-primary absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity"
                    onClick={(ev) => {
                      ev.stopPropagation();
                      handleReplay(r.id);
                    }}
                    title="Re-run this request"
                  >
                    replay
                  </button>
                </li>
              ))}
            </ul>
          )}
        </aside>

        {/* Main */}
        <main className="overflow-y-auto p-8 max-w-5xl">
          <header className="mb-6 flex items-start justify-between gap-4">
            <div>
              <h1 className="text-2xl font-bold">tech-decomposition</h1>
              <p className="text-sm text-base-content/60">multi-repo code Q&amp;A</p>
            </div>
            <button
              type="button"
              className="btn btn-sm btn-outline"
              onClick={() => setView("bakeoff")}
            >
              Adapter bake-off →
            </button>
          </header>

          <form
            onSubmit={handleSubmit}
            className="card bg-base-100 border border-base-300 mb-6"
          >
            <div className="card-body gap-2">
              <textarea
                placeholder="Ask a question about the codebase..."
                value={form.question}
                onChange={(e) => setForm({ ...form, question: e.target.value })}
                rows={4}
                className="textarea textarea-ghost w-full p-0 text-sm focus:outline-none resize-y"
              />
              <div className="flex items-center gap-3 flex-wrap pt-2 border-t border-base-200">
                <label className="form-control">
                  <span className="label-text text-xs mr-1">adapter</span>
                  <select
                    className="select select-bordered select-xs"
                    value={form.adapter}
                    onChange={(e) => setForm({ ...form, adapter: e.target.value })}
                    title="Empty = legacy /ask (baseline). Any other adapter routes through /v1/adapters/{name}/ask."
                  >
                    <option value="">default (baseline)</option>
                    {adapters.map((a) => (
                      <option
                        key={a.name}
                        value={a.name}
                        disabled={!a.health.ok}
                        title={a.health.ok ? a.description : a.health.reason}
                      >
                        {a.name}{a.health.ok ? "" : " (down)"}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="form-control">
                  <span className="label-text text-xs mr-1">engine</span>
                  <select
                    className="select select-bordered select-xs"
                    value={form.engine}
                    onChange={(e) => setForm({ ...form, engine: e.target.value as AskEngine })}
                  >
                    <option value="sourcebot">sourcebot</option>
                    <option value="local">local (experimental)</option>
                  </select>
                </label>
                <label className="form-control">
                  <span className="label-text text-xs mr-1">format</span>
                  <select
                    className="select select-bordered select-xs"
                    value={form.format}
                    onChange={(e) =>
                      setForm({ ...form, format: e.target.value as OutputFormat })
                    }
                  >
                    <option value="markdown">markdown</option>
                    <option value="html">html</option>
                    <option value="text">text</option>
                  </select>
                </label>
                <button
                  type="submit"
                  className="btn btn-primary btn-sm ml-auto"
                  disabled={submitting || !form.question.trim()}
                >
                  {submitting && <span className="loading loading-spinner loading-xs" />}
                  {submitting ? "Asking…" : "Ask"}
                </button>
              </div>
            </div>
          </form>

          {error && (
            <div className="alert alert-error mb-4 whitespace-pre-wrap font-mono text-xs">
              {error}
            </div>
          )}

          {answer && (
            <div className="flex flex-wrap gap-4 text-xs text-base-content/60 mb-3">
              <span>
                engine: <code className="bg-base-200 px-1.5 rounded">{answer.engine}</code>
              </span>
              {answer.model && (
                <span>
                  model: <code className="bg-base-200 px-1.5 rounded">{answer.model}</code>
                </span>
              )}
              {answer.wall_seconds != null && <span>wall: {answer.wall_seconds}s</span>}
              {answer.cost_usd != null && <span>cost: ${answer.cost_usd.toFixed(4)}</span>}
              {answer.citations.length > 0 && (
                <span>citations: {answer.citations.length}</span>
              )}
            </div>
          )}

          <article className="card bg-base-100 border border-base-300">
            <div className="card-body">
              {activeAnswer == null ? (
                <p className="text-base-content/50">
                  {submitting
                    ? "Working…"
                    : "Ask a question, or pick a prior run from the sidebar."}
                </p>
              ) : activeAnswer.format === "html" ? (
                <div
                  className="prose max-w-none"
                  dangerouslySetInnerHTML={{ __html: activeAnswer.body }}
                />
              ) : activeAnswer.format === "text" ? (
                <pre className="whitespace-pre-wrap font-mono text-sm">{activeAnswer.body}</pre>
              ) : (
                <div className="prose max-w-none">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>
                    {activeAnswer.body}
                  </ReactMarkdown>
                </div>
              )}
            </div>
          </article>
        </main>
      </div>
    </div>
  );
}
