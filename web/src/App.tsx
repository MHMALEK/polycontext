import { useCallback, useEffect, useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { api } from "./api";
import type { AskEngine, AskResponse, OutputFormat, RunListItem } from "./api";

type AskFormState = {
  question: string;
  engine: AskEngine;
  format: OutputFormat;
};

const INITIAL: AskFormState = { question: "", engine: "sourcebot", format: "markdown" };

function statusBadgeClass(status: string): string {
  if (status === "completed") return "badge badge-success badge-sm";
  if (status === "failed") return "badge badge-error badge-sm";
  if (status === "running") return "badge badge-warning badge-sm";
  return "badge badge-ghost badge-sm";
}

export function App() {
  const [form, setForm] = useState<AskFormState>(INITIAL);
  const [submitting, setSubmitting] = useState(false);
  const [answer, setAnswer] = useState<AskResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<RunListItem[]>([]);
  const [selectedRun, setSelectedRun] = useState<string | null>(null);
  const [historyAnswer, setHistoryAnswer] = useState<string | null>(null);

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
        const res = await api.ask({
          question: form.question.trim(),
          engine: form.engine,
          format: form.format,
        });
        setAnswer(res);
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
          <header className="mb-6">
            <h1 className="text-2xl font-bold">tech-decomposition</h1>
            <p className="text-sm text-base-content/60">multi-repo code Q&amp;A</p>
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
