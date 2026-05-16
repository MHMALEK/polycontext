import { useCallback, useEffect, useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { api } from "./api";
import type { AdapterInfo, BakeoffItem, Capability } from "./api";

type Job = "ask" | "decompose" | "implement";

const JOB_LABEL: Record<Job, string> = {
  ask: "Ask",
  decompose: "Decompose",
  implement: "Implement",
};

export function Bakeoff() {
  const [adapters, setAdapters] = useState<AdapterInfo[]>([]);
  const [job, setJob] = useState<Job>("ask");
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [query, setQuery] = useState("");
  const [decomposeQuery, setDecomposeQuery] = useState("");
  const [implementRepo, setImplementRepo] = useState("");
  const [implementText, setImplementText] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [results, setResults] = useState<BakeoffItem[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .listAdapters()
      .then((r) => setAdapters(r.adapters))
      .catch((e) => setError((e as Error).message));
  }, []);

  const eligible = adapters.filter((a) => a.capabilities.includes(job as Capability));

  const selectedAllHealthy = useMemo(() => {
    if (selected.size === 0) return false;
    return [...selected].every((name) =>
      Boolean(adapters.find((a) => a.name === name)?.health.ok),
    );
  }, [adapters, selected]);

  const toggle = useCallback((name: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });
  }, []);

  const handleSubmit = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault();
      if (selected.size === 0 || submitting) return;
      setSubmitting(true);
      setError(null);
      setResults([]);
      try {
        const body: Parameters<typeof api.bakeoff>[1] = { adapters: Array.from(selected) };
        if (job === "ask") body.ask = { query: query.trim() };
        if (job === "decompose")
          body.decompose = {
            query: decomposeQuery.trim() || undefined,
            mode: "auto",
          };
        if (job === "implement")
          body.implement = {
            repo: implementRepo.trim(),
            free_text: implementText.trim() || undefined,
            draft: true,
          };
        const r = await api.bakeoff(job, body);
        setResults(r.results);
      } catch (err) {
        setError((err as Error).message);
      } finally {
        setSubmitting(false);
      }
    },
    [job, selected, submitting, query, decomposeQuery, implementRepo, implementText],
  );

  return (
    <div className="p-8 max-w-7xl mx-auto">
      <header className="mb-6">
        <h1 className="text-2xl font-bold">Adapter bake-off</h1>
        <p className="text-sm text-base-content/60">
          Run the same input through multiple agent backends side-by-side.
        </p>
      </header>

      <form
        onSubmit={handleSubmit}
        className="card bg-base-100 border border-base-300 mb-6"
      >
        <div className="card-body gap-3">
          <div className="flex gap-2 flex-wrap">
            {(Object.keys(JOB_LABEL) as Job[]).map((j) => (
              <button
                key={j}
                type="button"
                className={`btn btn-sm ${job === j ? "btn-primary" : "btn-ghost"}`}
                onClick={() => {
                  setJob(j);
                  setSelected(new Set());
                  setResults([]);
                }}
              >
                {JOB_LABEL[j]}
              </button>
            ))}
          </div>

          {job === "ask" && (
            <textarea
              placeholder="Question about the codebase…"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              rows={3}
              className="textarea textarea-bordered w-full text-sm"
            />
          )}

          {job === "decompose" && (
            <div className="flex flex-col gap-2">
              <textarea
                placeholder="Query text"
                value={decomposeQuery}
                onChange={(e) => setDecomposeQuery(e.target.value)}
                rows={4}
                className="textarea textarea-bordered w-full text-sm"
              />
            </div>
          )}

          {job === "implement" && (
            <div className="flex flex-col gap-2">
              <input
                type="text"
                placeholder="Repo name (e.g. traceability)"
                value={implementRepo}
                onChange={(e) => setImplementRepo(e.target.value)}
                className="input input-bordered input-sm w-full"
              />
              <textarea
                placeholder="Task description"
                value={implementText}
                onChange={(e) => setImplementText(e.target.value)}
                rows={4}
                className="textarea textarea-bordered w-full text-sm"
              />
            </div>
          )}

          <div>
            <p className="text-xs text-base-content/60 mb-2">Adapters</p>
            <div className="flex gap-2 flex-wrap">
              {eligible.length === 0 && (
                <span className="text-sm text-base-content/50">
                  No adapter supports {job}. Install an optional dep or pick another job.
                </span>
              )}
              {eligible.map((a) => (
                <label
                  key={a.name}
                  className={`btn btn-sm ${
                    selected.has(a.name) ? "btn-secondary" : "btn-outline"
                  } ${a.health.ok ? "" : "opacity-60 ring-1 ring-warning/40"}`}
                  title={
                    a.health.ok
                      ? a.description
                      : `${a.description} — unhealthy: ${a.health.reason}`
                  }
                >
                  <input
                    type="checkbox"
                    className="sr-only"
                    checked={selected.has(a.name)}
                    onChange={() => toggle(a.name)}
                  />
                  {a.name}
                  {!a.health.ok ? " ⚠" : ""}
                </label>
              ))}
            </div>
          </div>

          <button
            type="submit"
            className="btn btn-primary btn-sm self-start"
            disabled={submitting || selected.size === 0 || !selectedAllHealthy}
          >
            {submitting && <span className="loading loading-spinner loading-xs" />}
            {submitting ? "Running…" : `Run on ${selected.size} adapter${selected.size === 1 ? "" : "s"}`}
          </button>
        </div>
      </form>

      {error && (
        <div className="alert alert-error mb-4 whitespace-pre-wrap font-mono text-xs">
          {error}
        </div>
      )}

      {results.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {results.map((r) => (
            <ResultCard key={r.adapter} item={r} job={job} />
          ))}
        </div>
      )}
    </div>
  );
}

function ResultCard({ item, job }: { item: BakeoffItem; job: Job }) {
  const m = item.result?.metrics;
  return (
    <article className="card bg-base-100 border border-base-300">
      <div className="card-body">
        <div className="flex items-center justify-between">
          <h3 className="font-mono font-semibold">{item.adapter}</h3>
          {item.ok ? (
            <span className="badge badge-success badge-sm">ok</span>
          ) : (
            <span className="badge badge-error badge-sm">{item.status ?? "err"}</span>
          )}
        </div>
        {m && (
          <div className="flex flex-wrap gap-x-3 gap-y-1 text-xs text-base-content/60 font-mono">
            {m.model && <span>{m.model}</span>}
            {m.duration_ms != null && <span>{(m.duration_ms / 1000).toFixed(1)}s</span>}
            {m.cost_usd != null && <span>${m.cost_usd.toFixed(4)}</span>}
            {m.tokens_in != null && <span>in {m.tokens_in}</span>}
            {m.tokens_out != null && <span>out {m.tokens_out}</span>}
            {m.tool_calls != null && m.tool_calls > 0 && <span>tools {m.tool_calls}</span>}
          </div>
        )}
        {!item.ok && (
          <pre className="whitespace-pre-wrap font-mono text-xs text-error">{item.error}</pre>
        )}
        {item.ok && job === "ask" && (
          <div className="prose prose-sm max-w-none">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {item.result?.answer || ""}
            </ReactMarkdown>
          </div>
        )}
        {item.ok && job === "decompose" && (
          <div className="prose prose-sm max-w-none">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {item.result?.markdown || ""}
            </ReactMarkdown>
          </div>
        )}
        {item.ok && job === "implement" && (
          <div className="flex flex-col gap-1 text-sm">
            {item.result?.mr_url ? (
              <a
                className="link link-primary truncate"
                href={item.result.mr_url}
                target="_blank"
                rel="noreferrer"
              >
                {item.result.mr_url}
              </a>
            ) : (
              <span className="text-base-content/60">branch: {item.result?.branch}</span>
            )}
            {item.result?.diff_summary && (
              <pre className="whitespace-pre-wrap font-mono text-xs bg-base-200 p-2 rounded">
                {item.result.diff_summary}
              </pre>
            )}
          </div>
        )}
      </div>
    </article>
  );
}
