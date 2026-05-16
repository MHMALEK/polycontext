import { useEffect, useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { api } from "./api";
import type {
  AdapterDecomposeResponse,
  AdapterInfo,
  Decomposition,
  Subtask,
} from "./api";

const DEFAULT_ADAPTER = "sourcebot";

const SUGGESTED = [
  "Pre-fill supplier questionnaire on behalf of supplier — what changes across repos?",
  "Add a new role 'DATA_REVIEWER' with audit permissions",
  "Migrate the auth flow from session cookies to JWT",
];

function complexityBadge(c: Subtask["estimated_complexity"]) {
  const cls = {
    small: "badge-success",
    medium: "badge-info",
    large: "badge-warning",
    unknown: "badge-ghost",
  }[c] ?? "badge-ghost";
  return <span className={`badge badge-sm ${cls}`}>{c}</span>;
}

function SubtaskCard({ st, idx }: { st: Subtask; idx: number }) {
  return (
    <article className="rounded-xl border border-base-300 bg-base-100 p-4 sm:p-5 shadow-sm">
      <header className="flex items-start gap-3 mb-2">
        <span className="text-xs font-mono text-base-content/50 mt-1 tabular-nums">
          {String(idx + 1).padStart(2, "0")}
        </span>
        <div className="flex-1 min-w-0">
          <h3 className="font-semibold text-base leading-snug">{st.title}</h3>
          <div className="mt-1 flex flex-wrap gap-1.5 items-center text-[11px]">
            {st.repo && (
              <span className="badge badge-sm badge-outline font-mono">{st.repo}</span>
            )}
            {complexityBadge(st.estimated_complexity)}
          </div>
        </div>
      </header>
      <p className="text-sm text-base-content/85 mt-2 leading-relaxed whitespace-pre-wrap">
        {st.description}
      </p>
      {st.files.length > 0 && (
        <div className="mt-3">
          <p className="text-[10px] uppercase tracking-wide text-base-content/55 mb-1">
            Files
          </p>
          <ul className="flex flex-wrap gap-1.5">
            {st.files.map((f, i) => (
              <li key={i}>
                {st.file_links[i] ? (
                  <a
                    href={st.file_links[i]}
                    target="_blank"
                    rel="noreferrer"
                    className="link link-hover font-mono text-[11px] bg-base-200 px-2 py-0.5 rounded"
                  >
                    {f}
                  </a>
                ) : (
                  <code className="font-mono text-[11px] bg-base-200 px-2 py-0.5 rounded">
                    {f}
                  </code>
                )}
              </li>
            ))}
          </ul>
        </div>
      )}
      {st.acceptance_criteria.length > 0 && (
        <div className="mt-3">
          <p className="text-[10px] uppercase tracking-wide text-base-content/55 mb-1">
            Acceptance criteria
          </p>
          <ul className="list-disc list-inside text-sm text-base-content/80 space-y-0.5">
            {st.acceptance_criteria.map((ac, i) => (
              <li key={i}>{ac}</li>
            ))}
          </ul>
        </div>
      )}
    </article>
  );
}

function ListBlock({
  title,
  items,
  emptyHint,
}: {
  title: string;
  items: string[];
  emptyHint?: string;
}) {
  if (items.length === 0) {
    return emptyHint ? (
      <p className="text-xs text-base-content/55 italic">{emptyHint}</p>
    ) : null;
  }
  return (
    <details className="rounded-lg border border-base-300 bg-base-100/70" open>
      <summary className="cursor-pointer select-none px-4 py-2 text-sm font-medium">
        {title} <span className="text-base-content/50">({items.length})</span>
      </summary>
      <ul className="px-5 pb-3 list-disc text-sm space-y-1">
        {items.map((x, i) => (
          <li key={i} className="text-base-content/85">{x}</li>
        ))}
      </ul>
    </details>
  );
}

function DecompositionView({ d, markdown }: { d: Decomposition; markdown: string }) {
  return (
    <div className="space-y-6">
      <section>
        <h2 className="text-lg font-semibold mb-2">Overview</h2>
        <div className="prose prose-sm max-w-none">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{d.overview || "_(empty)_"}</ReactMarkdown>
        </div>
        {d.affected_repos.length > 0 && (
          <div className="mt-3 flex flex-wrap gap-1.5">
            {d.affected_repos.map((r) => (
              <span key={r} className="badge badge-sm badge-outline font-mono">{r}</span>
            ))}
          </div>
        )}
      </section>

      <section>
        <h2 className="text-lg font-semibold mb-3">
          Subtasks <span className="text-base-content/50 text-sm">({d.subtasks.length})</span>
        </h2>
        {d.subtasks.length === 0 ? (
          <p className="text-sm text-base-content/55 italic">No subtasks produced.</p>
        ) : (
          <div className="grid gap-3">
            {d.subtasks.map((st, i) => (
              <SubtaskCard key={i} st={st} idx={i} />
            ))}
          </div>
        )}
      </section>

      <section className="grid gap-3 sm:grid-cols-2">
        <ListBlock title="Risks" items={d.risks} />
        <ListBlock title="Open questions" items={d.open_questions} />
      </section>

      {markdown && (
        <details className="rounded-lg border border-base-300 bg-base-100/70">
          <summary className="cursor-pointer select-none px-4 py-2 text-sm font-medium">
            Raw markdown
          </summary>
          <div className="px-5 pb-4 prose prose-sm max-w-none">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{markdown}</ReactMarkdown>
          </div>
        </details>
      )}
    </div>
  );
}

export function Decompose() {
  const [query, setQuery] = useState("");
  const [adapter, setAdapter] = useState(DEFAULT_ADAPTER);
  const [adapters, setAdapters] = useState<AdapterInfo[]>([]);
  const [submitting, setSubmitting] = useState(false);
  const [response, setResponse] = useState<AdapterDecomposeResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.listAdapters().then(
      (r) => {
        const usable = r.adapters.filter(
          (a) => a.installed && a.capabilities.includes("decompose"),
        );
        setAdapters(usable);
        if (usable.length > 0 && !usable.some((a) => a.name === adapter)) {
          setAdapter(usable[0].name);
        }
      },
      (e) => setError(`Could not load adapters: ${String(e)}`),
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const submit = async () => {
    if (!query.trim() || submitting) return;
    setSubmitting(true);
    setError(null);
    setResponse(null);
    try {
      const r = await api.adapterDecompose(adapter, { query: query.trim() });
      setResponse(r);
    } catch (e) {
      setError(String(e));
    } finally {
      setSubmitting(false);
    }
  };

  const result = response?.result;
  const adapterLabel = useMemo(() => result?.adapter || adapter, [result, adapter]);

  return (
    <div className="flex-1 min-h-0 overflow-y-auto px-4 sm:px-6 py-6">
      <div className="max-w-3xl mx-auto flex flex-col gap-6">
        <section className="rounded-2xl border border-base-300 bg-base-100/90 shadow-sm p-4 sm:p-5">
          <label className="block text-sm font-medium mb-2">
            Ticket or task description
          </label>
          <textarea
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Describe the work to decompose — title + body, or a full ticket."
            className="textarea textarea-bordered w-full text-sm min-h-24"
            disabled={submitting}
          />
          <div className="mt-3 flex flex-wrap items-center gap-3">
            <div>
              <label className="text-xs text-base-content/60 mr-2">Adapter</label>
              <select
                value={adapter}
                onChange={(e) => setAdapter(e.target.value)}
                className="select select-bordered select-sm"
                disabled={submitting || adapters.length === 0}
              >
                {adapters.length === 0 ? (
                  <option>{adapter}</option>
                ) : (
                  adapters.map((a) => (
                    <option key={a.name} value={a.name} disabled={!a.health.ok}>
                      {a.name}{!a.health.ok ? " (unhealthy)" : ""}
                    </option>
                  ))
                )}
              </select>
            </div>
            <button
              type="button"
              onClick={submit}
              disabled={submitting || !query.trim()}
              className="btn btn-primary btn-sm ml-auto"
            >
              {submitting ? (
                <>
                  <span className="loading loading-spinner loading-xs" />
                  Decomposing…
                </>
              ) : (
                "Decompose"
              )}
            </button>
          </div>
          {query.trim() === "" && (
            <div className="mt-4">
              <p className="text-[11px] uppercase tracking-wide text-base-content/55 mb-2">
                Try
              </p>
              <ul className="flex flex-col gap-1">
                {SUGGESTED.map((q) => (
                  <li key={q}>
                    <button
                      type="button"
                      onClick={() => setQuery(q)}
                      className="text-left text-sm text-base-content/80 hover:text-primary hover:underline"
                    >
                      {q}
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </section>

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

        {submitting && !response && (
          <div className="rounded-2xl border border-base-300 bg-base-100/90 p-6 flex items-center gap-3 text-sm text-base-content/85 shadow-sm">
            <span className="loading loading-spinner loading-sm text-primary" />
            Running through <span className="font-mono">{adapter}</span>…
          </div>
        )}

        {result && (
          <div>
            <header className="mb-4 flex items-center gap-2 text-xs text-base-content/60">
              <span>via</span>
              <span className="badge badge-sm badge-ghost font-mono">{adapterLabel}</span>
              {result.metrics.model && (
                <>
                  <span>·</span>
                  <span>model</span>
                  <span className="badge badge-sm badge-ghost font-mono">
                    {result.metrics.model}
                  </span>
                </>
              )}
              {typeof result.metrics.duration_ms === "number" && (
                <>
                  <span>·</span>
                  <span>{(result.metrics.duration_ms / 1000).toFixed(1)}s</span>
                </>
              )}
            </header>
            <DecompositionView d={result.decomposition} markdown={result.markdown} />
          </div>
        )}
      </div>
    </div>
  );
}
