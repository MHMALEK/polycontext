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

// Bake-off (eval/outputs/bakeoff-20260518T134739Z) settled the default:
// cursor scored 0.773 ungrounded and was the only adapter with stable
// run-to-run results. Sourcebot scored 0.45.
const DEFAULT_ADAPTER = "cursor";

const SUGGESTED = [
  "Pre-fill supplier questionnaire on behalf of supplier — what changes across repos?",
  "Add a new role 'DATA_REVIEWER' with audit-only permissions",
  "Migrate the auth flow from session cookies to JWT",
];

// Adapters listed in preference order (matches bake-off ranking).
const ADAPTER_PRIORITY: Record<string, number> = {
  cursor: 0,
  gemini: 1,
  claude_code: 2,
  opencode: 3,
  sourcebot: 4,
  cline_sdk: 5,
  openai_agents: 6,
};

const COMPLEXITY_CLS: Record<Subtask["estimated_complexity"], string> = {
  small: "badge-success",
  medium: "badge-info",
  large: "badge-warning",
  unknown: "badge-ghost",
};

function complexityBadge(c: Subtask["estimated_complexity"]) {
  return (
    <span className={`badge badge-sm ${COMPLEXITY_CLS[c] ?? "badge-ghost"}`}>
      {c}
    </span>
  );
}

function SubtaskCard({ st, idx }: { st: Subtask; idx: number }) {
  return (
    <article
      id={`subtask-${idx + 1}`}
      className="rounded-xl border border-base-300 bg-base-100 shadow-sm scroll-mt-4"
    >
      <header className="flex items-start gap-3 px-4 sm:px-5 pt-4 pb-3 border-b border-base-200/70">
        <span className="text-sm font-mono text-base-content/45 mt-0.5 tabular-nums select-none">
          {String(idx + 1).padStart(2, "0")}
        </span>
        <div className="flex-1 min-w-0">
          <h3 className="font-semibold text-base sm:text-lg leading-snug">
            {st.title}
          </h3>
          <div className="mt-1.5 flex flex-wrap gap-1.5 items-center text-[11px]">
            {st.repo && (
              <span className="badge badge-sm badge-outline font-mono">
                {st.repo}
              </span>
            )}
            {complexityBadge(st.estimated_complexity)}
            {st.files.length > 0 && (
              <span className="text-[10px] text-base-content/55">
                · {st.files.length} file{st.files.length === 1 ? "" : "s"}
              </span>
            )}
          </div>
        </div>
      </header>
      <div className="px-4 sm:px-5 py-4 space-y-4">
        <p className="text-sm text-base-content/85 leading-relaxed whitespace-pre-wrap">
          {st.description}
        </p>
        {st.files.length > 0 && (
          <div>
            <p className="text-[10px] uppercase tracking-wide text-base-content/55 mb-1.5 font-medium">
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
          <div>
            <p className="text-[10px] uppercase tracking-wide text-base-content/55 mb-1.5 font-medium">
              Acceptance criteria
            </p>
            <ul className="text-sm text-base-content/80 space-y-1">
              {st.acceptance_criteria.map((ac, i) => (
                <li key={i} className="flex items-start gap-2">
                  <span className="text-base-content/40 mt-0.5">✓</span>
                  <span>{ac}</span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
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
    <details className="rounded-lg border border-base-300 bg-base-100" open>
      <summary className="cursor-pointer select-none px-4 py-2.5 text-sm font-medium">
        {title}{" "}
        <span className="text-base-content/50 font-normal">
          ({items.length})
        </span>
      </summary>
      <ul className="px-5 pb-3 list-disc text-sm space-y-1">
        {items.map((x, i) => (
          <li key={i} className="text-base-content/85">
            {x}
          </li>
        ))}
      </ul>
    </details>
  );
}

function StatsHeader({ d }: { d: Decomposition }) {
  const counts = useMemo(() => {
    const total = d.subtasks.length;
    const byComplexity: Record<string, number> = {
      large: 0,
      medium: 0,
      small: 0,
      unknown: 0,
    };
    const byRepo: Record<string, number> = {};
    for (const st of d.subtasks) {
      byComplexity[st.estimated_complexity] =
        (byComplexity[st.estimated_complexity] ?? 0) + 1;
      if (st.repo) byRepo[st.repo] = (byRepo[st.repo] ?? 0) + 1;
    }
    return { total, byComplexity, byRepo };
  }, [d]);
  const repoCount = Object.keys(counts.byRepo).length;
  return (
    <div className="flex flex-wrap items-center gap-4 text-sm text-base-content/75 px-4 py-3 rounded-lg bg-base-200/40 border border-base-300/70">
      <span>
        <strong className="text-base-content">{counts.total}</strong>{" "}
        subtask{counts.total === 1 ? "" : "s"}
      </span>
      <span className="text-base-content/30">·</span>
      <span>
        <strong className="text-base-content">{repoCount}</strong>{" "}
        repo{repoCount === 1 ? "" : "s"}
      </span>
      {counts.byComplexity.large > 0 && (
        <>
          <span className="text-base-content/30">·</span>
          <span className="flex items-center gap-1.5">
            <span className="inline-block w-2 h-2 rounded-full bg-warning" />
            {counts.byComplexity.large} large
          </span>
        </>
      )}
      {counts.byComplexity.medium > 0 && (
        <span className="flex items-center gap-1.5">
          <span className="inline-block w-2 h-2 rounded-full bg-info" />
          {counts.byComplexity.medium} medium
        </span>
      )}
      {counts.byComplexity.small > 0 && (
        <span className="flex items-center gap-1.5">
          <span className="inline-block w-2 h-2 rounded-full bg-success" />
          {counts.byComplexity.small} small
        </span>
      )}
    </div>
  );
}

/** Format the decomposition as plaintext / markdown suitable for Jira description body. */
function toJiraMarkdown(d: Decomposition): string {
  const lines: string[] = [];
  if (d.overview) {
    lines.push("## Overview", "", d.overview.trim(), "");
  }
  if (d.affected_repos.length > 0) {
    lines.push(
      "**Affected repos:** " +
        d.affected_repos.map((r) => `\`${r}\``).join(", "),
      "",
    );
  }
  lines.push("## Subtasks", "");
  d.subtasks.forEach((st, i) => {
    lines.push(
      `### ${String(i + 1).padStart(2, "0")} — ${st.title}` +
        (st.repo ? `  *(${st.repo}, ${st.estimated_complexity})*` : ""),
    );
    lines.push("");
    if (st.description) lines.push(st.description.trim(), "");
    if (st.files.length > 0) {
      lines.push("**Files:** " + st.files.map((f) => `\`${f}\``).join(", "), "");
    }
    if (st.acceptance_criteria.length > 0) {
      lines.push("**Acceptance criteria:**");
      st.acceptance_criteria.forEach((ac) => lines.push(`- ${ac}`));
      lines.push("");
    }
  });
  if (d.risks.length > 0) {
    lines.push("## Risks", "");
    d.risks.forEach((r) => lines.push(`- ${r}`));
    lines.push("");
  }
  if (d.open_questions.length > 0) {
    lines.push("## Open questions", "");
    d.open_questions.forEach((q) => lines.push(`- ${q}`));
    lines.push("");
  }
  return lines.join("\n").trim() + "\n";
}

function CopyButton({
  label,
  value,
  title,
}: {
  label: string;
  value: string;
  title: string;
}) {
  const [done, setDone] = useState(false);
  return (
    <button
      type="button"
      onClick={async () => {
        try {
          await navigator.clipboard.writeText(value);
          setDone(true);
          setTimeout(() => setDone(false), 1500);
        } catch {
          /* clipboard unavailable; ignore */
        }
      }}
      className="btn btn-xs btn-ghost gap-1"
      title={title}
    >
      {done ? (
        <>
          <span>✓</span>
          <span>Copied</span>
        </>
      ) : (
        <span>{label}</span>
      )}
    </button>
  );
}

function ResultActions({
  d,
  markdown,
}: {
  d: Decomposition;
  markdown: string;
}) {
  const json = useMemo(() => JSON.stringify(d, null, 2), [d]);
  const jiraMd = useMemo(() => toJiraMarkdown(d), [d]);
  const mdValue = markdown.trim() ? markdown : jiraMd;
  return (
    <div className="flex flex-wrap items-center gap-1 -ml-1">
      <CopyButton
        label="Copy Markdown"
        value={mdValue}
        title="Markdown body ready to paste into a Jira / Linear / GitHub description"
      />
      <CopyButton
        label="Copy JSON"
        value={json}
        title="Structured Decomposition object — feed to a script or API"
      />
      <CopyButton
        label="Copy Jira body"
        value={jiraMd}
        title="Decomposition rendered as a Jira-friendly markdown body"
      />
    </div>
  );
}

function SubtaskJumpList({ d }: { d: Decomposition }) {
  if (d.subtasks.length <= 1) return null;
  return (
    <nav className="sticky top-0 z-10 -mx-4 sm:-mx-5 px-4 sm:px-5 py-2 bg-base-100/95 backdrop-blur-sm border-b border-base-200">
      <ul className="flex gap-2 overflow-x-auto whitespace-nowrap text-[11px] scrollbar-thin">
        {d.subtasks.map((st, i) => (
          <li key={i}>
            <a
              href={`#subtask-${i + 1}`}
              className="link link-hover text-base-content/65 hover:text-primary"
            >
              <span className="font-mono text-base-content/45">
                {String(i + 1).padStart(2, "0")}
              </span>{" "}
              {st.title.length > 38 ? st.title.slice(0, 36) + "…" : st.title}
            </a>
          </li>
        ))}
      </ul>
    </nav>
  );
}

function DecompositionView({
  d,
  markdown,
}: {
  d: Decomposition;
  markdown: string;
}) {
  return (
    <div className="space-y-5">
      <StatsHeader d={d} />

      <section className="rounded-xl border border-base-300 bg-base-100 p-4 sm:p-5">
        <h2 className="text-base sm:text-lg font-semibold mb-2">Overview</h2>
        <div className="prose prose-sm max-w-none">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>
            {d.overview || "_(empty)_"}
          </ReactMarkdown>
        </div>
        {d.affected_repos.length > 0 && (
          <div className="mt-4 pt-3 border-t border-base-200 flex flex-wrap items-center gap-2">
            <span className="text-[10px] uppercase tracking-wide text-base-content/55 font-medium">
              Affected repos
            </span>
            {d.affected_repos.map((r) => (
              <span
                key={r}
                className="badge badge-sm badge-outline font-mono"
              >
                {r}
              </span>
            ))}
          </div>
        )}
      </section>

      <section>
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-base sm:text-lg font-semibold">
            Subtasks{" "}
            <span className="text-base-content/50 text-sm font-normal">
              ({d.subtasks.length})
            </span>
          </h2>
        </div>
        <SubtaskJumpList d={d} />
        {d.subtasks.length === 0 ? (
          <p className="text-sm text-base-content/55 italic">
            No subtasks produced.
          </p>
        ) : (
          <div className="grid gap-3 mt-3">
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
        <details className="rounded-lg border border-base-300 bg-base-100">
          <summary className="cursor-pointer select-none px-4 py-2.5 text-sm font-medium">
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
  const [response, setResponse] = useState<AdapterDecomposeResponse | null>(
    null,
  );
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.listAdapters().then(
      (r) => {
        const usable = r.adapters
          .filter((a) => a.installed && a.capabilities.includes("decompose"))
          .sort(
            (a, b) =>
              (ADAPTER_PRIORITY[a.name] ?? 99) -
              (ADAPTER_PRIORITY[b.name] ?? 99),
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
  const adapterLabel = useMemo(
    () => result?.adapter || adapter,
    [result, adapter],
  );

  return (
    <div className="flex-1 min-h-0 overflow-y-auto px-4 sm:px-6 py-6">
      <div className="max-w-3xl mx-auto flex flex-col gap-6">
        <section className="rounded-2xl border border-base-300 bg-base-100/90 shadow-sm p-4 sm:p-5">
          <div className="flex items-center justify-between mb-2">
            <label className="block text-sm font-medium">
              Ticket or task description
            </label>
            <span className="text-[10px] text-base-content/50">
              Paste a Jira / Linear ticket body, or describe the work
            </span>
          </div>
          <textarea
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Describe the work to decompose — title + body, or a full ticket."
            className="textarea textarea-bordered w-full text-sm min-h-28 leading-relaxed"
            disabled={submitting}
          />
          <div className="mt-3 flex flex-wrap items-center gap-3">
            <div>
              <label className="text-xs text-base-content/60 mr-2">
                Adapter
              </label>
              <select
                value={adapter}
                onChange={(e) => setAdapter(e.target.value)}
                className="select select-bordered select-sm"
                disabled={submitting || adapters.length === 0}
                title="Adapters are listed in measured-quality order (see eval/outputs/bakeoff-*/comparison.md)"
              >
                {adapters.length === 0 ? (
                  <option>{adapter}</option>
                ) : (
                  adapters.map((a) => (
                    <option
                      key={a.name}
                      value={a.name}
                      disabled={!a.health.ok}
                    >
                      {a.name}
                      {!a.health.ok ? " (unhealthy)" : ""}
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
              <p className="text-[11px] uppercase tracking-wide text-base-content/55 mb-2 font-medium">
                Try one of
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
            <header className="mb-4 flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-base-content/60">
              <span>via</span>
              <span className="badge badge-sm badge-ghost font-mono">
                {adapterLabel}
              </span>
              {result.metrics.model && (
                <>
                  <span>·</span>
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
              {typeof result.metrics.cost_usd === "number" &&
                result.metrics.cost_usd > 0 && (
                  <>
                    <span>·</span>
                    <span>${result.metrics.cost_usd.toFixed(4)}</span>
                  </>
                )}
              <div className="ml-auto">
                <ResultActions
                  d={result.decomposition}
                  markdown={result.markdown}
                />
              </div>
            </header>
            <DecompositionView
              d={result.decomposition}
              markdown={result.markdown}
            />
          </div>
        )}
      </div>
    </div>
  );
}
