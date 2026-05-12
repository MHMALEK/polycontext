# Making ticket-recon's tech decomposition better and more efficient

A grounded survey of tools, techniques, and architectural moves that would
materially improve the accuracy, cost, and reliability of `ticket-recon`'s
output. Ordered by impact-per-effort, with honest tradeoffs.

The premise: **retrieval and grounding are now solid** (anchors + 3-hop
import follow + Sourcebot + Serena + tree-sitter). The remaining gap is
**reliability** — the model still mis-cites lines, sometimes invents files,
and under-uses its own tools. The work below addresses that gap.

---

## 1. Evaluation infrastructure (highest leverage; nothing else is measurable without this)

You can't improve what you can't measure. This is genuinely the biggest gap.

### Build a 20-ticket eval set
- 5 single-repo tickets (slug-style, easy)
- 5 cross-repo refactors (API + frontend)
- 5 migrations (DAG → CF style, hard)
- 5 ambiguous / under-specified tickets

For each: expected `affected_repos`, expected key files, expected critical
risks. Score outputs on these axes. Even a rough rubric beats no rubric.

### Tools to use
- **[Inspect](https://inspect.ai-safety-institute.org.uk/)** — UK AISI's eval framework.
  Python-native, structured around tasks + scorers + solvers. The best one
  for code-agent evals right now.
- **[Ragas](https://docs.ragas.io/)** — RAG-specific metrics: faithfulness,
  answer relevancy, context precision. Use for the `--ask` mode.
- **[promptfoo](https://www.promptfoo.dev/)** — YAML-driven LLM eval CLI.
  Good for "did this prompt change make it better or worse" A/B comparisons.
- **[SWE-bench](https://www.swebench.com/)** — academic benchmark for code
  agents (real GitHub issues + tests). Not directly applicable but the
  scoring methodology is worth borrowing.
- **[LangSmith](https://docs.smith.langchain.com/)** / **[LangFuse](https://langfuse.com/)** —
  trace + eval observability for LLM apps. LangFuse is OSS, hostable.

### What this unlocks
- Every prompt tweak, retrieval change, or model swap becomes measurable.
- Cost/latency trends tracked alongside quality.
- You can run regression eval in CI before merging prompt changes.

---

## 2. Better retrieval (concrete cost reduction + quality boost)

### Code embeddings + vector search
Our keyword search misses semantically-related code. A vector index over
chunks of code would catch "what files implement this concept" even when
the keywords don't match.

- **[Voyage AI's `voyage-code-3`](https://blog.voyageai.com/2024/12/04/voyage-code-3/)** —
  state-of-the-art code embeddings, beats OpenAI/Cohere on code benchmarks
  by 10-15%. $0.06/1M tokens.
- **[OpenAI `text-embedding-3-large`](https://platform.openai.com/docs/guides/embeddings)** —
  general-purpose, cheap ($0.13/1M), respectable on code.
- **[Cohere `embed-v4`](https://cohere.com/blog/embed-v4)** — multimodal, multilingual,
  excellent on code.
- **[Nomic `nomic-embed-code`](https://huggingface.co/nomic-ai/nomic-embed-code)** —
  OSS, runs locally, ~7B params. Free if you have a GPU.

Vector store: **[LanceDB](https://lancedb.com/)** (embedded, file-based, no
extra service) is the easiest. Alternatives: pgvector (if you already
have Postgres), Qdrant (standalone service), Chroma.

### Cross-encoder reranker
Pull 100+ candidates from retrieval, rerank with a cross-encoder, send
the top 20 to Pro. Cuts noise and improves precision in the context window.

- **[Cohere Rerank 3.5](https://cohere.com/blog/rerank-3pt5)** — best API option.
  $2/1k searches. Hosted.
- **[BAAI `bge-reranker-v2-m3`](https://huggingface.co/BAAI/bge-reranker-v2-m3)** —
  OSS, runs locally on CPU/small GPU.
- **[Voyage `rerank-2.5`](https://docs.voyageai.com/docs/reranker)** — code-aware reranker.

Wire this in between `gather_context` and `decompose` — same code path, just
a filter step.

### Anthropic-style contextual retrieval
Before embedding/indexing each code chunk, prepend a 1-2 sentence summary
of what that chunk is and where it lives in the codebase. Anthropic showed
this cuts retrieval failures by ~50%. Cheap to implement (one Flash call
per chunk during indexing).

Read: <https://www.anthropic.com/news/contextual-retrieval>

### GraphRAG over the import graph
We already build a per-repo import index (`import_index.py`). A natural
next step: build a real **knowledge graph** over the repos (modules,
classes, functions, calls) and query it with graph-aware retrieval.

- **[Microsoft GraphRAG](https://microsoft.github.io/graphrag/)** — well-engineered
  framework, integrates with embeddings.
- **[HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG)** — combines vector +
  graph retrieval, claims 20% improvement over pure RAG.
- **[Repograph](https://github.com/CGCL-codes/RepoGraph)** — repo-as-graph approach,
  built specifically for code understanding.

For our case, a lighter version is probably enough — extend `import_index.py`
to also track symbol-level relationships (calls, inheritance) via Serena's
LSP tools.

### Tree-sitter symbol extraction (extend what we have)
We use tree-sitter for TS imports. Extend to extract **symbols** (functions,
classes, exported names) so we can build a symbol table per repo. Lets us
answer "is there a function called X anywhere?" in O(1).

- **[tree-sitter-language-pack](https://pypi.org/project/tree-sitter-language-pack/)** —
  100+ grammars in one package.
- **Aider's repomap** — they do exactly this. Read their code:
  <https://github.com/Aider-AI/aider/blob/main/aider/repomap.py>

---

## 3. Better agent reliability (closes the hallucination gap)

### Verifier loop / constitutional AI
After the agent produces a Decomposition, run a second LLM pass that
checks each subtask: "Is this file path real? Was this symbol cited?
Does this contradict the ticket?" Reject failures, regenerate with the
failures as constraints.

We have a partial version already (the contradiction-check stage). Extend
to a full verifier:
- Every cited file path → verify via filesystem
- Every cited symbol → verify via `find_symbol` (we have this tool)
- Every acceptance criterion → check it doesn't contradict the ticket

Read: <https://arxiv.org/abs/2310.10501> ("Verify-then-Generate" patterns)

### Self-consistency / majority voting
Run the decomposition N times with `temperature > 0`, take subtasks that
appear in ≥2/3 runs. Hallucinations are uncorrelated; real findings agree.

Cost goes 3x. For high-stakes tickets only (gate on confidence).

Read: Wang et al. "Self-Consistency Improves Chain of Thought Reasoning"
<https://arxiv.org/abs/2203.11171>

### Tool-use enforcement (deterministic guardrails)
Rather than asking the model to use a tool ("you MUST call find_symbol"),
**enforce it programmatically**: when the model proposes a subtask
referencing a symbol it didn't verify, our code calls `find_symbol` on
that symbol before accepting the output, and rejects subtasks where the
symbol wasn't verified.

This is a hard barrier the model can't talk its way around. Half-day of
work in our `decompose.py`.

### Routing by complexity
Use Flash to classify ticket complexity, then route:
- Simple (single-file change): Flash decomposes too. ~$0.005/ticket.
- Medium (multi-file): Pro single-shot (current cheap mode). ~$0.07.
- Complex (multi-repo, ambiguous): Pro agentic loop (current deep mode). ~$0.30.
- Critical (production migrations): Opus / Sonnet 4.6 agentic. ~$1-2.

The routing decision itself costs $0.001. Average cost per ticket drops.

---

## 4. Observability & feedback loops

### Tracing / LLM observability
- **[LangFuse](https://langfuse.com/)** — OSS, self-hostable, traces every
  LLM call + tool call + retrieval step. Best for ticket-recon.
- **[Helicone](https://www.helicone.ai/)** — drop-in proxy, simpler setup,
  hosted.
- **[Arize Phoenix](https://phoenix.arize.com/)** — OSS, good UI for agent traces.
- **[OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)** —
  if you want to integrate with existing observability.

Pydantic AI has first-class OTel support. Wire up LangFuse via OTel for
free traces.

### User feedback collection
Add a thumbs-up/down on each Jira comment posted. Track which decompositions
were useful (delivered) vs not. Use to iterate prompts / build training data.

### Prompt versioning + A/B
- **[LangSmith](https://docs.smith.langchain.com/)** has prompt versioning.
- **[Pezzo](https://pezzo.ai/)** is OSS for this.
- Or roll your own: a `prompts/` directory with versioned YAML files,
  recorded in metrics.

---

## 5. Specialized retrieval frameworks worth knowing

### Aider's repomap (above) — proven, simple
Symbol-aware repo summarization for LLM context. Pure tree-sitter, no LLM
in the loop. Could replace our anchor + import-follow with something more
structured.

### Cody's context engine
Sourcegraph's Cody uses a custom retrieval engine with embeddings + symbol
search + graph awareness. Their approach is well-documented:
<https://sourcegraph.com/blog/cody-context>

### Cursor's approach (not open)
Inferred from behavior: embeddings + symbol graph + recency weighting +
LLM-driven query rewriting. The reason Cursor is good is mostly the
**iteration loop with the user**, not a single retrieval trick.

### **[grep.app](https://grep.app)** for grounding
Web-scale code search. Free API for low-volume use. Useful when the LLM
proposes an API and you want to verify it's real before suggesting it.

### **[CodeQL](https://codeql.github.com/)** / **[Semgrep](https://semgrep.dev/)**
Static analyzers. Run them on the repo, feed findings into the
decomposition context. Catches "this subtask would introduce a SQL
injection" before the agent ships it.

### **[Tantivy](https://github.com/quickwit-oss/tantivy)** / **[zoekt](https://github.com/sourcegraph/zoekt)**
What Sourcebot is built on. If you outgrow Sourcebot or want lower-level
control, you can build directly on these.

---

## 6. Schema and product-shape improvements

### Spec-as-output (instead of subtasks)
Right now `Decomposition` produces subtasks. An alternative output: a
**technical spec** with sections for "Approach", "Risks", "Test plan",
"Migration steps". More flexible, less prescriptive. Agents could still
parse it.

### Confidence-scored output
Every subtask gets a `confidence: low/medium/high` based on grounding.
Routes high-confidence subtasks to "agent picks this up" and
low-confidence to "human review first". Cheap to add.

### Streaming output
Right now the user waits 60-120s for the full markdown. Streaming the
sections as they're written feels much better. Pydantic AI supports it.

### Multi-turn refinement
"This subtask is too big, split it" / "consolidate subtasks 2 and 3"
as follow-up turns. Closer to a conversation. Bigger product change.

---

## 7. Less-obvious quality wins

### Read the ticket's history (Jira comments)
Tickets accumulate clarifying comments. Right now we only read the
description. Pulling the comment thread often resolves ambiguities the
description leaves open. Free, just an extra Jira API call.

### Cross-reference adjacent tickets
"This ticket is one of 5 in epic X. Read the other 4 first." Costs more
context but often resolves ambiguity. Use sparingly.

### Read the PR templates / commit history
Each repo has conventions visible in recent PRs (`gh pr list` / `git log`).
Feeding the last 10 PRs' titles + descriptions of a repo to enrichment
helps the model match style.

### Pre-load CODEOWNERS
For each subtask, hint at who owns the touched files (from CODEOWNERS).
Useful downstream for routing.

### Use Anthropic's prompt caching
We're paying for the same system prompt every call. Anthropic and Gemini
both cache identical prefixes at a discount. Pydantic AI supports the
Anthropic version. ~50-90% input cost reduction on repeat calls.

---

## What I'd actually do, in order

If you have an afternoon:

1. **Eval set (5-10 tickets, rough scoring)** — half day. Without this every other change is gambling.
2. **Verifier loop** — half day. Half the contradictions and hallucinations go away.

If you have a week:

3. **Code embeddings + vector retrieval** — 1-2 days. Materially better recall.
4. **Reranker** — half day. Tightens precision on the context sent to Pro.
5. **Routing by complexity** — half day. Cost per ticket drops 30-50%.
6. **LangFuse tracing** — half day. Visibility into where time/cost actually goes.

If you have a month:

7. Tool-use enforcement (deterministic verifier)
8. Aider-style repomap as an alternative retrieval source
9. Contextual retrieval (Anthropic style) on the symbol index
10. Self-consistency for critical tickets

---

## Anti-recommendations (don't do these, even though they're tempting)

- **Don't fine-tune a custom model.** You don't have the eval set or the
  training data. Spend that time on prompts and retrieval instead.
- **Don't add LangGraph for orchestration.** The pipeline is 4 stages.
  LangGraph adds nodes/state for branching that you don't need.
- **Don't add another retriever.** You have ripgrep + Sourcebot + Serena
  + anchors + import-follow. The bottleneck is no longer retrieval breadth,
  it's reranking + verification.
- **Don't add multi-model ensembling.** 3x cost for 5% quality. Self-consistency
  with one model is cheaper and works better.
- **Don't pretend Sourcebot's `/api/ask` will work soon.** Either get
  Enterprise / hosted Sourcebot, or commit to the local agentic path. The
  current half-and-half is fine for now but don't build more on the
  Sourcebot path without proving it works.
