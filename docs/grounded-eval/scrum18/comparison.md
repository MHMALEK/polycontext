# SCRUM-18 — Hand-grounded vs. AI decompositions

A side-by-side scoring of what the tool produced against a hand-grounded
decomposition built by actually reading the code.

- **Hand-grounded:** [decomposition.md](decomposition.md) — every file path
  verified by reading the file at `REPOS_ROOT` on 2026-05-19.
- **Gemini (comment 10178):** raw JSON at
  `eval/outputs/eval-20260518T*` runs and the comment itself in Jira.
- **Cursor (comment 10179):** same.

## Headline finding

Your instinct was half-right.

| Adapter | Verdict |
|---|---|
| **Gemini (gpt-2.5-pro via @google/genai)** | **Hallucinated.** Invented an entire `gcp/cloud_functions/traceability_validator/` directory tree that doesn't exist anywhere in the codebase. Cited a fake `traceability-service` repo. None of its 7 subtask file paths point at real code. |
| **Cursor (composer-2)** | **Grounded.** Cited `composer_dag_trigger/main.py`, `composer_dag_trigger/constants.py`, `composer_dag_trigger/helpers/utils.py` — all real. Named the real classes (`StaticValidationEngine`, `TraceabilityStaticValidator`, `TraceabilityErrorReporter`) and the real placeholder function (`process_traceability_file`). Even pushed back on the ticket: the ticket says error columns are `Row, Column, Reason` but the live code uses `File, Field, Value, Issue, Suggestion` — verified, Cursor was right. |

If you opened both comments in Jira and felt one hallucinated, you were
looking at Gemini's. Cursor's output is shippable as-is for a developer
to pick up.

## Scoring rubric (0–5 per dimension)

| Dimension | What I'm measuring | Gemini | Cursor | Mine |
|---|---|---|---|---|
| **Real file paths cited** | Fraction of `path/file.py` references that exist in the repo | 0/5 | 4/5 | 5/5 |
| **Real symbols / function names** | Cites actual classes, functions, constants from the code | 0/5 | 5/5 | 5/5 |
| **Identifies the right repos** | Affected repos match the codebase reality | 0/5 (says `traceability-service`) | 5/5 (says `data`, `data-cloud-functions`) | 5/5 |
| **Identifies the migration template** | Notices the GeoJSON CF migration is the pattern to mirror | 0/5 (no mention) | 5/5 (mentions `run_geojson_static_validation`) | 5/5 |
| **Catches errors in the ticket** | Flags acceptance criteria that contradict the live code | 0/5 | 5/5 (CSV columns) | 5/5 |
| **Granularity for pickup** | Subtasks are small enough to land in 1–3 days | 4/5 (7 subtasks) | 3/5 (only 2 subtasks, both large) | 4/5 (6 subtasks) |
| **Acceptance criteria specificity** | Acceptance criteria are testable on the real code | 2/5 (textbook) | 4/5 (specific to behaviors) | 5/5 |
| **Risk identification** | Calls out the non-obvious traps | 1/5 (none meaningful) | 3/5 (memory + benchmarks) | 5/5 (code-sharing strategy, memory ceiling, ticket-vs-code mismatch) |
| **TOTAL** | / 40 | **7/40** | **34/40** | **39/40** |

## Concrete examples of the difference

### Path: where the new CF code should live

| | Said |
|---|---|
| **Gemini** | `gcp/cloud_functions/traceability_validator/main.py` ❌ no such directory |
| **Cursor** | `data-cloud-functions/src/cloud_functions/composer_dag_trigger/main.py` ✅ exists, 379 lines, has the placeholder |
| **Hand** | Same as Cursor, plus a new `composer_dag_trigger/traceability/` subdirectory (mirroring `geolocation/`) |

### Symbol: what the implementation should be replacing

| | Said |
|---|---|
| **Gemini** | "the existing Airflow DAG" — generic, doesn't name a function |
| **Cursor** | `process_traceability_file` ✅ — line 379 of `main.py`, currently `def process_traceability_file(...): pass` |
| **Hand** | Same as Cursor, plus pointer to `run_geojson_static_validation` (line 114) as the structural template |

### Ticket vs. code mismatch

The Jira ticket asks for error report columns `Row, Column, Reason`. Real
code in `data/src/composer/dag/static_validation/reporters.py:25`:

```python
TRACEABILITY_CSV_COLUMNS = ["File", "Field", "Value", "Issue", "Suggestion"]
```

| | Caught it? |
|---|---|
| **Gemini** | No — confidently repeats "(Row, Column, Reason)" as if it were ground truth |
| **Cursor** | **Yes** — "Evidence shows the DAG error-report CSV uses columns File, Field, Value, Issue, Suggestion—not Row, Column, Reason—so the written acceptance criteria needs reconciliation before sign-off." |
| **Hand** | Yes — listed under Risks |

This is the highest-signal data point in the comparison. Gemini is happily
generating plausible engineering paragraphs without checking anything;
Cursor actually went to the file, saw the column list, and flagged the
ticket discrepancy. That's the difference between "looks like a tech
decomposition" and "is one."

## What's still missing in Cursor's output (vs. mine)

Cursor's decomposition is good but not perfect. Three gaps:

1. **Granularity.** Only 2 subtasks, both labeled `large`. The first one
   ("entrypoint + flow") and the second one ("port validation runtime")
   together are roughly a 2-3 week effort and need to be split before
   anyone picks them up. My version splits it into 6 subtasks (port core,
   wire entrypoint, route uploads, configure deploy, decommission DAG,
   tests).
2. **Tests as a discrete subtask.** Cursor folds tests inside the two
   large subtasks. Splitting them out makes them assignable to a
   different engineer / batched at the end.
3. **Decommission step is missing.** Cursor doesn't say what happens to
   the existing Airflow DAG once the CF is live. The ticket asks for
   rollback parity — that's a real subtask.

These are all about *shape*, not *truth*. Cursor's claims are accurate;
its decomposition just needs a junior PM to split a couple of cards.
Gemini's would need to be thrown away and started over.

## Why one hallucinated and the other didn't

Both adapters got the same prompt, the same shared `decompose_preamble`,
the same workspace mount. The difference is the **driving model**:

- **Gemini 2.5 Pro via `@google/genai`** — answered from training
  knowledge of "what does a typical Airflow → CF migration ticket look
  like". Never grounded the claims with a tool call. The path
  `gcp/cloud_functions/traceability_validator/main.py` is exactly the
  shape GCP tutorials use; it's just not your shape.
- **Cursor Composer-2 via `@cursor/sdk` local runtime** — actually
  walked the workspace. The bake-off
  ([eval/outputs/bakeoff-20260518T134739Z/comparison.md](../../../eval/outputs/bakeoff-20260518T134739Z/comparison.md))
  showed Cursor uses 25–30 native tool calls per question; this
  comment is what 25–30 file reads looks like.

This matches the broader finding from the eval: **Cursor is the only
adapter where the model reliably calls tools** with the current default
settings. Gemini under-uses tools (the forced-tool-use prompt fix
recovered ~half the gap in `ask`, but `decompose` doesn't yet inherit
that forcing).

## Recommendation for the bridge default

**Switch the `TECH_DECOMP_ADAPTER` default in `services/jira-bridge/.env.example`
from `cursor` (which it already is) to stay on `cursor`** — and add a
note in the README warning against `gemini` for decompose until the
forced-tool-use prompt is ported into the decompose path.

For comments where Cursor isn't available or the user explicitly wants
to compare, the bridge could be invoked twice in parallel — but you'd
want the Gemini comment clearly labeled as a low-confidence draft, not
a peer to Cursor's.

## What this implies for the tool's value

Concrete answer to "is this tool good?":

- Pointed at **cursor** for decompose, **yes** — it produces a comment a
  developer could pick up after a 10-minute clean-up pass (split the
  two large subtasks, add a decommission task). That's faster than a
  human PM doing it from scratch.
- Pointed at **gemini** for decompose, **no** — the output is plausible
  fiction. A developer trying to act on it would waste an afternoon
  chasing paths that don't exist before realizing the whole comment
  was made up.

The bake-off already measures `ask`. We need a decompose bake-off too
— this SCRUM-18 case is exactly the kind of question that should be in
it, with the hand-grounded version above as the gold reference.
