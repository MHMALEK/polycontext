# SCRUM-18 — Traceability Static Validation Move to CF

**Hand-grounded decomposition** for comparison against the AI-generated
versions ([comment 10178](https://code-geppetto.atlassian.net/browse/SCRUM-18?focusedCommentId=10178) — Gemini;
[comment 10179](https://code-geppetto.atlassian.net/browse/SCRUM-18?focusedCommentId=10179) — Cursor).

Every file path below was verified to exist at the time of writing
(2026-05-19) by reading the actual files in `REPOS_ROOT`. Line numbers
are included where they help.

---

## Overview

The traceability static-validation runtime currently lives in the
`data` repo as an Airflow DAG —
`data/src/composer/dag/traceability_static_validations_checker.py` —
that wires together a custom operator
(`TraceabilityStaticValidationOperator`) and a strategy engine
(`StaticValidationEngine` + `TraceabilityStaticValidator` +
`TraceabilityErrorReporter`). The DAG is triggered by the
`composer_dag_trigger` cloud function for traceability uploads.

The ticket asks to move the validation **runtime** out of Airflow and
into a Cloud Function. The `data-cloud-functions` repo already has the
exact extension point: a `process_traceability_file` **stub** (line 379
of `composer_dag_trigger/main.py`) waiting for this implementation. The
pattern to mirror is `run_geojson_static_validation` (line 114 of the
same file) — same repo did this migration for GeoJSON.

**Affected repos:** `data`, `data-cloud-functions`

## Risks

- **Code-sharing strategy.** The strategy engine lives in
  `data/src/composer/dag/static_validation/` and uses Airflow-coupled
  modules (`airflow.models`, `airflow.providers.google.cloud.hooks.gcs`).
  Two viable options: (a) copy the Airflow-free subset of
  `static_validation/` into
  `data-cloud-functions/src/cloud_functions/composer_dag_trigger/traceability/`
  (mirroring how `geolocation/` is structured), or (b) extract
  `static_validation/` into a shared wheel both repos can install.
  Option (a) is faster; option (b) avoids drift. Pick before line 1
  of work.
- **Memory ceiling for 250 MB files.** The ticket targets <2 GB memory
  at 250 MB file size. `StaticValidationEngine._load_data` currently
  uses `TemplateDataLoader().load(file_path)` which loads the entire
  file via pandas. A Cloud Function 2nd-gen instance can be sized to
  2 GiB, but Cold start + framework overhead leaves <1.5 GiB for the
  DataFrame. Verify with a real 250 MB fixture before merging.
- **Acceptance criteria mismatch with current code.** The ticket says
  the error CSV format must be "Row, Column, Reason" — but the live
  reporter emits columns `File, Field, Value, Issue, Suggestion` (see
  `data/src/composer/dag/static_validation/reporters.py:25`). Resolve
  with the ticket author: keep current columns (recommended — UI
  consumers depend on them) or rename across all consumers.

## Open questions

- Does the new CF run on Cloud Functions Gen 2 (recommended for higher
  memory + longer timeout) or Gen 1? Other CFs in this repo (e.g.
  `data_sharing/main.py`) use the Gen 2 shape (`functions_framework.cloud_event`)
  — assume Gen 2 unless told otherwise.
- The GeoJSON migration (the template here) uses
  `composer2_airflow_rest_api.trigger_dag` to fire the downstream
  ingestion DAG (see `composer_dag_trigger/composer2_airflow_rest_api.py`).
  Should the traceability version trigger the same way, or move that to
  Pub/Sub for decoupling?
- The current DAG branches between `traceability_excel_conversion` and
  `traceability_segment` based on file type
  (`branch_by_file_type_for_ingestion` in
  `static_validation/traceability_static_validation_helpers.py`). The
  CF needs to make the same choice. Is it OK to inline that branch in
  the CF, or does the orchestrator (`composer_dag_trigger`) need to
  remain the single point of DAG selection?

## Subtasks

### 01 — Port the Airflow-free validation core into the CF package
**Repo:** `data-cloud-functions` · **Complexity:** large

Copy (or shared-wheel — see Risk #1) the parts of
`data/src/composer/dag/static_validation/` that don't import `airflow.*`
into `data-cloud-functions/src/cloud_functions/composer_dag_trigger/traceability/`
(matching the existing `geolocation/` shape). Concretely:

- `engine.py` (StaticValidationEngine) — drop the
  `helpers.template_data_loaders` import; that helper itself is
  Airflow-free, just needs to be vendored in.
- `strategy.py` (TraceabilityStaticValidator) — pure pandas, copies cleanly.
- `reporters.py` (TraceabilityErrorReporter) — uses
  `data_files_utils.helpers.DataFileHelper` which talks to the DB;
  decide whether the CF uses the same helper or writes its own thin
  status updater (the CF env already has DB credentials via
  `composer_dag_trigger/db.py`).
- `generic_validations.py`, `protocols.py`, `contracts.py`,
  `exceptions.py`, `config.py`, `traceability_static_checker.py`,
  `traceability_static_validation_helpers.py` (just the validators
  pulled in by `traceability_static_checker`, not the
  `initialize_static_validation` / `update_static_validation_status` /
  `should_trigger_ingestion` Airflow helpers — those move to subtask 02).

**Files (definite):**
- `data/src/composer/dag/static_validation/engine.py` (source)
- `data/src/composer/dag/static_validation/strategy.py` (source)
- `data/src/composer/dag/static_validation/reporters.py` (source)
- `data/src/composer/dag/static_validation/traceability_static_checker.py` (source)
- `data/src/composer/dag/static_validation/traceability_static_validation_helpers.py` (source — port only the validator imports, not the DAG helpers)
- `data-cloud-functions/src/cloud_functions/composer_dag_trigger/traceability/__init__.py` (NEW)
- `data-cloud-functions/src/cloud_functions/composer_dag_trigger/traceability/engine.py` (NEW)
- `data-cloud-functions/src/cloud_functions/composer_dag_trigger/traceability/strategy.py` (NEW)
- `data-cloud-functions/src/cloud_functions/composer_dag_trigger/traceability/reporters.py` (NEW)
- `data-cloud-functions/src/cloud_functions/composer_dag_trigger/traceability/validators.py` (NEW — the field validators currently in `traceability_static_checker.py`)

**Acceptance criteria:**
- New module has zero `import airflow` references (`grep -r 'import airflow' data-cloud-functions/src/cloud_functions/composer_dag_trigger/traceability/` returns nothing).
- Existing unit tests for the validators (`data/test/composer/dag/static_validation/test_traceability_static_checker.py`) pass against the copied module with at most import-path adjustments.
- CSV report column headers remain `File, Field, Value, Issue, Suggestion` (reconciled with ticket — see Risk #3).

---

### 02 — Implement `process_traceability_file` to mirror `run_geojson_static_validation`
**Repo:** `data-cloud-functions` · **Complexity:** medium

Replace the placeholder at
`data-cloud-functions/src/cloud_functions/composer_dag_trigger/main.py:379`
with a real implementation. Use `run_geojson_static_validation` at line
114 of the same file as the template:

1. Download the file from GCS via `storage_client` (already imported in `main.py`).
2. Instantiate the new traceability engine from subtask 01.
3. On success: set DB status PASSED, call
   `composer2_airflow_rest_api.trigger_dag` for *either*
   `traceability_excel_conversion` or `traceability_segment` based on
   file extension (the branch the DAG does today via
   `branch_by_file_type_for_ingestion`).
4. On failure: set DB status FAILED, attach the error report blob name
   to the file row, do not trigger ingestion.
5. Respect the existing `trigger_ingestion` flag (passed through from
   the cloud-event handler).

**Files:**
- `data-cloud-functions/src/cloud_functions/composer_dag_trigger/main.py` (modify lines 379–381 + add a `handle_traceability_validation_success` / `_failure` pair below `handle_geojson_validation_success` at line ~135)
- `data-cloud-functions/src/cloud_functions/composer_dag_trigger/composer2_airflow_rest_api.py` (read-only — reuse `trigger_dag`)
- `data-cloud-functions/src/cloud_functions/composer_dag_trigger/db.py` (read-only — reuse status update helpers)

**Acceptance criteria:**
- Uploading a known-good `.xlsx` or `.csv` to the supplier bucket no
  longer triggers the Airflow `traceability_static_validations_checker`
  DAG just to run validation; the CF performs equivalent checks.
- On validation pass, the corresponding `traceability_excel_conversion` or `traceability_segment` DAG is triggered exactly once.
- On validation fail, the DB row has `staticValidationPassed=false`,
  `errorReportPath` populated, and the ingestion DAG is NOT triggered.

---

### 03 — Route supplier-bucket uploads through the new CF
**Repo:** `data-cloud-functions` · **Complexity:** small

Update the cloud-event handler in
`composer_dag_trigger/main.py` so that traceability-template uploads go
to `process_traceability_file` instead of triggering the old DAG.
Currently `process_excel_or_csv_file` (line 271) does template detection
and triggers the validation DAG via `trigger_dag`; the modification is
to route through the new local function when the template matches a
traceability template.

**Files:**
- `data-cloud-functions/src/cloud_functions/composer_dag_trigger/main.py`
- `data-cloud-functions/src/cloud_functions/composer_dag_trigger/constants.py` (template-name constants)

**Acceptance criteria:**
- A test event for a traceability-template file calls
  `process_traceability_file`, not `trigger_dag('traceability_static_validations_checker')`.
- A test event for a non-traceability file (e.g. master-data) is unchanged.

---

### 04 — Configure CF memory + timeout for the size envelope
**Repo:** `data-cloud-functions` · **Complexity:** small

The ticket gives target wall times and memory budgets per file size.
Deploy with `--memory 2GiB --timeout 540s` (CF Gen 2 limits) — these
cover the 250 MB / 60 s target with headroom. Specific deploy config
typically lives in the repo's CI yaml.

**Files:**
- `data-cloud-functions/ci/` (find the existing deploy config and add/modify the `composer_dag_trigger` entry)
- `data-cloud-functions/src/cloud_functions/composer_dag_trigger/requirements.txt` — verify `pandas`, `openpyxl` versions match what the DAG uses

**Acceptance criteria:**
- Deployed CF can process a 250 MB traceability CSV without OOM.
- Wall time at the 250 MB envelope is below 60 s in staging.
- Cold-start time is documented (it counts against the same 60 s).

---

### 05 — Decommission or thin the Airflow validation DAG
**Repo:** `data` · **Complexity:** small

Once subtasks 01-04 are deployed and verified, the Airflow DAG
`traceability_static_validations_checker` exists only as a fallback.
Two paths: (a) remove it entirely, (b) keep it disabled for emergency
re-enablement. The ticket asks for rollback parity, so (b) is safer for
one release cycle then (a).

**Files:**
- `data/src/composer/dag/traceability_static_validations_checker.py` (mark `is_paused_upon_creation=True` or remove)
- `data/src/composer/dag/custom_operators/traceability_static_validation_operator.py` (remove if DAG is removed)
- `data/src/composer/dag/static_validation/traceability_static_validation_helpers.py` (the DAG-specific helpers — `initialize_static_validation`, `update_static_validation_status`, `should_trigger_ingestion`, `branch_by_file_type_for_ingestion`)

**Acceptance criteria:**
- DAG is either removed or set to `is_paused_upon_creation=True`.
- Documentation references the new CF path.
- If kept, a runbook describes how to re-enable it within 15 minutes.

---

### 06 — Tests + telemetry parity
**Repo:** `data-cloud-functions` · **Complexity:** medium

Port the existing validation tests at
`data/test/composer/dag/static_validation/test_traceability_static_checker.py`
to the new CF location, and add CF-level tests (HTTP event in,
DB status + DAG trigger out).

**Files:**
- `data/test/composer/dag/static_validation/test_traceability_static_checker.py` (source)
- `data/test/composer/dag/static_validation/test_static_validation_engine.py` (source)
- `data-cloud-functions/test/cloud_functions/composer_dag_trigger/test_traceability_validator.py` (NEW)
- `data-cloud-functions/test/cloud_functions/composer_dag_trigger/test_process_traceability_file.py` (NEW)

**Acceptance criteria:**
- All validator-level tests pass against the new module.
- A CF-level test for a known-good file asserts: DB status `PASSED`, downstream DAG triggered once, no error report.
- A CF-level test for a known-bad file asserts: DB status `FAILED`, error report uploaded, no DAG trigger.
- Validation gap vs DAG is <5% in a sample of 50 historical files (the ticket's stated rollback criterion).
