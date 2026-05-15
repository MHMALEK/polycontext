"""SQLite-backed history of pipeline runs.

Each completed (or failed) Pipeline.run is recorded as one row with the full
answer markdown, structured payload, citations, tokens, cost, and the original
input ref (so replays can rebuild the same call). Backs the `tech-decomposition
history` CLI and the `/runs` API endpoints.

Schema is forward-compatible: new columns can be added with a CREATE-IF and an
ALTER-IF-MISSING dance — but for now the schema is small and stable.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .pipeline import PipelineRun


_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  id              TEXT PRIMARY KEY,
  mode            TEXT NOT NULL,
  status          TEXT NOT NULL,
  engine          TEXT,
  model           TEXT,
  output_format   TEXT,
  input_ref_json  TEXT NOT NULL,
  input_preview   TEXT,
  answer          TEXT,
  citations_json  TEXT,
  payload_json    TEXT,
  total_seconds   REAL,
  total_cost_usd  REAL,
  input_tokens    INTEGER,
  output_tokens   INTEGER,
  error           TEXT,
  created_at      TEXT NOT NULL,
  completed_at    TEXT,
  current_stage   TEXT,
  stages_done     TEXT
);

CREATE INDEX IF NOT EXISTS idx_runs_created ON runs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_runs_mode    ON runs(mode);
CREATE INDEX IF NOT EXISTS idx_runs_status  ON runs(status);
"""

# Columns added after v0.2.0 — applied as ALTERs against existing dbs.
_MIGRATIONS = (
    ("current_stage", "TEXT"),
    ("stages_done", "TEXT"),
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    for k in ("input_ref_json", "citations_json", "payload_json"):
        if d.get(k):
            try:
                d[k.replace("_json", "")] = json.loads(d[k])
            except json.JSONDecodeError:
                pass
            del d[k]
    if d.get("stages_done"):
        try:
            d["stages_done"] = json.loads(d["stages_done"])
        except json.JSONDecodeError:
            d["stages_done"] = []
    return d


class RunStore:
    """Thin DAO over a single-file SQLite database. Thread-safe enough for
    the CLI + a uvicorn worker — uses ``sqlite3.connect(..., check_same_thread=False)``
    and short-lived cursors. For multi-worker setups, point at a shared volume
    and let SQLite's WAL mode handle concurrency.
    """

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        existing = {r[1] for r in self._conn.execute("PRAGMA table_info(runs)").fetchall()}
        for col, typ in _MIGRATIONS:
            if col not in existing:
                self._conn.execute(f"ALTER TABLE runs ADD COLUMN {col} {typ}")
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # ----- writes ---------------------------------------------------------

    def start(
        self,
        *,
        run_id: str,
        mode: str,
        input_ref: Any,
        input_preview: str | None = None,
        output_format: str | None = None,
    ) -> None:
        """Insert a placeholder ``running`` row so live progress updates and
        history polling can find this run before the pipeline finishes."""
        self._conn.execute(
            """
            INSERT OR IGNORE INTO runs
                (id, mode, status, input_ref_json, input_preview, output_format, created_at)
            VALUES (?, ?, 'running', ?, ?, ?, ?)
            """,
            (
                run_id, mode,
                json.dumps(input_ref, default=str),
                input_preview, output_format, _now(),
            ),
        )
        self._conn.commit()

    def update_progress(
        self,
        *,
        run_id: str,
        current_stage: str | None,
        stages_done: list[str],
    ) -> None:
        """UPDATE only the progress columns. No-op if the row doesn't exist."""
        self._conn.execute(
            "UPDATE runs SET current_stage = ?, stages_done = ? WHERE id = ?",
            (current_stage, json.dumps(stages_done), run_id),
        )
        self._conn.commit()

    def record(
        self,
        *,
        run_id: str,
        mode: str,
        status: str,
        input_ref: Any,
        output_format: str | None = None,
        pipeline_run: PipelineRun | None = None,
        engine: str | None = None,
        error: str | None = None,
        created_at: str | None = None,
    ) -> None:
        """Insert (or replace) a row for this run.

        ``engine`` may be supplied directly when there's no PipelineRun — used
        by the adapter bake-off routes which record a tag like ``cline_sdk:ask``
        without running through the legacy Pipeline class.
        """
        model = answer = None
        citations_json = payload_json = None
        total_seconds = total_cost_usd = None
        input_tokens = output_tokens = None
        input_preview = None
        if pipeline_run is not None:
            er = pipeline_run.engine_result
            engine = engine or er.engine
            model = er.model
            answer = er.answer_markdown
            citations_json = json.dumps(er.citations or [])
            payload_json = json.dumps(er.payload or {})
            total_seconds = pipeline_run.total_seconds
            total_cost_usd = er.cost_usd
            input_tokens = er.input_tokens
            output_tokens = er.output_tokens
            input_preview = (pipeline_run.loaded.title or pipeline_run.loaded.body or "")[:160].strip()
        self._conn.execute(
            """
            INSERT OR REPLACE INTO runs
                (id, mode, status, engine, model, output_format, input_ref_json,
                 input_preview, answer, citations_json, payload_json,
                 total_seconds, total_cost_usd, input_tokens, output_tokens,
                 error, created_at, completed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id, mode, status, engine, model, output_format,
                json.dumps(input_ref, default=str), input_preview,
                answer, citations_json, payload_json,
                total_seconds, total_cost_usd, input_tokens, output_tokens,
                error, created_at or _now(), _now() if status != "running" else None,
            ),
        )
        self._conn.commit()

    # ----- reads ----------------------------------------------------------

    def get(self, run_id: str) -> dict[str, Any] | None:
        cur = self._conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,))
        row = cur.fetchone()
        return _row_to_dict(row) if row else None

    def list(
        self,
        *,
        mode: str | None = None,
        status: str | None = None,
        engine_contains: str | None = None,
        since_iso: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        where: list[str] = []
        params: list[Any] = []
        if mode:
            where.append("mode = ?")
            params.append(mode)
        if status:
            where.append("status = ?")
            params.append(status)
        if engine_contains:
            where.append("engine LIKE ?")
            params.append(f"%{engine_contains}%")
        if since_iso:
            where.append("created_at >= ?")
            params.append(since_iso)
        sql = "SELECT * FROM runs"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(int(limit))
        cur = self._conn.execute(sql, params)
        return [_row_to_dict(r) for r in cur.fetchall()]


def open_default_store(settings) -> RunStore:
    """Default location: ``{OUTPUT_DIR}/runstore.db``."""
    return RunStore(settings.output_dir / "runstore.db")


@contextmanager
def open_store(settings) -> Iterator[RunStore]:
    store = open_default_store(settings)
    try:
        yield store
    finally:
        store.close()
