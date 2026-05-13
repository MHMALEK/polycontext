"""FastAPI surface — wires the same Pipeline factory the CLI uses.

Endpoints:
    GET  /health
    POST /ask        - Q&A via Sourcebot (or experimental local agent).
    POST /decompose  - ticket → subtasks (cheap/deep/auto).

Both endpoints accept the same JSON shape as the CLI flags they mirror and
return the EngineResult plus rendered markdown. Auth is intentionally not
implemented yet — out of scope for v1.
"""
from __future__ import annotations

from typing import Any, Literal

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import get_settings
from .core.context import RunContext
from .core.factory import build_ask_pipeline, build_decompose_pipeline, build_metrics_observer
from .core.runstore import RunStore, open_default_store


class _LiveProgressObserver:
    """Wraps a base MetricsObserver and pushes live stage progress to the
    run store so the UI can poll ``/runs/{id}`` while a request is in flight.
    Forwards every call to the underlying observer unchanged.
    """

    def __init__(self, base: Any, store: RunStore, run_id: str) -> None:
        self._base = base
        self._store = store
        self._run_id = run_id
        self._done: list[str] = []

    def on_stage_start(self, *, run_id: str, stage: str, strategy: str) -> None:
        # Display label is "<stage>(<strategy>)" — short, debuggable, and
        # matches the JSONL metric rows.
        label = f"{stage}:{strategy}"
        try:
            self._store.update_progress(
                run_id=run_id, current_stage=label, stages_done=self._done,
            )
        except Exception:
            pass
        starter = getattr(self._base, "on_stage_start", None)
        if starter is not None:
            try:
                starter(run_id=run_id, stage=stage, strategy=strategy)
            except Exception:
                pass

    def on_stage_complete(self, *, run_id: str, stage: str, strategy: str, **kw: Any) -> None:
        self._done.append(f"{stage}:{strategy}")
        try:
            self._store.update_progress(
                run_id=run_id, current_stage=None, stages_done=self._done,
            )
        except Exception:
            pass
        self._base.on_stage_complete(run_id=run_id, stage=stage, strategy=strategy, **kw)

    def on_run_complete(self, **kw: Any) -> None:
        self._base.on_run_complete(**kw)

app = FastAPI(title="tech-decomposition", version="0.2.0")

# Permissive CORS for the bundled UI + local dev. Tighten in prod via a
# reverse proxy or by replacing this with an explicit origin allowlist.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# If a pre-built UI exists, mount it at /ui. Path resolution order:
#   1. UI_DIST_DIR env var (set in the Docker image to /app/web/dist)
#   2. web/dist relative to the current working directory (dev mode)
# Mount is silently skipped if neither exists, so the API runs without the UI.
import os as _os
_ui_env = _os.environ.get("UI_DIST_DIR")
_ui_dist = Path(_ui_env) if _ui_env else (Path.cwd() / "web" / "dist")
if _ui_dist.is_dir():
    app.mount("/ui", StaticFiles(directory=str(_ui_dist), html=True), name="ui")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# /ask
# ---------------------------------------------------------------------------


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    engine: Literal["sourcebot", "local"] = "sourcebot"
    max_steps: int | None = Field(default=None, ge=1, le=50)
    structure_responses: bool = True
    write_file: bool = False
    format: Literal["markdown", "html", "text"] = "markdown"
    repos: list[str] | None = None


class AskResponse(BaseModel):
    engine: str
    answer: str
    citations: list[dict[str, Any]] = Field(default_factory=list)
    model: str | None = None
    transport: str | None = None
    wall_seconds: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    markdown_path: str | None = None
    run_id: str


@app.post("/ask", response_model=AskResponse)
async def ask_endpoint(req: AskRequest) -> AskResponse:
    settings = get_settings()
    try:
        pipeline = build_ask_pipeline(
            settings,
            engine=req.engine,
            max_steps=req.max_steps,
            structure_responses=req.structure_responses,
            output_format=req.format,
            include_cli_sink=False,
            include_file_sink=req.write_file,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"{type(e).__name__}: {e}") from e

    base_obs = build_metrics_observer(settings, mode="ask")
    store = open_default_store(settings)
    ctx = RunContext(settings=settings, mode="ask")
    ctx.metrics = _LiveProgressObserver(base_obs, store, ctx.run_id)
    try:
        # Pre-seed a ``running`` row so the UI can find this run and poll
        # progress before the pipeline returns.
        store.start(
            run_id=ctx.run_id, mode="ask",
            input_ref=req.question,
            input_preview=req.question[:160].strip(),
            output_format=req.format,
        )
        try:
            run = await pipeline.run(req.question, ctx)
        except Exception as e:
            store.record(run_id=ctx.run_id, mode="ask", status="failed",
                         input_ref=req.question, output_format=req.format,
                         error=f"{type(e).__name__}: {e}")
            raise HTTPException(status_code=502, detail=f"{type(e).__name__}: {e}") from e
        store.record(run_id=ctx.run_id, mode="ask", status="completed",
                     input_ref=req.question, output_format=req.format, pipeline_run=run)
    finally:
        store.close()

    er = run.engine_result
    md_path = next(
        (sr.location for sr in run.sink_results if sr.sink == "file" and sr.location),
        None,
    )
    return AskResponse(
        engine=er.engine,
        answer=er.answer_markdown,
        citations=er.citations,
        model=er.model,
        transport=er.transport,
        wall_seconds=er.wall_seconds,
        input_tokens=er.input_tokens,
        output_tokens=er.output_tokens,
        cost_usd=er.cost_usd,
        markdown_path=md_path,
        run_id=ctx.run_id,
    )


# ---------------------------------------------------------------------------
# /decompose
# ---------------------------------------------------------------------------


class DecomposeBody(BaseModel):
    ticket_key: str | None = None
    ticket_url: str | None = None
    ticket_text: str | None = None
    repos: list[str] | None = None
    mode: Literal["cheap", "deep", "auto"] = "auto"
    post_to_jira: bool = False
    format: Literal["markdown", "html", "text"] = "markdown"


class DecomposeAPIResponse(BaseModel):
    engine: str
    decomposition: dict[str, Any]
    markdown: str
    markdown_path: str | None = None
    jira_comment_id: str | None = None
    affected_repos: list[str] = Field(default_factory=list)
    subtask_count: int = 0
    wall_seconds: float
    cost_usd: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    run_id: str


@app.post("/decompose", response_model=DecomposeAPIResponse)
async def decompose_endpoint(req: DecomposeBody) -> DecomposeAPIResponse:
    if not (req.ticket_key or req.ticket_url or req.ticket_text):
        raise HTTPException(
            status_code=400,
            detail="Provide one of: ticket_key, ticket_url, ticket_text.",
        )

    settings = get_settings()
    if req.ticket_text:
        # /decompose with literal text writes the text to a temp file and feeds
        # the TextFileSource. Avoids adding a new "raw_text" InputSource for now.
        import tempfile
        from pathlib import Path

        tmp = Path(tempfile.mkstemp(suffix=".txt")[1])
        tmp.write_text(req.ticket_text)
        source = "text_file"
        ref: str | dict[str, Any] = {
            "path": str(tmp), "key": req.ticket_key, "url": req.ticket_url,
        }
    else:
        source = "jira"
        ref = {"key": req.ticket_key, "url": req.ticket_url}

    pipeline = build_decompose_pipeline(
        settings,
        source=source,
        mode=req.mode,
        repos=req.repos,
        post_to_jira=req.post_to_jira,
        output_format=req.format,
        include_cli_sink=False,
        include_file_sink=True,
    )
    obs = build_metrics_observer(settings, mode="decompose")
    ctx = RunContext(settings=settings, mode="decompose", metrics=obs)
    store = open_default_store(settings)
    try:
        try:
            run = await pipeline.run(ref, ctx)
        except Exception as e:
            store.record(run_id=ctx.run_id, mode="decompose", status="failed",
                         input_ref=ref, output_format=req.format,
                         error=f"{type(e).__name__}: {e}")
            raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}") from e
        store.record(run_id=ctx.run_id, mode="decompose", status="completed",
                     input_ref=ref, output_format=req.format, pipeline_run=run)
    finally:
        store.close()

    er = run.engine_result
    md_path = next(
        (sr.location for sr in run.sink_results if sr.sink == "file" and sr.location),
        None,
    )
    jira_sr = next((sr for sr in run.sink_results if sr.sink == "jira_comment"), None)
    return DecomposeAPIResponse(
        engine=er.engine,
        decomposition=er.payload.get("decomposition") or {},
        markdown=er.answer_markdown,
        markdown_path=md_path,
        jira_comment_id=(jira_sr.extra.get("comment_id") if jira_sr else None),
        affected_repos=er.extra.get("affected_repos") or [],
        subtask_count=er.extra.get("subtask_count") or 0,
        wall_seconds=run.total_seconds,
        cost_usd=er.cost_usd,
        input_tokens=er.input_tokens,
        output_tokens=er.output_tokens,
        run_id=ctx.run_id,
    )


# ---------------------------------------------------------------------------
# /runs — history
# ---------------------------------------------------------------------------


class RunListItem(BaseModel):
    id: str
    mode: str
    status: str
    engine: str | None = None
    model: str | None = None
    total_seconds: float | None = None
    total_cost_usd: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    input_preview: str | None = None
    output_format: str | None = None
    created_at: str
    completed_at: str | None = None
    current_stage: str | None = None
    stages_done: list[str] | None = None


class RunDetail(RunListItem):
    answer: str | None = None
    citations: list[dict[str, Any]] = Field(default_factory=list)
    payload: dict[str, Any] = Field(default_factory=dict)
    input_ref: Any = None
    error: str | None = None


@app.get("/runs", response_model=list[RunListItem])
def list_runs(
    mode: str | None = None,
    status: str | None = None,
    engine: str | None = None,
    since: str | None = None,
    limit: int = 20,
) -> list[RunListItem]:
    """List recent runs. ``since`` is an ISO-8601 timestamp."""
    settings = get_settings()
    store = open_default_store(settings)
    try:
        rows = store.list(
            mode=mode, status=status, engine_contains=engine,
            since_iso=since, limit=limit,
        )
    finally:
        store.close()
    return [RunListItem(**{k: v for k, v in r.items() if k in RunListItem.model_fields}) for r in rows]


@app.get("/runs/{run_id}", response_model=RunDetail)
def get_run(run_id: str) -> RunDetail:
    settings = get_settings()
    store = open_default_store(settings)
    try:
        row = store.get(run_id)
    finally:
        store.close()
    if not row:
        raise HTTPException(status_code=404, detail="run not found")
    return RunDetail(**{k: v for k, v in row.items() if k in RunDetail.model_fields})


@app.post("/runs/{run_id}/replay")
async def replay_run(run_id: str) -> dict[str, Any]:
    """Re-run a prior request. Returns the new run_id; poll /runs/{new_id}."""
    settings = get_settings()
    store = open_default_store(settings)
    try:
        row = store.get(run_id)
    finally:
        store.close()
    if not row:
        raise HTTPException(status_code=404, detail="run not found")
    mode = row["mode"]
    ref = row.get("input_ref")
    fmt = row.get("output_format") or "markdown"
    if mode == "ask":
        question = ref if isinstance(ref, str) else (ref.get("question") if isinstance(ref, dict) else "")
        resp = await ask_endpoint(AskRequest(question=question, format=fmt))
        return {"replayed_from": run_id, "new_run_id": resp.run_id}
    if mode == "decompose":
        body = DecomposeBody(
            ticket_key=(ref or {}).get("key") if isinstance(ref, dict) else None,
            ticket_url=(ref or {}).get("url") if isinstance(ref, dict) else None,
            format=fmt,
        )
        resp = await decompose_endpoint(body)
        return {"replayed_from": run_id, "new_run_id": resp.run_id}
    raise HTTPException(status_code=400, detail=f"replay not supported for mode {mode!r}")
