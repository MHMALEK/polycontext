"""FastAPI surface — every code-agent request routes through an adapter.

Endpoints:
    GET  /health
    GET  /v1/adapters
    POST /v1/adapters/{name}/{ask|decompose|implement}
    POST /v1/bakeoff/{job}
    GET  /runs, /runs/{id}, POST /runs/{id}/replay

There is no ``/ask`` or ``/decompose`` HTTP surface — those existed when
this app shipped its own Sourcebot+Gemini pipeline. The pipeline still
exists for the CLI (``cli.py``), but the HTTP API is adapter-only so the
UI and external callers have a single, uniform shape to target.
"""
from __future__ import annotations

import asyncio
from typing import Any, Literal

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .adapters import (
    AdapterAskInput,
    AdapterDecomposeInput,
    AdapterImplementInput,
    NotSupported,
    get_adapter,
    list_adapters,
)
from .config import get_settings
from .core.context import RunContext
from .core.implement_runner import run_implement
from .core.runstore import RunStore, open_default_store


def _storage_ask_ref(inp: AdapterAskInput, thread_key: str) -> dict[str, Any]:
    d = inp.model_dump()
    d["thread_id"] = thread_key
    return d


def _augmented_ask_query_for_thread(
    store: RunStore,
    thread_id: str,
    exclude_run_id: str,
    user_query: str,
) -> str:
    """Fold prior turns into one prompt so adapters stay single-shot."""
    rows = store.list_thread_completed(thread_id, exclude_run_id=exclude_run_id)
    if not rows:
        return user_query
    parts: list[str] = [
        "You are continuing an existing conversation. Use the prior turns for context. "
        "Answer only the final user message at the end.\n",
    ]
    for i, row in enumerate(rows, 1):
        ref = row.get("input_ref")
        q = ""
        if isinstance(ref, dict) and isinstance(ref.get("query"), str):
            q = ref["query"].strip()
        if not q:
            q = (row.get("input_preview") or "").strip()
        ans = (row.get("answer") or "").strip()
        parts.append(f"--- Turn {i} ---\nUser: {q}\nAssistant:\n{ans}\n")
    parts.append(f"--- Current ---\nUser: {user_query.strip()}")
    return "\n".join(parts)


def _adapter_progress_started(store: RunStore, run_id: str, *, decompose: bool) -> None:
    """Set runstore progress columns so /runs polling can drive the UI timeline.

    Adapter calls are opaque (no per-stage hooks), so we emit coarse stages
    that match the web client's ``input`` / ``enrich`` / ``engine`` / ``render`` keys.
    """
    store.update_progress(
        run_id=run_id,
        current_stage="engine:decompose" if decompose else "engine:structured",
        stages_done=["input:blocking", "enrich:(none)"],
    )


def _adapter_progress_finishing(store: RunStore, run_id: str, *, decompose: bool) -> None:
    """Last hop before the row is finalized — brief ``render`` state for pollers."""
    engine_tag = "engine:decompose" if decompose else "engine:structured"
    store.update_progress(
        run_id=run_id,
        current_stage="render:markdown",
        stages_done=["input:blocking", "enrich:(none)", engine_tag],
    )


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
    thread_id: str | None = None
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
    per_thread: bool = False,
    thread_id: str | None = None,
    order: Literal["asc", "desc"] = "desc",
) -> list[RunListItem]:
    """List recent runs. ``since`` is an ISO-8601 timestamp.

    ``per_thread`` (ask mode): return only the latest row per chat thread.
    ``thread_id``: filter to one conversation; use with ``order=asc`` for transcript order.
    """
    settings = get_settings()
    store = open_default_store(settings)
    try:
        if per_thread and mode in (None, "ask"):
            rows = store.list_latest_per_thread(mode=mode or "ask", limit=limit)
        elif thread_id is not None:
            rows = store.list(
                mode=mode,
                status=status,
                engine_contains=engine,
                since_iso=since,
                thread_id=thread_id,
                order_desc=(order == "desc"),
                limit=limit,
            )
        else:
            rows = store.list(
                mode=mode, status=status, engine_contains=engine,
                since_iso=since, limit=limit,
            )
    finally:
        store.close()
    return [RunListItem(**{k: v for k, v in r.items() if k in RunListItem.model_fields}) for r in rows]


@app.get("/runs/thread/{thread_id}", response_model=list[RunDetail])
def list_thread_runs(thread_id: str, limit: int = 100) -> list[RunDetail]:
    """All ask turns in a thread (chronological), including answers — for chat UI."""
    settings = get_settings()
    store = open_default_store(settings)
    try:
        if not store.thread_exists(thread_id):
            raise HTTPException(status_code=404, detail="thread not found")
        rows = store.list_by_thread(thread_id, mode="ask", limit=limit)
    finally:
        store.close()
    out: list[RunDetail] = []
    for r in rows:
        out.append(RunDetail(**{k: v for k, v in r.items() if k in RunDetail.model_fields}))
    return out


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
    """Re-run a prior request through the same adapter that handled it.

    The original adapter name is parsed from the row's ``engine`` field
    (stored as ``"<adapter>:<op>"`` by the adapter endpoints). Pre-adapter
    rows whose engine doesn't match that shape can't be replayed and
    return 400 — re-issue the request manually instead.
    """
    settings = get_settings()
    store = open_default_store(settings)
    try:
        row = store.get(run_id)
    finally:
        store.close()
    if not row:
        raise HTTPException(status_code=404, detail="run not found")

    mode = row["mode"]
    engine = row.get("engine") or ""
    ref = row.get("input_ref")

    # Expected engine shape from the adapter endpoints: "<adapter>:<op>".
    adapter_name = engine.split(":", 1)[0] if ":" in engine else ""
    if not adapter_name:
        raise HTTPException(
            status_code=400,
            detail=f"cannot replay run with engine={engine!r}; only adapter-routed runs are replayable",
        )

    if mode == "ask":
        if not isinstance(ref, dict):
            raise HTTPException(status_code=400, detail="run input_ref is not an adapter ask payload")
        inp = AdapterAskInput.model_validate(ref)
        # Re-run as a single turn so the adapter is not fed a duplicate transcript.
        inp = inp.model_copy(update={"thread_id": None})
        resp = await adapter_ask(adapter_name, inp)
        return {"replayed_from": run_id, "new_run_id": resp["run_id"]}
    if mode == "decompose":
        if not isinstance(ref, dict):
            raise HTTPException(status_code=400, detail="run input_ref is not an adapter decompose payload")
        inp = AdapterDecomposeInput.model_validate(ref)
        resp = await adapter_decompose(adapter_name, inp)
        return {"replayed_from": run_id, "new_run_id": resp["run_id"]}
    raise HTTPException(status_code=400, detail=f"replay not supported for mode {mode!r}")


# ---------------------------------------------------------------------------
# /v1/adapters — bake-off harness
# ---------------------------------------------------------------------------


def _resolve_adapter(name: str):
    """Look up the adapter, applying the ``enabled_adapters`` allowlist."""
    settings = get_settings()
    allowlist = settings.enabled_adapters
    if allowlist and name not in allowlist:
        raise HTTPException(status_code=404, detail=f"adapter {name!r} not enabled")
    try:
        return get_adapter(name, settings)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown adapter: {name!r}")
    except ImportError as e:
        raise HTTPException(status_code=503, detail=f"adapter not installed: {e}") from e


@app.get("/v1/adapters")
def list_adapters_endpoint() -> dict[str, Any]:
    settings = get_settings()
    items = list_adapters(settings)
    if settings.enabled_adapters:
        items = [i for i in items if i["name"] in settings.enabled_adapters]
    return {"adapters": items}


@app.post("/v1/adapters/{name}/ask")
async def adapter_ask(name: str, inp: AdapterAskInput) -> dict[str, Any]:
    adapter = _resolve_adapter(name)
    settings = get_settings()
    store = open_default_store(settings)
    ctx = RunContext(settings=settings, mode="ask")
    try:
        thread_key = inp.thread_id or ctx.run_id
        if inp.thread_id and not store.thread_exists(inp.thread_id):
            raise HTTPException(status_code=404, detail="thread not found")

        ref_for_store = _storage_ask_ref(inp, thread_key)
        q_preview = inp.query[:160].strip()
        aug_query = _augmented_ask_query_for_thread(
            store, thread_key, ctx.run_id, inp.query,
        )
        adapter_inp = inp.model_copy(update={"query": aug_query})

        # Pre-seed a ``running`` row so the UI can find this run and poll
        # progress while the adapter is still executing.
        store.start(
            run_id=ctx.run_id, mode="ask",
            input_ref=ref_for_store,
            input_preview=q_preview,
            output_format="markdown",
            thread_id=thread_key,
        )
        _adapter_progress_started(store, ctx.run_id, decompose=False)
        try:
            result = await adapter.ask(adapter_inp)
        except NotSupported as e:
            store.record(
                run_id=ctx.run_id, mode="ask", status="failed",
                input_ref=ref_for_store, engine=f"{name}:ask",
                error=str(e), input_preview=q_preview,
                thread_id=thread_key,
            )
            raise HTTPException(status_code=501, detail=str(e)) from e
        except Exception as e:
            store.record(
                run_id=ctx.run_id, mode="ask", status="failed",
                input_ref=ref_for_store, engine=f"{name}:ask",
                error=f"{type(e).__name__}: {e}", input_preview=q_preview,
                thread_id=thread_key,
            )
            raise HTTPException(status_code=502, detail=f"{type(e).__name__}: {e}") from e
        _adapter_progress_finishing(store, ctx.run_id, decompose=False)
        m = result.metrics
        store.record(
            run_id=ctx.run_id, mode="ask", status="completed",
            input_ref=ref_for_store, engine=f"{name}:ask",
            answer=result.answer,
            citations=list(result.citations),
            model=m.model,
            total_seconds=(m.duration_ms / 1000.0) if m.duration_ms else None,
            total_cost_usd=m.cost_usd,
            input_tokens=m.tokens_in,
            output_tokens=m.tokens_out,
            input_preview=q_preview,
            thread_id=thread_key,
        )
    finally:
        store.close()
    return {"run_id": ctx.run_id, "thread_id": thread_key, "result": result.model_dump()}


@app.post("/v1/adapters/{name}/decompose")
async def adapter_decompose(name: str, inp: AdapterDecomposeInput) -> dict[str, Any]:
    if not inp.query:
        raise HTTPException(status_code=400, detail="Provide a query.")
    adapter = _resolve_adapter(name)
    store = open_default_store(get_settings())
    ctx = RunContext(settings=get_settings(), mode="decompose")
    preview = (inp.query or "")[:160].strip()
    try:
        store.start(
            run_id=ctx.run_id, mode="decompose",
            input_ref=inp.model_dump(),
            input_preview=preview,
            output_format="markdown",
        )
        _adapter_progress_started(store, ctx.run_id, decompose=True)
        try:
            result = await adapter.decompose(inp)
        except NotSupported as e:
            store.record(
                run_id=ctx.run_id, mode="decompose", status="failed",
                input_ref=inp.model_dump(), engine=f"{name}:decompose",
                error=str(e), input_preview=preview,
            )
            raise HTTPException(status_code=501, detail=str(e)) from e
        except Exception as e:
            store.record(
                run_id=ctx.run_id, mode="decompose", status="failed",
                input_ref=inp.model_dump(), engine=f"{name}:decompose",
                error=f"{type(e).__name__}: {e}", input_preview=preview,
            )
            raise HTTPException(status_code=502, detail=f"{type(e).__name__}: {e}") from e
        _adapter_progress_finishing(store, ctx.run_id, decompose=True)
        dm = result.metrics
        store.record(
            run_id=ctx.run_id, mode="decompose", status="completed",
            input_ref=inp.model_dump(), engine=f"{name}:decompose",
            answer=result.markdown or "",
            citations=[],
            payload=result.decomposition.model_dump(mode="json"),
            model=dm.model,
            total_seconds=(dm.duration_ms / 1000.0) if dm.duration_ms else None,
            total_cost_usd=dm.cost_usd,
            input_tokens=dm.tokens_in,
            output_tokens=dm.tokens_out,
            input_preview=preview,
        )
    finally:
        store.close()
    return {"run_id": ctx.run_id, "result": result.model_dump()}


@app.post("/v1/adapters/{name}/implement")
async def adapter_implement(name: str, inp: AdapterImplementInput) -> dict[str, Any]:
    adapter = _resolve_adapter(name)
    if not adapter.supports("implement"):
        raise HTTPException(status_code=501, detail=f"adapter {name!r} does not support implement")
    store = open_default_store(get_settings())
    ctx = RunContext(settings=get_settings(), mode="implement")
    try:
        try:
            result = await run_implement(
                adapter=adapter, inp=inp, settings=get_settings(), run_id=ctx.run_id,
            )
        except NotSupported as e:
            raise HTTPException(status_code=501, detail=str(e)) from e
        except Exception as e:
            store.record(run_id=ctx.run_id, mode="implement", status="failed",
                         input_ref=inp.model_dump(), engine=f"{name}:implement",
                         error=f"{type(e).__name__}: {e}")
            raise HTTPException(status_code=502, detail=f"{type(e).__name__}: {e}") from e
        imp_lines: list[str] = []
        if result.mr_url:
            imp_lines.append(f"**Merge request:** {result.mr_url}")
        if result.branch:
            imp_lines.append(f"**Branch:** `{result.branch}`")
        if result.files_changed:
            tail = ", ".join(f"`{f}`" for f in result.files_changed[:24])
            imp_lines.append(f"**Files:** {tail}")
        if result.diff_summary:
            imp_lines.append(result.diff_summary.strip())
        imp_body = "\n\n".join(imp_lines) if imp_lines else "_(implement completed)_"
        im = result.metrics
        store.record(
            run_id=ctx.run_id, mode="implement", status="completed",
            input_ref=inp.model_dump(), engine=f"{name}:implement",
            answer=imp_body,
            citations=[],
            payload={
                "mr_url": result.mr_url,
                "branch": result.branch,
                "commits": result.commits,
                "files_changed": result.files_changed,
            },
            model=im.model,
            total_seconds=(im.duration_ms / 1000.0) if im.duration_ms else None,
            total_cost_usd=im.cost_usd,
            input_tokens=im.tokens_in,
            output_tokens=im.tokens_out,
        )
    finally:
        store.close()
    return {"run_id": ctx.run_id, "result": result.model_dump()}


# ---------------------------------------------------------------------------
# /v1/bakeoff — fan-out across N adapters
# ---------------------------------------------------------------------------


class BakeoffBody(BaseModel):
    adapters: list[str] = Field(min_length=1)
    ask: AdapterAskInput | None = None
    decompose: AdapterDecomposeInput | None = None
    implement: AdapterImplementInput | None = None


@app.post("/v1/bakeoff/{job}")
async def bakeoff(job: Literal["ask", "decompose", "implement"], body: BakeoffBody) -> dict[str, Any]:
    """Run the same input through every named adapter sequentially.

    Sequential rather than parallel because some backends share resources
    (Sourcebot, the local SQLite store, your Anthropic rate limit). Each
    adapter's response is independent; failures don't short-circuit the
    others.
    """
    inp = getattr(body, job)
    if inp is None:
        raise HTTPException(status_code=400, detail=f"body.{job} is required for job={job!r}")

    results: list[dict[str, Any]] = []
    settings = get_settings()
    for name in body.adapters:
        try:
            adapter = _resolve_adapter(name)
        except HTTPException as e:
            results.append({"adapter": name, "ok": False, "error": e.detail, "status": e.status_code})
            continue
        try:
            if job == "ask":
                r = await adapter.ask(inp)
            elif job == "decompose":
                r = await adapter.decompose(inp)
            else:
                r = await run_implement(adapter=adapter, inp=inp, settings=settings,
                                         run_id=f"bo-{name}-{asyncio.get_event_loop().time():.0f}")
            results.append({"adapter": name, "ok": True, "result": r.model_dump()})
        except NotSupported as e:
            results.append({"adapter": name, "ok": False, "error": str(e), "status": 501})
        except Exception as e:  # noqa: BLE001 — surface, don't break the loop
            results.append({"adapter": name, "ok": False, "error": f"{type(e).__name__}: {e}", "status": 502})
    return {"job": job, "results": results}
