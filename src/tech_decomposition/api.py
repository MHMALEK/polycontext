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
from .core.runstore import open_default_store


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
    store = open_default_store(get_settings())
    ctx = RunContext(settings=get_settings(), mode="ask")
    try:
        # Pre-seed a ``running`` row so the UI can find this run and poll
        # progress while the adapter is still executing.
        store.start(
            run_id=ctx.run_id, mode="ask",
            input_ref=inp.model_dump(),
            input_preview=inp.query[:160].strip(),
            output_format="markdown",
        )
        try:
            result = await adapter.ask(inp)
        except NotSupported as e:
            store.record(run_id=ctx.run_id, mode="ask", status="failed",
                         input_ref=inp.model_dump(), engine=f"{name}:ask",
                         error=str(e))
            raise HTTPException(status_code=501, detail=str(e)) from e
        except Exception as e:
            store.record(run_id=ctx.run_id, mode="ask", status="failed",
                         input_ref=inp.model_dump(), engine=f"{name}:ask",
                         error=f"{type(e).__name__}: {e}")
            raise HTTPException(status_code=502, detail=f"{type(e).__name__}: {e}") from e
        store.record(run_id=ctx.run_id, mode="ask", status="completed",
                     input_ref=inp.model_dump(), engine=f"{name}:ask")
    finally:
        store.close()
    return {"run_id": ctx.run_id, "result": result.model_dump()}


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
        try:
            result = await adapter.decompose(inp)
        except NotSupported as e:
            store.record(run_id=ctx.run_id, mode="decompose", status="failed",
                         input_ref=inp.model_dump(), engine=f"{name}:decompose",
                         error=str(e))
            raise HTTPException(status_code=501, detail=str(e)) from e
        except Exception as e:
            store.record(run_id=ctx.run_id, mode="decompose", status="failed",
                         input_ref=inp.model_dump(), engine=f"{name}:decompose",
                         error=f"{type(e).__name__}: {e}")
            raise HTTPException(status_code=502, detail=f"{type(e).__name__}: {e}") from e
        store.record(run_id=ctx.run_id, mode="decompose", status="completed",
                     input_ref=inp.model_dump(), engine=f"{name}:decompose")
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
        store.record(run_id=ctx.run_id, mode="implement", status="completed",
                     input_ref=inp.model_dump(), engine=f"{name}:implement")
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
