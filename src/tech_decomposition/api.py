"""FastAPI surface — every code-agent request routes through an adapter.

Endpoints:
    GET  /health
    GET  /v1/adapters
    POST /v1/adapters/{name}/{ask|decompose}
    POST /v1/bakeoff/{job}
    GET  /runs, /runs/{id}, POST /runs/{id}/replay

There is no ``/ask`` or ``/decompose`` HTTP surface — those existed when
this app shipped its own Sourcebot+Gemini pipeline. The **in-process**
decompose pipeline (``core.factory.build_decompose_pipeline``) remains for
tests and programmatic use; the HTTP API is adapter-only so the UI and
external callers have a single, uniform shape to target.
"""
from __future__ import annotations

import asyncio
from typing import Any, Literal

from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .adapters import (
    AdapterAskInput,
    AdapterDecomposeInput,
    NotSupported,
    get_adapter,
    list_adapters,
)
from .config import get_settings
from .core.context import RunContext
from .core.grounding import (
    GroundedContext,
    build_grounded_prompt,
    retrieve_grounded_context,
)
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


@app.get("/v1/adapters/{name}/models")
def list_adapter_models_endpoint(
    name: str,
    include_openrouter: bool = True,
) -> dict[str, Any]:
    """Combined model catalog for the named adapter.

    Returns two sections:
    - **curated**: hand-maintained list from ``adapters/_models_catalog.py``
      with annotations (price, caveats, eval notes).
    - **openrouter_live**: ALL ~350 OpenRouter models fetched live, deduped
      against the curated list. Only included when the adapter accepts
      OpenRouter-format ids (currently ``opencode`` and ``pipeline``) and
      ``include_openrouter=true`` (default).

    The OpenRouter fetch is cached for 1h in-process — repeated UI page
    loads don't slam the provider. Set ``include_openrouter=false`` to
    skip the live fetch (e.g. when offline).
    """
    from .adapters._models_catalog import models_for_adapter
    from .adapters._openrouter_models import fetch_openrouter_models

    curated = models_for_adapter(name)
    openrouter_live: list[dict[str, Any]] = []
    # opencode uses ``openrouter/<id>`` ids natively. pipeline uses
    # ``openrouter:<id>``; we still surface the same live list but the UI
    # rewrites ``openrouter/foo`` -> ``openrouter:foo`` before sending.
    if include_openrouter and name in {"opencode", "pipeline"}:
        live = fetch_openrouter_models()
        # Dedup: skip ids that already appear in curated.
        curated_ids = {m["id"] for m in curated}
        openrouter_live = [m for m in live if m["id"] not in curated_ids]
    return {
        "adapter": name,
        "models": curated,
        "openrouter_live": openrouter_live,
    }


@app.post("/v1/providers/openrouter/models/refresh")
def refresh_openrouter_models() -> dict[str, Any]:
    """Force a cache bust on the OpenRouter live model list. Useful when
    OpenRouter ships a new model and the in-process TTL hasn't expired."""
    from .adapters._openrouter_models import fetch_openrouter_models

    models = fetch_openrouter_models(force_refresh=True)
    return {"count": len(models), "refreshed_at": _utcnow_iso()}


def _utcnow_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


class GroundingRequest(BaseModel):
    """Standalone grounded-retrieval request — bypasses any adapter."""

    query: str = Field(min_length=1)
    repos: list[str] | None = None
    top_k: int = Field(default=8, ge=1, le=50)
    context_lines: int = Field(default=3, ge=0, le=20)


@app.post("/v1/grounding/retrieve")
async def grounding_retrieve(body: GroundingRequest) -> dict[str, Any]:
    """Run grounded retrieval alone — see the snippets, latency, and char count
    without involving an adapter. Useful for debugging or A/B comparing whether
    grounding is worth injecting on a given question.
    """
    ctx = await retrieve_grounded_context(
        query=body.query,
        settings=get_settings(),
        repos=body.repos,
        top_k=body.top_k,
        context_lines=body.context_lines,
    )
    return ctx.model_dump(mode="json")


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

        # 1) Optional grounded pre-fetch — runs alone, has its own metrics.
        #    Skipped for the pipeline adapter, which does retrieval internally.
        #    Without this guard, the pipeline adapter would receive a query that
        #    already contains a grounding block and re-extract terms FROM that
        #    block — pulling things like {TEST_RATE_LIMIT}/minute out of the
        #    embedded test snippets and using them as search terms.
        grounding: GroundedContext | None = None
        if inp.grounded and name != "pipeline":
            grounding = await retrieve_grounded_context(
                query=inp.query, settings=settings, repos=inp.repos, top_k=inp.top_k,
            )

        # 2) Thread context — fold prior turns in for follow-ups.
        threaded_query = _augmented_ask_query_for_thread(
            store, thread_key, ctx.run_id, inp.query,
        )

        # 3) Prepend the grounding block (if any) to the threaded query.
        final_query = (
            build_grounded_prompt(
                grounding_block=grounding.grounding_block, query=threaded_query,
            )
            if grounding and grounding.grounding_block
            else threaded_query
        )
        adapter_inp = inp.model_copy(update={"query": final_query})

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
        # When grounding ran, surface the retrieved paths into the adapter's
        # metrics so the recall scorer can compute context_recall for ANY
        # adapter — not just the pipeline adapter that builds them itself.
        if grounding is not None:
            from .core.coverage import snippet_paths
            extra = dict(result.metrics.extra or {})
            extra.setdefault("grounding_paths", sorted(snippet_paths(grounding.snippets)))
            result = result.model_copy(update={
                "metrics": result.metrics.model_copy(update={"extra": extra}),
            })
        m = result.metrics
        # Optional rich-card shaper pass — one extra Gemini Flash call that
        # coerces the raw markdown answer into a structured Answer (summary,
        # details, citations, confidence, caveats, next_steps). Always
        # returns gracefully; never raises. Skipped per-request via
        # ``shape=false`` or globally via ANSWER_SHAPER_ENABLED=false.
        shaped = await _maybe_shape_answer(
            settings=settings, query=inp.query, raw_text=result.answer, override=inp.shape,
        )
        # Build the persisted payload so /runs/{id} can replay the full
        # telemetry. Three sources of telemetry live in different places:
        #   - grounding   : when the API itself prefetched snippets
        #                   (skipped for the pipeline adapter, which does
        #                   its own internal grounding instead)
        #   - metrics.extra : per-adapter extras — pipeline route, coverage,
        #                   grounding metrics, grounding_paths, opencode
        #                   tool_names, etc. Without this in payload, the
        #                   UI's Telemetry panel renders empty after reload
        #                   because /runs/{id} responses don't carry it.
        #   - structured_answer : the shaper output the UI's rich card binds to.
        payload: dict[str, Any] = {}
        if grounding is not None:
            payload["grounding"] = grounding.model_dump(mode="json")
        if m.extra:
            payload["metrics_extra"] = dict(m.extra)
        if shaped is not None:
            payload["structured_answer"] = shaped.model_dump(mode="json")
        store.record(
            run_id=ctx.run_id, mode="ask", status="completed",
            input_ref=ref_for_store, engine=f"{name}:ask",
            answer=result.answer,
            citations=list(result.citations),
            payload=payload or None,
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
    result_body = result.model_dump()
    if grounding is not None:
        result_body["grounding"] = grounding.model_dump(mode="json")
    if shaped is not None:
        result_body["structured_answer"] = shaped.model_dump(mode="json")
    return {"run_id": ctx.run_id, "thread_id": thread_key, "result": result_body}


# ---------------------------------------------------------------------------
# Universal native-model streaming
# ---------------------------------------------------------------------------


def _wants_tools_disabled(inp: AdapterAskInput) -> bool:
    """Direct mode (no tools, no grounding) and Grounded-only mode (tools off,
    grounding on) both reduce to a single LLM call — they can be streamed
    natively via pydantic-AI's ``run_stream`` regardless of which adapter
    the user picked. Hybrid + Agent need the SDK's tool loop and can't take
    this path."""
    return not inp.tools_enabled


def _resolve_synthesis_model(inp: AdapterAskInput, adapter_name: str, settings) -> str:
    """Choose the model spec for the native-streaming path.

    Priority:
      1. Explicit ``inp.model`` (per-request override from the UI dropdown).
         Already in ``provider:name`` or adapter-native form — normalized below.
      2. Fall back to the adapter's configured default model.
      3. Last resort: ``pipeline_synthesis_model`` (Flash by default).
    """
    raw = (inp.model or "").strip()
    if not raw:
        # Adapter-specific config keys hold the default. We look them up by
        # name so users picking "claude_code" with no override get
        # claude-sonnet-* instead of gemini-2.5-flash.
        defaults: dict[str, str] = {
            "opencode": settings.opencode_sdk_model,
            "gemini": settings.gemini_sdk_model if hasattr(settings, "gemini_sdk_model") else "gemini-2.5-flash",
            "claude_code": settings.claude_code_model if hasattr(settings, "claude_code_model") else "claude-sonnet-4-5",
            "openai_agents": settings.openai_agents_sdk_model if hasattr(settings, "openai_agents_sdk_model") else "gpt-4o",
            "cursor": settings.cursor_sdk_model if hasattr(settings, "cursor_sdk_model") else "composer-2",
        }
        raw = (defaults.get(adapter_name) or settings.pipeline_synthesis_model or "gemini-2.5-flash").strip()

    # Normalize to a provider:name spec the llm_registry understands.
    # OpenCode-style "anthropic/claude-..." → "anthropic:claude-..."
    # Bare names like "gemini-2.5-pro" stay bare (registry infers gemini).
    if "/" in raw and ":" not in raw:
        provider, name = raw.split("/", 1)
        # OpenCode + OpenRouter both use slashes. If provider is "openrouter",
        # the model id keeps its embedded slash (e.g. openrouter/deepseek/...).
        if provider == "openrouter":
            raw = f"openrouter:{name}"
        elif provider in ("anthropic", "google", "openai"):
            raw = f"{'gemini' if provider == 'google' else provider}:{name}"
    return raw


@app.post("/v1/ask/stream")
async def ask_stream(inp: AdapterAskInput, request: Request) -> StreamingResponse:
    """SSE endpoint that streams a single-shot LLM call directly via
    pydantic-AI when the request is in a no-tools mode (Direct or
    Grounded-only). Works for ANY adapter selection because we bypass the
    SDK entirely — the ``adapter`` parameter only influences which model to
    default to.

    For requests that DO need the agent loop (Hybrid / Agent mode), this
    endpoint returns a 400 directing the caller to the adapter-specific
    streaming endpoint (currently only ``/v1/adapters/opencode/stream``).
    """
    if not _wants_tools_disabled(inp):
        raise HTTPException(
            status_code=400,
            detail=(
                "ask/stream supports only no-tools modes (Direct, Grounded-only). "
                "Use /v1/adapters/opencode/stream for tool-loop streaming."
            ),
        )

    settings = get_settings()
    store = open_default_store(settings)
    ctx = RunContext(settings=settings, mode="ask")
    thread_key = inp.thread_id or ctx.run_id
    q_preview = inp.query[:160].strip()
    ref_for_store = _storage_ask_ref(inp, thread_key)

    # Optional grounding prefetch — same shape as the blocking /ask endpoint.
    grounding: GroundedContext | None = None
    if inp.grounded:
        grounding = await retrieve_grounded_context(
            query=inp.query, settings=settings, repos=inp.repos, top_k=inp.top_k,
        )

    threaded_query = _augmented_ask_query_for_thread(
        store, thread_key, ctx.run_id, inp.query,
    )

    store.start(
        run_id=ctx.run_id, mode="ask",
        input_ref=ref_for_store,
        input_preview=q_preview,
        output_format="markdown",
        thread_id=thread_key,
    )

    # The "adapter" field on the input is just a default-model hint —
    # we always go native (pydantic-AI) regardless.
    adapter_name = (inp.adapter or "pipeline").lower()
    model_spec = _resolve_synthesis_model(inp, adapter_name, settings)
    grounding_block = (grounding.grounding_block if grounding else "")

    import time as _time
    start = _time.monotonic()

    async def event_source():
        yield f"data: {_sse_json({'kind': 'run', 'run_id': ctx.run_id, 'thread_id': thread_key})}\n\n"
        # Emit a synthetic "session" with the model so the UI's bubble
        # has something to show in the header strip.
        yield f"data: {_sse_json({'kind': 'session', 'sessionID': f'native:{model_spec}'})}\n\n"
        yield f"data: {_sse_json({'kind': 'status', 'status': 'running'})}\n\n"

        final_text = ""
        toks_in: int | None = None
        toks_out: int | None = None
        duration_ms = 0
        from .core.synthesize import synthesize_ask_stream
        try:
            async for ev in synthesize_ask_stream(
                query=threaded_query,
                grounding_block=grounding_block,
                settings=settings,
                model_id=model_spec,
            ):
                if await request.is_disconnected():
                    break
                if ev["kind"] == "text.delta":
                    yield f"data: {_sse_json({'kind': 'text.delta', 'text': ev['text']})}\n\n"
                elif ev["kind"] == "result":
                    final_text = ev["text"]
                    toks_in = ev.get("tokens_in")
                    toks_out = ev.get("tokens_out")
                    duration_ms = ev.get("duration_ms") or 0
        except Exception as e:  # noqa: BLE001
            err = {"kind": "error", "error": f"{type(e).__name__}: {e}"}
            yield f"data: {_sse_json(err)}\n\n"
            _persist_failed(store, ctx.run_id, ref_for_store, q_preview, thread_key, err["error"])
            try:
                store.close()
            except Exception:  # noqa: BLE001
                pass
            return

        # Build the done envelope mirroring opencode's shape so the UI
        # client can use the same reducer.
        done_result = {
            "ok": True,
            "answer": final_text,
            "model": model_spec,
            "tokensIn": toks_in,
            "tokensOut": toks_out,
            "costUsd": None,
            "toolCalls": 0,
            "toolNames": [],
            "toolTrace": [],
        }
        yield f"data: {_sse_json({'kind': 'done', 'result': done_result})}\n\n"

        # Optional shaper pass + emit shaped event.
        shaped = await _maybe_shape_answer(
            settings=settings, query=inp.query, raw_text=final_text, override=inp.shape,
        )
        if shaped is not None:
            yield f"data: {_sse_json({'kind': 'shaped', 'shaped': shaped.model_dump(mode='json')})}\n\n"

        # Persist completion the same way the streamed opencode path does.
        try:
            extra: dict[str, Any] = {
                "native_stream": True,
                "synthesis_model": model_spec,
                "wall_ms_total": int((_time.monotonic() - start) * 1000),
                "synth_duration_ms": duration_ms,
            }
            payload: dict[str, Any] = {"metrics_extra": extra}
            if grounding is not None:
                payload["grounding"] = grounding.model_dump(mode="json")
                from .core.coverage import snippet_paths
                extra["grounding_paths"] = sorted(snippet_paths(grounding.snippets))
            if shaped is not None:
                payload["structured_answer"] = shaped.model_dump(mode="json")
            s2 = open_default_store(settings)
            try:
                s2.record(
                    run_id=ctx.run_id, mode="ask", status="completed",
                    input_ref=ref_for_store, engine=f"{adapter_name}:ask:native_stream",
                    answer=final_text,
                    citations=[],
                    payload=payload,
                    model=model_spec,
                    total_seconds=duration_ms / 1000.0 if duration_ms else None,
                    total_cost_usd=None,
                    input_tokens=toks_in,
                    output_tokens=toks_out,
                    input_preview=q_preview,
                    thread_id=thread_key,
                )
            finally:
                s2.close()
        except Exception:  # noqa: BLE001
            # Persistence failure shouldn't surface to the user — they
            # already have the answer. The runstore reload will just miss
            # this row.
            pass
        try:
            store.close()
        except Exception:  # noqa: BLE001
            pass

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# OpenCode streaming + abort
# ---------------------------------------------------------------------------


class OpencodeAbortRequest(BaseModel):
    session_id: str
    repos: list[str] | None = None


@app.post("/v1/adapters/opencode/stream")
async def opencode_stream(inp: AdapterAskInput, request: Request) -> StreamingResponse:
    """SSE bridge that proxies agent-node ``streamOpencode`` straight to the
    browser. Each event line is ``data: <json>\\n\\n``; the terminal event is
    ``{kind:"done", result: {...}}``. The recorded run row is written under
    the same shape ``/v1/adapters/{name}/ask`` produces — so /runs and the
    history sidebar pick it up without changes.
    """
    from .adapters._opencode_sdk import OpencodeSDKAdapter

    adapter = _resolve_adapter("opencode")
    if not isinstance(adapter, OpencodeSDKAdapter):
        raise HTTPException(status_code=500, detail="opencode adapter not registered")
    settings = get_settings()
    store = open_default_store(settings)
    ctx = RunContext(settings=settings, mode="ask")

    # Pre-fetch grounding identically to /ask so the UI can keep its modes.
    grounding: GroundedContext | None = None
    if inp.grounded:
        grounding = await retrieve_grounded_context(
            query=inp.query, settings=settings, repos=inp.repos, top_k=inp.top_k,
        )

    thread_key = inp.thread_id or ctx.run_id
    threaded_query = _augmented_ask_query_for_thread(
        store, thread_key, ctx.run_id, inp.query,
    )
    final_query = (
        build_grounded_prompt(grounding_block=grounding.grounding_block, query=threaded_query)
        if grounding and grounding.grounding_block
        else threaded_query
    )
    adapter_inp = inp.model_copy(update={"query": final_query})

    # When this is a thread continuation, try to reuse the previous OpenCode
    # session id so the SDK keeps conversation memory + warm prompt cache.
    reuse_session_id = (
        _latest_opencode_session_for_thread(store, thread_key)
        if inp.thread_id
        else None
    )
    ref_for_store = _storage_ask_ref(inp, thread_key)
    q_preview = inp.query[:160].strip()
    store.start(
        run_id=ctx.run_id, mode="ask",
        input_ref=ref_for_store,
        input_preview=q_preview,
        output_format="markdown",
        thread_id=thread_key,
    )
    _adapter_progress_started(store, ctx.run_id, decompose=False)
    import time as _time
    start = _time.monotonic()

    async def event_source():
        # Emit a tiny preamble so the UI can render shell chrome before
        # opencode's session even starts.
        yield f"data: {_sse_json({'kind': 'run', 'run_id': ctx.run_id, 'thread_id': thread_key})}\n\n"
        final_event: dict[str, Any] | None = None
        try:
            async for ev in adapter.astream(
                adapter_inp,
                session_id=reuse_session_id,
                keep_session=bool(inp.thread_id),
            ):
                if await request.is_disconnected():
                    break
                yield f"data: {_sse_json(ev)}\n\n"
                if ev.get("kind") == "done":
                    final_event = ev
                    break
        except Exception as e:  # noqa: BLE001
            err = {"kind": "error", "error": f"{type(e).__name__}: {e}"}
            yield f"data: {_sse_json(err)}\n\n"
            _persist_failed(store, ctx.run_id, ref_for_store, q_preview, thread_key, err["error"])
            return
        finally:
            try:
                store.close()
            except Exception:  # noqa: BLE001
                pass

        # Persist on completion so reload picks the row up. We re-open the
        # store because the `finally` above has closed it. This mirrors the
        # blocking /ask handler — same payload shape (metrics_extra,
        # citations, structured_answer).
        if final_event and (final_event.get("result") or {}).get("ok"):
            result = final_event["result"]
            duration_ms = int((_time.monotonic() - start) * 1000)
            # Shape the final answer the same way the blocking endpoint does,
            # then emit ONE more SSE event so the UI can swap to the rich
            # card immediately — without it the client would have to refetch
            # the run row to discover the structured_answer.
            shaped = await _maybe_shape_answer(
                settings=settings,
                query=inp.query,
                raw_text=result.get("answer") or "",
                override=inp.shape,
            )
            if shaped is not None:
                yield f"data: {_sse_json({'kind': 'shaped', 'shaped': shaped.model_dump(mode='json')})}\n\n"
            _persist_streamed_ok(
                run_id=ctx.run_id,
                thread_key=thread_key,
                ref_for_store=ref_for_store,
                q_preview=q_preview,
                grounding=grounding,
                result=result,
                duration_ms=duration_ms,
                shaped=shaped,
            )

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/v1/adapters/{name}/stream")
async def adapter_stream(name: str, inp: AdapterAskInput, request: Request) -> StreamingResponse:
    """Generic SSE bridge for any adapter that exposes an ``astream`` method.

    Same event shape as the opencode-specific endpoint (session/status/
    text.delta/tool.update/done + a trailing shaped event). Adapters that
    don't implement astream return 501. The opencode handler above is kept
    separate because it threads session-id reuse for chat continuity, which
    is opencode-specific — other adapters just route through here.
    """
    # ``opencode`` has its own endpoint above with session-reuse. Route the
    # generic case for everything else.
    if name == "opencode":
        return await opencode_stream(inp, request)

    adapter = _resolve_adapter(name)
    if not hasattr(adapter, "astream"):
        raise HTTPException(
            status_code=501,
            detail=f"adapter {name!r} does not implement streaming",
        )
    settings = get_settings()
    store = open_default_store(settings)
    ctx = RunContext(settings=settings, mode="ask")

    # Pre-fetch grounding the same way /ask does (skipped for the pipeline
    # adapter which does its own internal grounding).
    grounding: GroundedContext | None = None
    if inp.grounded and name != "pipeline":
        grounding = await retrieve_grounded_context(
            query=inp.query, settings=settings, repos=inp.repos, top_k=inp.top_k,
        )

    thread_key = inp.thread_id or ctx.run_id
    threaded_query = _augmented_ask_query_for_thread(
        store, thread_key, ctx.run_id, inp.query,
    )
    final_query = (
        build_grounded_prompt(grounding_block=grounding.grounding_block, query=threaded_query)
        if grounding and grounding.grounding_block
        else threaded_query
    )
    adapter_inp = inp.model_copy(update={"query": final_query})
    ref_for_store = _storage_ask_ref(inp, thread_key)
    q_preview = inp.query[:160].strip()
    store.start(
        run_id=ctx.run_id, mode="ask",
        input_ref=ref_for_store,
        input_preview=q_preview,
        output_format="markdown",
        thread_id=thread_key,
    )
    _adapter_progress_started(store, ctx.run_id, decompose=False)
    import time as _time
    start = _time.monotonic()

    async def event_source():
        yield f"data: {_sse_json({'kind': 'run', 'run_id': ctx.run_id, 'thread_id': thread_key})}\n\n"
        final_event: dict[str, Any] | None = None
        try:
            async for ev in adapter.astream(adapter_inp):
                if await request.is_disconnected():
                    break
                yield f"data: {_sse_json(ev)}\n\n"
                if ev.get("kind") == "done":
                    final_event = ev
                    break
        except Exception as e:  # noqa: BLE001
            err = {"kind": "error", "error": f"{type(e).__name__}: {e}"}
            yield f"data: {_sse_json(err)}\n\n"
            _persist_failed(store, ctx.run_id, ref_for_store, q_preview, thread_key, err["error"])
            return
        finally:
            try:
                store.close()
            except Exception:  # noqa: BLE001
                pass

        if final_event and (final_event.get("result") or {}).get("ok"):
            result = final_event["result"]
            duration_ms = int((_time.monotonic() - start) * 1000)
            shaped = await _maybe_shape_answer(
                settings=settings,
                query=inp.query,
                raw_text=result.get("answer") or "",
                override=inp.shape,
            )
            if shaped is not None:
                yield f"data: {_sse_json({'kind': 'shaped', 'shaped': shaped.model_dump(mode='json')})}\n\n"
            # Persist via a generalized variant of _persist_streamed_ok.
            _persist_generic_streamed_ok(
                adapter_name=name,
                run_id=ctx.run_id,
                thread_key=thread_key,
                ref_for_store=ref_for_store,
                q_preview=q_preview,
                grounding=grounding,
                result=result,
                duration_ms=duration_ms,
                shaped=shaped,
            )

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _persist_generic_streamed_ok(
    *,
    adapter_name: str,
    run_id: str,
    thread_key: str,
    ref_for_store: dict[str, Any],
    q_preview: str,
    grounding: GroundedContext | None,
    result: dict[str, Any],
    duration_ms: int,
    shaped: Any = None,
) -> None:
    """Adapter-agnostic version of _persist_streamed_ok. Pulls common
    metric keys out of the agent-node result envelope and writes the run
    row with engine=``{name}:ask:stream``."""
    settings = get_settings()
    s2 = open_default_store(settings)
    try:
        extra: dict[str, Any] = {"agent_node": True, "stream": True, "adapter": adapter_name}
        for src, dst in (
            ("toolCalls", "tool_calls"),
            ("toolNames", "tool_names"),
            ("toolTrace", "tool_trace"),
            ("thoughtsTokens", "reasoning_tokens"),
        ):
            if result.get(src) is not None:
                extra[dst] = result[src]
        payload: dict[str, Any] = {"metrics_extra": extra}
        if grounding is not None:
            payload["grounding"] = grounding.model_dump(mode="json")
            from .core.coverage import snippet_paths
            extra["grounding_paths"] = sorted(snippet_paths(grounding.snippets))
        if shaped is not None:
            payload["structured_answer"] = shaped.model_dump(mode="json")
        s2.record(
            run_id=run_id, mode="ask", status="completed",
            input_ref=ref_for_store, engine=f"{adapter_name}:ask:stream",
            answer=result.get("answer") or "",
            citations=[],
            payload=payload,
            model=result.get("model"),
            total_seconds=duration_ms / 1000.0,
            total_cost_usd=result.get("costUsd"),
            input_tokens=result.get("tokensIn"),
            output_tokens=result.get("tokensOut"),
            input_preview=q_preview,
            thread_id=thread_key,
        )
    finally:
        s2.close()


@app.post("/v1/adapters/opencode/abort")
async def opencode_abort(body: OpencodeAbortRequest) -> dict[str, Any]:
    """Stop a running OpenCode session by id. Requires OPENCODE_SDK_BASE_URL."""
    from .adapters._opencode_sdk import OpencodeSDKAdapter

    adapter = _resolve_adapter("opencode")
    if not isinstance(adapter, OpencodeSDKAdapter):
        raise HTTPException(status_code=500, detail="opencode adapter not registered")
    cwd = (
        adapter.settings.repo_path(body.repos[0])
        if body.repos and len(body.repos) == 1
        else Path(adapter.settings.repos_root)
    )
    try:
        out = await adapter.abort(body.session_id, cwd=cwd)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"{type(e).__name__}: {e}") from e
    return out


def _sse_json(obj: Any) -> str:
    import json as _json
    return _json.dumps(obj, ensure_ascii=False)


async def _maybe_shape_answer(
    *,
    settings,
    query: str,
    raw_text: str,
    override: bool | None,
):
    """Run the structured-answer shaper if globally enabled and not opted
    out for this request.

    Returns ``ShapedAnswer | None``. None means we skipped — caller does
    not persist ``structured_answer``. Errors inside the shaper degrade
    to a fallback ShapedAnswer (not None) — the only "skip" cases are
    user override or settings disabled.
    """
    enabled = settings.answer_shaper_enabled if override is None else bool(override)
    if not enabled:
        return None
    if not (raw_text or "").strip():
        return None
    from .core.answer_shaper import shape_answer
    return await shape_answer(query=query, raw_text=raw_text, settings=settings)


def _latest_opencode_session_for_thread(store: RunStore, thread_id: str) -> str | None:
    """Return the last completed opencode run's OpenCode sessionID in this
    thread, so the next streamed turn can reuse it (chat memory + cache hits).

    The id is stored under ``payload_json.metrics_extra.session_id`` by
    ``_persist_streamed_ok``. We walk newest-first to grab the most recent.
    """
    try:
        rows = store.list_by_thread(thread_id, mode="ask", limit=20)
    except Exception:  # noqa: BLE001
        return None
    for row in reversed(rows):
        if (row.get("engine") or "") != "opencode:ask":
            continue
        if row.get("status") != "completed":
            continue
        payload = row.get("payload_json") or row.get("payload") or {}
        if isinstance(payload, str):
            try:
                import json as _json
                payload = _json.loads(payload)
            except Exception:  # noqa: BLE001
                payload = {}
        extra = (payload or {}).get("metrics_extra") or {}
        sid = extra.get("session_id")
        if isinstance(sid, str) and sid.strip():
            return sid.strip()
    return None


def _persist_failed(
    store: RunStore,
    run_id: str,
    ref_for_store: dict[str, Any],
    q_preview: str,
    thread_key: str,
    error: str,
) -> None:
    try:
        store.record(
            run_id=run_id, mode="ask", status="failed",
            input_ref=ref_for_store, engine="opencode:ask",
            error=error, input_preview=q_preview,
            thread_id=thread_key,
        )
    except Exception:  # noqa: BLE001
        pass


def _persist_streamed_ok(
    *,
    run_id: str,
    thread_key: str,
    ref_for_store: dict[str, Any],
    q_preview: str,
    grounding: GroundedContext | None,
    result: dict[str, Any],
    duration_ms: int,
    shaped: Any = None,
) -> None:
    """Write the completed streamed run into the runstore so the history
    sidebar, telemetry panel, and bake-off all see it — same shape the
    blocking /ask handler uses. ``shaped`` is the optional ShapedAnswer
    from the post-processing pass; persisted under ``structured_answer``."""
    settings = get_settings()
    s2 = open_default_store(settings)
    try:
        extra: dict[str, Any] = {"agent_node": True, "opencode_mode": True}
        for src, dst in (
            ("toolNames", "opencode_tool_names"),
            ("toolTrace", "tool_trace"),
            ("toolWallMs", "tool_wall_ms"),
            ("cacheReadTokens", "cache_read_tokens"),
            ("cacheWriteTokens", "cache_write_tokens"),
            ("tokensReasoning", "reasoning_tokens"),
            ("finishReason", "finish_reason"),
            ("providerID", "provider_id"),
            ("agentMode", "agent_mode"),
            ("sessionID", "session_id"),
            ("groundingPaths", "grounding_paths"),
        ):
            if result.get(src) is not None:
                extra[dst] = result[src]
        payload: dict[str, Any] = {"metrics_extra": extra}
        if grounding is not None:
            payload["grounding"] = grounding.model_dump(mode="json")
        if shaped is not None:
            payload["structured_answer"] = shaped.model_dump(mode="json")
        s2.record(
            run_id=run_id, mode="ask", status="completed",
            input_ref=ref_for_store, engine="opencode:ask",
            answer=result.get("answer") or "",
            citations=[],
            payload=payload,
            model=result.get("model"),
            total_seconds=duration_ms / 1000.0,
            total_cost_usd=result.get("costUsd"),
            input_tokens=result.get("tokensIn"),
            output_tokens=result.get("tokensOut"),
            input_preview=q_preview,
            thread_id=thread_key,
        )
    finally:
        s2.close()


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


# ---------------------------------------------------------------------------
# /v1/bakeoff — fan-out across N adapters
# ---------------------------------------------------------------------------


class BakeoffBody(BaseModel):
    adapters: list[str] = Field(min_length=1)
    ask: AdapterAskInput | None = None
    decompose: AdapterDecomposeInput | None = None


@app.post("/v1/bakeoff/{job}")
async def bakeoff(job: Literal["ask", "decompose"], body: BakeoffBody) -> dict[str, Any]:
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
    for name in body.adapters:
        try:
            adapter = _resolve_adapter(name)
        except HTTPException as e:
            results.append({"adapter": name, "ok": False, "error": e.detail, "status": e.status_code})
            continue
        try:
            if job == "ask":
                r = await adapter.ask(inp)
            else:
                r = await adapter.decompose(inp)
            results.append({"adapter": name, "ok": True, "result": r.model_dump()})
        except NotSupported as e:
            results.append({"adapter": name, "ok": False, "error": str(e), "status": 501})
        except Exception as e:  # noqa: BLE001 — surface, don't break the loop
            results.append({"adapter": name, "ok": False, "error": f"{type(e).__name__}: {e}", "status": 502})
    return {"job": job, "results": results}
