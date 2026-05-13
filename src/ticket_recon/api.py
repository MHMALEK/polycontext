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

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import get_settings
from .core.context import RunContext
from .core.factory import build_ask_pipeline, build_decompose_pipeline, build_metrics_observer

app = FastAPI(title="ticket-recon", version="0.2.0")


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
    write_markdown: bool = False
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
            include_cli_sink=False,
            include_file_sink=req.write_markdown,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"{type(e).__name__}: {e}") from e

    obs = build_metrics_observer(settings, mode="ask")
    ctx = RunContext(settings=settings, mode="ask", metrics=obs)
    try:
        run = await pipeline.run(req.question, ctx)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"{type(e).__name__}: {e}") from e

    er = run.engine_result
    md_path = next(
        (sr.location for sr in run.sink_results if sr.sink == "markdown_file" and sr.location),
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
        include_cli_sink=False,
        include_file_sink=True,
    )
    obs = build_metrics_observer(settings, mode="decompose")
    ctx = RunContext(settings=settings, mode="decompose", metrics=obs)
    try:
        run = await pipeline.run(ref, ctx)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}") from e

    er = run.engine_result
    md_path = next(
        (sr.location for sr in run.sink_results if sr.sink == "markdown_file" and sr.location),
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
