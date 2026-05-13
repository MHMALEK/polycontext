from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .ask import AskResult, SourcebotAskError, ask_sourcebot, render_ask_markdown, write_ask_markdown
from .config import get_settings
from .models import DecomposeRequest, DecomposeResponse
from .pipeline import run_pipeline

app = FastAPI(title="ticket-recon", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/decompose", response_model=DecomposeResponse)
async def decompose_endpoint(req: DecomposeRequest) -> DecomposeResponse:
    if not (req.ticket_text or req.ticket_key or req.ticket_url):
        raise HTTPException(
            status_code=400,
            detail="Provide one of: ticket_text, ticket_key, ticket_url.",
        )
    settings = get_settings()
    try:
        return await run_pipeline(req, settings)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}") from e


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    repos: list[str] | None = None
    max_steps: int | None = Field(default=None, ge=1, le=50)
    write_markdown: bool = False
    # When true, only ``POST {SOURCEBOT_URL}/api/ask`` (SSE); no MCP fallback.
    sse_only: bool = False


class AskResponse(BaseModel):
    answer: str
    citations: list = Field(default_factory=list)
    metadata: dict | None = None
    wall_seconds: float | None = None
    markdown: str
    markdown_path: str | None = None


@app.post("/ask", response_model=AskResponse)
async def ask_endpoint(req: AskRequest) -> AskResponse:
    settings = get_settings()
    try:
        result = await ask_sourcebot(
            req.question,
            settings=settings,
            repos=req.repos,
            max_steps=req.max_steps,
            disable_mcp_fallback=True if req.sse_only else None,
        )
    except SourcebotAskError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}") from e
    md = render_ask_markdown(req.question, result)
    md_path: str | None = None
    if req.write_markdown:
        md_path = str(write_ask_markdown(req.question, result, settings))
    return AskResponse(
        answer=result.answer,
        citations=[c.model_dump(mode="json") for c in result.citations],
        metadata=result.metadata.model_dump(mode="json") if result.metadata else None,
        wall_seconds=result.wall_seconds,
        markdown=md,
        markdown_path=md_path,
    )
