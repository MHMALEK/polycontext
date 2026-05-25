"""Pydantic-AI answer shaper — turn raw adapter markdown into a rich card.

Why this exists
---------------
Adapters return ``answer: str`` markdown — sometimes great, sometimes a
wall of text, sometimes "I think it's in ``app.py``" with no path. The UI
wants a uniform structured shape so it can render:

- a punchy 1-sentence headline at the top
- the full markdown details underneath (collapsible if long)
- inline file/line citations the user can click
- a confidence pill (low/medium/high) so the user can calibrate trust
- caveats / open questions called out separately so they don't get lost
- optional next-step suggestions

Implementation
--------------
One extra LLM call per turn, by default Gemini Flash (~$0.0001/call,
1-3s wall). The shaper agent has ``output_type=Answer`` so pydantic-AI
coerces the model output into the schema with retries on validation
failure. On any error (no API key, schema rejection after retries,
timeout) we degrade gracefully to ``Answer(details=raw, confidence="low")``
— the user still sees the raw answer, just without the structured chrome.

The shaper does NOT regenerate the answer. The system prompt is explicit:
*re-shape only*, don't add information, don't change conclusions, don't
hallucinate citations. The model's job is purely structural.
"""
from __future__ import annotations

import os
import time
from typing import Any, Literal

from pydantic import BaseModel, Field

from ..config import Settings


class Citation(BaseModel):
    """A file reference cited in the answer.

    Lines are 1-indexed and may both be None for whole-file references.
    The renderer turns this into a clickable chip like ``app.py:120-145``.
    """

    path: str = Field(description="Repo-relative file path, e.g. 'src/foo.py' or 'web/src/App.tsx'.")
    start_line: int | None = Field(
        default=None,
        description="First line of the cited range (1-indexed). Omit for whole-file.",
    )
    end_line: int | None = Field(
        default=None,
        description="Last line of the cited range (1-indexed). Omit for whole-file or single-line.",
    )
    note: str | None = Field(
        default=None,
        description="One-sentence label for why this file matters in the answer.",
    )


Confidence = Literal["low", "medium", "high"]


class Answer(BaseModel):
    """Structured shape for the rich answer card.

    Field ordering mirrors the UI's visual hierarchy: summary is shown
    first, then confidence, then details, then citations + caveats + next
    steps below. Keep field names stable — the web UI binds to them.
    """

    summary: str = Field(
        description=(
            "One- or two-sentence headline that DIRECTLY answers the question. "
            "Plain prose, no markdown, no preambles like 'Sure, here's...'."
        ),
    )
    details: str = Field(
        description=(
            "Full markdown body of the answer (lists, code fences allowed). "
            "Reproduces the upstream content faithfully — do not invent. "
            "If upstream was concise, this can equal the summary."
        ),
    )
    citations: list[Citation] = Field(
        default_factory=list,
        description=(
            "File:line references the upstream answer relied on. Include only "
            "paths that actually appear in the upstream text — do not invent."
        ),
    )
    confidence: Confidence = Field(
        description=(
            "Grounded in the upstream answer's specificity. 'high' = concrete "
            "file paths with line ranges; 'medium' = files but vague locations; "
            "'low' = upstream punted, asked for clarification, or said it didn't "
            "find anything."
        ),
    )
    caveats: list[str] = Field(
        default_factory=list,
        description=(
            "Things the upstream answer flagged as uncertain, untested, or "
            "incomplete. Empty list when upstream was confident."
        ),
    )
    next_steps: list[str] = Field(
        default_factory=list,
        description=(
            "Optional concrete follow-up actions the engineer could take, "
            "extracted from the upstream answer. Empty list if not applicable."
        ),
    )


class ShapedAnswer(BaseModel):
    """Container the API layer persists alongside the raw answer.

    Carries the structured ``Answer`` plus telemetry (model used, latency,
    token counts, fallback reason) so the UI can show shaper status in the
    Telemetry panel and we can debug regressions without rerunning.
    """

    answer: Answer
    shaper_model: str | None = None
    duration_ms: int = 0
    tokens_in: int | None = None
    tokens_out: int | None = None
    used_fallback: bool = False
    fallback_reason: str | None = None


_SHAPER_SYSTEM = """\
You are a structural re-shaper, not an author. You receive an engineer's
question and an UPSTREAM ANSWER produced by a code Q&A agent. Your job is
to produce a structured Answer object matching the requested schema —
nothing more.

HARD RULES:
1. Re-shape the upstream answer faithfully. Do NOT invent file paths,
   line numbers, or facts not present in the upstream. If the upstream
   said "I think it's in app.py" without confirming, your `confidence`
   is 'low' and the citation should be omitted or noted as unconfirmed.
2. `summary` is one or two sentences that DIRECTLY answer the question.
   No preambles ("Sure!", "Based on the code..."). No markdown.
3. `details` reproduces the substantive upstream content as markdown.
   Code fences, lists, headings allowed. Trim chit-chat and preambles.
4. `citations` ONLY contains paths that explicitly appear in the upstream
   text. If the upstream cited `src/foo.py:120-145`, emit
   {path: 'src/foo.py', start_line: 120, end_line: 145}. If only a path
   is mentioned with no lines, omit start/end_line.
5. `confidence` calibrates to the upstream's specificity:
     - 'high' = concrete files + line ranges + actual code excerpts
     - 'medium' = file names but vague locations
     - 'low' = upstream punted, asked for clarification, or said "not found"
6. `caveats` collects anything the upstream flagged as uncertain or
   untested. Empty list if upstream was confident.
7. `next_steps` extracts concrete follow-ups the upstream proposed.
   Empty list if upstream didn't suggest any.

If the upstream is empty or completely unrelated to the question,
return summary="No usable answer was produced.", details=upstream,
confidence='low'.
"""


async def shape_answer(
    *,
    query: str,
    raw_text: str,
    settings: Settings,
    model_id: str | None = None,
) -> ShapedAnswer:
    """Run the shaper. Always returns a ShapedAnswer — never raises.

    On any failure (missing key, schema rejection, timeout), the fallback
    path returns an Answer with details=raw_text, confidence='low', and
    used_fallback=True so the UI can degrade gracefully. The caller does
    not need to wrap this in try/except.
    """
    t0 = time.monotonic()
    raw = (raw_text or "").strip()
    if not raw:
        return _fallback(
            raw_text="",
            reason="empty upstream",
            t0=t0,
            confidence="low",
        )

    if not settings.gemini_api_key:
        return _fallback(
            raw_text=raw,
            reason="GEMINI_API_KEY not set",
            t0=t0,
            confidence=_heuristic_confidence(raw),
        )

    try:
        from pydantic_ai import Agent
        from pydantic_ai.models.gemini import GeminiModel
        from pydantic_ai.settings import ModelSettings
    except ImportError:
        return _fallback(
            raw_text=raw,
            reason="pydantic-ai not installed",
            t0=t0,
            confidence=_heuristic_confidence(raw),
        )

    os.environ.setdefault("GEMINI_API_KEY", settings.gemini_api_key)
    chosen_model = (model_id or settings.answer_shaper_model or "gemini-2.5-flash").strip()
    # Strip provider prefix if present (pydantic-ai's GeminiModel wants bare name).
    if ":" in chosen_model:
        chosen_model = chosen_model.split(":", 1)[1].strip()
    max_chars = int(settings.answer_shaper_max_input_chars or 16000)

    try:
        agent = Agent(
            model=GeminiModel(chosen_model),
            output_type=Answer,
            system_prompt=_SHAPER_SYSTEM,
            model_settings=ModelSettings(temperature=0.1),
        )
        prompt = (
            f"USER QUESTION:\n{query.strip()[:2000]}\n\n"
            f"UPSTREAM ANSWER:\n{raw[:max_chars]}\n"
        )
        result = await agent.run(prompt)
        answer: Answer = result.output
        toks_in, toks_out = _usage_from(result)
        return ShapedAnswer(
            answer=answer,
            shaper_model=chosen_model,
            duration_ms=int((time.monotonic() - t0) * 1000),
            tokens_in=toks_in,
            tokens_out=toks_out,
            used_fallback=False,
        )
    except Exception as e:  # noqa: BLE001 — degrade on ANY error so UI is robust
        return _fallback(
            raw_text=raw,
            reason=f"{type(e).__name__}: {e}",
            t0=t0,
            confidence=_heuristic_confidence(raw),
            shaper_model=chosen_model,
        )


# ---------------------------------------------------------------------------
# Fallback / heuristics
# ---------------------------------------------------------------------------


def _fallback(
    *,
    raw_text: str,
    reason: str,
    t0: float,
    confidence: Confidence = "low",
    shaper_model: str | None = None,
) -> ShapedAnswer:
    """Build a best-effort ShapedAnswer without the LLM.

    Summary is the first sentence/line of the raw text (max 240 chars).
    Citations are pulled from any ``path:line-range`` patterns we can spot
    in the raw text — cheap regex, not perfect. Better than no card.
    """
    summary = _heuristic_summary(raw_text)
    citations = _heuristic_citations(raw_text)
    return ShapedAnswer(
        answer=Answer(
            summary=summary or "No structured summary available.",
            details=raw_text,
            citations=citations,
            confidence=confidence,
            caveats=[],
            next_steps=[],
        ),
        shaper_model=shaper_model,
        duration_ms=int((time.monotonic() - t0) * 1000),
        used_fallback=True,
        fallback_reason=reason,
    )


def _heuristic_summary(raw: str) -> str:
    if not raw:
        return ""
    # Skip leading markdown headings — they're often labels, not the answer.
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    for line in lines:
        if line.startswith("#"):
            continue
        # Strip leading list bullets so the headline reads cleanly.
        clean = line.lstrip("-*0123456789. ").strip()
        if len(clean) < 8:
            continue
        return clean[:240]
    return raw[:240]


def _heuristic_confidence(raw: str) -> Confidence:
    """Crude confidence guess used when the LLM shaper is unavailable.

    'high' if we see clear file:line patterns; 'low' if the text looks
    like a punt ("I'm not sure", "could you clarify"); else 'medium'.
    """
    lower = raw.lower()
    punt_markers = (
        "could you clarify",
        "i'm not sure",
        "i am not sure",
        "i couldn't find",
        "did not find",
        "not enough context",
        "please specify",
    )
    if any(m in lower for m in punt_markers):
        return "low"
    import re as _re
    # Concrete code references like `path/foo.py:120-145` or `path/foo.py:120`.
    if _re.search(r"\b[\w./-]+\.(py|ts|tsx|js|jsx|go|rs|java|kt|swift|sql)(:\d+(-\d+)?)?\b", raw):
        return "high"
    return "medium"


def _heuristic_citations(raw: str) -> list[Citation]:
    """Pull `file.ext:start-end` mentions out of the raw text. Cheap regex.

    The shaper LLM does this far better — this is the fallback when the
    LLM is unavailable. Deduplicated by (path, start_line, end_line) so
    the same file referenced 5 times only shows up once.
    """
    import re as _re

    # Longest extensions first so `tsx`/`jsx`/`yaml` win over `ts`/`js`/`yml`.
    # Without this, regex alternation matches `ts` greedily in `App.tsx:42-78`
    # and drops the line numbers (because they fall outside the capture).
    pattern = _re.compile(
        r"`?([\w./-]+\.(?:tsx|jsx|yaml|swift|java|html|toml|json|yml|css|sql|sh|md|py|ts|js|go|rs|kt))"
        r"(?::(\d+)(?:[-–](\d+))?)?`?"
    )
    seen: set[tuple[str, int | None, int | None]] = set()
    out: list[Citation] = []
    for m in pattern.finditer(raw):
        path = m.group(1)
        # Filter out node_modules / dist / common false positives.
        if "/node_modules/" in path or path.startswith("node_modules/"):
            continue
        if path.endswith(".d.ts"):
            continue
        start = int(m.group(2)) if m.group(2) else None
        end = int(m.group(3)) if m.group(3) else None
        key = (path, start, end)
        if key in seen:
            continue
        seen.add(key)
        out.append(Citation(path=path, start_line=start, end_line=end))
        if len(out) >= 20:  # cap so a chatty answer doesn't blow up the card
            break
    return out


def _usage_from(result: Any) -> tuple[int | None, int | None]:
    try:
        u = result.usage()
        return (
            getattr(u, "request_tokens", None) or getattr(u, "input_tokens", None),
            getattr(u, "response_tokens", None) or getattr(u, "output_tokens", None),
        )
    except Exception:  # noqa: BLE001
        return (None, None)
