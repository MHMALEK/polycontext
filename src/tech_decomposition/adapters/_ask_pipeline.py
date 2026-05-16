"""Adapter-agnostic ask pipeline stages.

This module isolates code-QA processing into explicit, reusable stages:

1) input preparation
2) context retrieval
3) prompt preparation
4) model/tool invocation (adapter-provided callback)
5) response formatting

Adapters can plug different invoke callbacks (CLI, HTTP SDK, local model)
without changing retrieval/prompt/formatting behavior.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from ..models import Snippet
from ._grounding import DEFAULT_MAX_FILES, format_grounding_block, retrieve_context_snippets
from .base import AdapterAskInput


@dataclass
class AskPreparedInput:
    query: str
    repos: list[str] | None
    top_k: int
    stage_ms: int


@dataclass
class AskRetrievedContext:
    snippets: list[Snippet] = field(default_factory=list)
    grounding_block: str = ""
    stage_ms: int = 0


@dataclass
class AskPrompt:
    text: str
    stage_ms: int


@dataclass
class AskInvocation:
    """Result returned by the adapter-specific invoke callback."""

    answer: str
    payload: Any = None
    stage_ms: int = 0


@dataclass
class AskResponse:
    answer: str
    citations: list[Snippet]
    stage_ms: int
    attempts: int = 1
    contract_issues: list[str] = field(default_factory=list)


@dataclass
class AskPipelineResult:
    prepared_input: AskPreparedInput
    retrieved_context: AskRetrievedContext
    prompt: AskPrompt
    invocation: AskInvocation
    response: AskResponse


InvokeAsk = Callable[[str], Awaitable[AskInvocation]]


@dataclass
class AskPipelinePolicy:
    max_attempts: int = 1
    min_answer_chars: int = 120
    enforce_contract: bool = False
    required_sections: tuple[str, ...] = ()
    require_inline_citations_when_grounded: bool = False
    retry_without_grounding_on_failure: bool = True


class AskPipeline:
    """Shared, adapter-agnostic ask orchestration."""

    def __init__(
        self,
        settings,
        *,
        grounded: bool,
        preamble: str,
        policy: AskPipelinePolicy | None = None,
    ):
        self.settings = settings
        self.grounded = grounded
        self.preamble = preamble
        self.policy = policy or AskPipelinePolicy()

    async def run(self, inp: AdapterAskInput, invoke: InvokeAsk) -> AskPipelineResult:
        prepared = self.prepare_input(inp)
        retrieved = await self.retrieve_context(prepared)
        prompt = self.prepare_prompt(prepared, retrieved)
        invocation, response = await self._invoke_with_retries(
            prepared=prepared,
            retrieved=retrieved,
            prompt=prompt,
            invoke=invoke,
        )
        return AskPipelineResult(
            prepared_input=prepared,
            retrieved_context=retrieved,
            prompt=prompt,
            invocation=invocation,
            response=response,
        )

    def prepare_input(self, inp: AdapterAskInput) -> AskPreparedInput:
        t = time.monotonic()
        # Keep intentional newlines but collapse noisy blank runs and trim edges.
        query = inp.query.strip()
        query = re.sub(r"\n{3,}", "\n\n", query)
        top_k = max(1, min(inp.top_k, 50))
        return AskPreparedInput(
            query=query,
            repos=inp.repos,
            top_k=top_k,
            stage_ms=int((time.monotonic() - t) * 1000),
        )

    async def retrieve_context(self, prepared: AskPreparedInput) -> AskRetrievedContext:
        t = time.monotonic()
        if not self.grounded:
            return AskRetrievedContext(stage_ms=int((time.monotonic() - t) * 1000))
        try:
            snippets = await retrieve_context_snippets(
                prepared.query,
                self.settings,
                repos=prepared.repos,
                max_files=min(prepared.top_k, DEFAULT_MAX_FILES),
            )
            grounding_block = format_grounding_block(snippets, self.settings.repos_root)
        except Exception:
            # In strict mode we fail closed; otherwise stay best-effort.
            if bool(getattr(self.settings, "grounding_require_sourcebot", False)):
                raise
            snippets = []
            grounding_block = ""
        return AskRetrievedContext(
            snippets=snippets,
            grounding_block=grounding_block,
            stage_ms=int((time.monotonic() - t) * 1000),
        )

    def prepare_prompt(
        self,
        prepared: AskPreparedInput,
        retrieved: AskRetrievedContext,
    ) -> AskPrompt:
        t = time.monotonic()
        context_block = retrieved.grounding_block.strip()
        if context_block:
            context_block = context_block + "\n"
        preamble = self.preamble.rstrip()
        if not preamble.endswith("\n"):
            preamble += "\n"
        question_label = "" if preamble.lower().rstrip().endswith("question:") else "Question:\n"
        text = (
            f"{preamble}"
            "Use the retrieved context below as primary evidence.\n"
            "If evidence is missing, explicitly say what is missing.\n\n"
            "Retrieved context:\n"
            f"{context_block}\n"
            f"{question_label}"
            f"{prepared.query}\n"
        )
        return AskPrompt(text=text, stage_ms=int((time.monotonic() - t) * 1000))

    async def invoke_model(self, prompt: AskPrompt, invoke: InvokeAsk) -> AskInvocation:
        t = time.monotonic()
        out = await invoke(prompt.text)
        # Trust adapter-provided stage time if set; otherwise compute here.
        if out.stage_ms <= 0:
            out.stage_ms = int((time.monotonic() - t) * 1000)
        return out

    def format_response(self, invocation: AskInvocation, retrieved: AskRetrievedContext) -> AskResponse:
        t = time.monotonic()
        answer = (invocation.answer or "").strip()
        # Normalize excessive blank lines for stable rendering.
        answer = re.sub(r"\n{3,}", "\n\n", answer)
        return AskResponse(
            answer=answer,
            citations=list(retrieved.snippets),
            stage_ms=int((time.monotonic() - t) * 1000),
        )

    async def _invoke_with_retries(
        self,
        *,
        prepared: AskPreparedInput,
        retrieved: AskRetrievedContext,
        prompt: AskPrompt,
        invoke: InvokeAsk,
    ) -> tuple[AskInvocation, AskResponse]:
        max_attempts = max(1, int(self.policy.max_attempts))
        issues: list[str] = []
        current_prompt = prompt
        last_error: Exception | None = None
        total_invoke_ms = 0

        for attempt in range(1, max_attempts + 1):
            try:
                invocation = await self.invoke_model(current_prompt, invoke)
                total_invoke_ms += invocation.stage_ms
                response = self.format_response(invocation, retrieved)
            except Exception as e:  # noqa: BLE001
                last_error = e
                if attempt >= max_attempts:
                    raise
                current_prompt = self._repair_prompt(
                    prepared=prepared,
                    retrieved=retrieved,
                    issues=[f"model invocation failed: {type(e).__name__}"],
                    strip_grounding=self.policy.retry_without_grounding_on_failure,
                )
                continue

            ok, issues = self._check_contract(
                answer=response.answer,
                grounded=bool(retrieved.snippets),
            )
            if ok or attempt >= max_attempts:
                response.attempts = attempt
                response.contract_issues = issues
                invocation.stage_ms = total_invoke_ms
                return invocation, response

            current_prompt = self._repair_prompt(
                prepared=prepared,
                retrieved=retrieved,
                issues=issues,
                strip_grounding=False,
            )

        # Defensive fallback; loop returns or raises.
        if last_error:
            raise last_error
        invocation = AskInvocation(answer="", stage_ms=total_invoke_ms)
        response = AskResponse(answer="", citations=list(retrieved.snippets), stage_ms=0, attempts=max_attempts, contract_issues=issues)
        return invocation, response

    def _check_contract(self, *, answer: str, grounded: bool) -> tuple[bool, list[str]]:
        if not self.policy.enforce_contract:
            return True, []
        issues: list[str] = []
        if len(answer) < int(self.policy.min_answer_chars):
            issues.append(f"answer too short (<{self.policy.min_answer_chars} chars)")

        low = answer.lower()
        for section in self.policy.required_sections:
            if not _section_satisfied(section, low):
                issues.append(f"missing section: {section}")

        if grounded and self.policy.require_inline_citations_when_grounded:
            if not _has_citation(answer):
                issues.append("missing inline code citations in required format")

        return (len(issues) == 0), issues

    def _repair_prompt(
        self,
        *,
        prepared: AskPreparedInput,
        retrieved: AskRetrievedContext,
        issues: list[str],
        strip_grounding: bool,
    ) -> AskPrompt:
        t = time.monotonic()
        grounding = "" if strip_grounding else retrieved.grounding_block
        repair = (
            "\n\nQUALITY REPAIR INSTRUCTIONS:\n"
            "Your previous answer did not meet required quality checks.\n"
            "Fix all issues below and return a complete revised answer.\n"
            + "\n".join(f"- {i}" for i in issues)
        )
        text = f"{grounding}{self.preamble}{prepared.query}{repair}"
        return AskPrompt(text=text, stage_ms=int((time.monotonic() - t) * 1000))


def _section_satisfied(section: str, text_lower: str) -> bool:
    """Tolerant section contract checks (semantic, not exact-heading based)."""
    s = section.strip().lower()
    if s == "end-to-end flow":
        repo_mentions = len(
            set(re.findall(r"\b(frontend|traceability|data-cloud-functions|data)\b", text_lower))
        )
        sequence_markers = len(
            re.findall(r"\b(starts?|initiates?|receives?|then|upon|after|on success|finally|next)\b", text_lower)
        )
        return bool(
            re.search(r"end[\s-]*to[\s-]*end", text_lower)
            or re.search(r"\bworkflow\b", text_lower)
            or re.search(r"\bflow\b", text_lower)
            or (repo_mentions >= 2 and sequence_markers >= 2)
        )
    if s == "repo-by-repo responsibilities":
        repo_mentions = len(
            set(re.findall(r"\b(frontend|traceability|data-cloud-functions|data)\b", text_lower))
        )
        return bool(
            re.search(r"repo[\s-]*by[\s-]*repo", text_lower)
            or (repo_mentions >= 2 and re.search(r"responsibilit|owns|handled by", text_lower))
        )
    if s == "validation, persistence, and async/background processing":
        has_validation = re.search(r"\bvalidation\b", text_lower) is not None
        has_persistence = re.search(r"\bpersist|database|store|insert|update|save", text_lower) is not None
        has_async = re.search(r"\basync\b|\bbackground\b|\bqueue\b|\bjob\b|\bevent\b|\btrigger\b", text_lower) is not None
        return bool(has_validation and has_persistence and has_async)
    if s == "user-visible statuses/errors":
        has_user_surface = re.search(r"user[\s-]*visible|\bui\b|\bfrontend\b", text_lower) is not None
        has_status_or_error = re.search(r"\bstatus(?:es)?\b|\berror|\btoast\b|\bmessage\b", text_lower) is not None
        return bool(has_user_surface and has_status_or_error)

    # Fallback to a literal-ish phrase match for custom policy strings.
    esc = re.escape(s).replace(r"\ ", r"[\s-]+")
    return re.search(esc, text_lower) is not None


def _has_citation(answer: str) -> bool:
    """Accept common code citation forms with line anchors.

    Examples accepted:
    - `frontend/src/a.ts:L10-L22`
    - frontend/src/a.ts:L10-L22
    - traceability/src/x.py:L44
    """
    return re.search(
        r"`?[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)+:L\d+(?:-L?\d+)?`?",
        answer,
    ) is not None
