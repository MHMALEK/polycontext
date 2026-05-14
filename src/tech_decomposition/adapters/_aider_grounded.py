"""Hybrid adapter: retrieve relevant files first, then drive Aider on them.

Background: plain Aider points its repo-map at one directory and chats
about that codebase. When the directory is ``REPOS_ROOT`` (which holds 4
sibling repos in this project), Aider's repo map gets confused and Aider
falls back to asking "please paste the relevant code" — useless for
grounded Q&A.

This adapter sidesteps the problem. Per https://aider.chat/docs/faq.html,
Aider does what you tell it to with the files you put in chat, regardless
of where they live. So:

  1. Use the project's existing retrievers (Sourcebot + ripgrep) to find
     files relevant to the question, across **all** configured repos.
  2. Hand those file paths to ``Coder.create(fnames=...)``.
  3. Let Aider answer (ask) / decompose / edit with that grounded context.

For ``implement`` we don't add retrieval on top — a subtask already names
``repo`` and ``files``, so the standard Aider implement path is the right
one. The adapter just delegates to ``AiderAdapter.implement`` for that.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from pathlib import Path

from ..models import (
    Decomposition,
    EnrichedQuery,
    RetrievedContext,
)
from ..retrievers.ripgrep import RipgrepRetriever
from ..retrievers.sourcebot import SourcebotRetriever
from ._aider import AiderAdapter, _import_aider, _last_message
from ._subprocess import extract_json
from .base import (
    AdapterAskInput,
    AdapterAskResult,
    AdapterDecomposeInput,
    AdapterDecomposeResult,
    AdapterImplementInput,
    AdapterImplementResult,
    AdapterMetrics,
    Capability,
    ImplementContext,
)

log = logging.getLogger(__name__)


# Cap on how many files we hand to Aider per call. Aider has its own token
# budget; pushing 50 files in usually blows past it and degrades quality.
DEFAULT_MAX_FILES = 12


# Naive code-keyword extraction for when we don't want to spend a Gemini
# Flash call on enrichment. Picks up CamelCase, snake_case, dotted paths,
# and bare identifiers — same heuristic the deep-decompose engine uses
# for its grep tool inputs.
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}(?:\.[A-Za-z_][A-Za-z0-9_]+)*")
_STOPWORDS = {
    "the", "this", "that", "with", "from", "into", "have", "has", "are",
    "and", "for", "what", "where", "when", "which", "how", "why", "who",
    "page", "code", "file", "files", "function", "method", "class",
    "validation", "validator", "validate",  # too generic for our codebase
}


class AiderGroundedAdapter(AiderAdapter):
    """Same as ``AiderAdapter`` but with a retrieval prelude on ask/decompose.

    Inherits ``implement`` from ``AiderAdapter`` unchanged — implement
    already has explicit files from the subtask, so the retrieval step
    would just be noise.
    """

    name = "aider_grounded"
    capabilities: set[Capability] = {"ask", "decompose", "implement"}
    description = (
        "Hybrid: retrieve files via Sourcebot + ripgrep across all repos, "
        "then run Aider with those files in chat. Gives Aider multi-repo "
        "grounding it can't do on its own."
    )

    # ---- retrieval ---------------------------------------------------------

    def _build_pseudo_query(self, text: str) -> EnrichedQuery:
        """Construct an ``EnrichedQuery`` from raw text without an LLM call.

        Retrievers expect ``code_keywords`` + ``search_queries``. We hand-roll
        both from the question's salient tokens, deduped, capped. Not as good
        as running the real enricher, but cheap and good enough for the
        bake-off — and the user can always run the real enricher externally
        before calling us.
        """
        candidates = []
        for m in _TOKEN_RE.finditer(text):
            tok = m.group(0)
            low = tok.lower()
            if low in _STOPWORDS:
                continue
            candidates.append(tok)
        # Dedupe preserving order, cap at 12.
        seen = set()
        keywords: list[str] = []
        for c in candidates:
            if c.lower() in seen:
                continue
            seen.add(c.lower())
            keywords.append(c)
            if len(keywords) >= 12:
                break
        return EnrichedQuery(
            summary=text[:200],
            intent="investigation",
            entities=[],
            code_keywords=keywords,
            suspected_repos=list(self.settings.repos),
            search_queries=keywords[:6],
            open_questions=[],
            confidence="medium",
        )

    async def _retrieve_paths(self, text: str, *, max_files: int = DEFAULT_MAX_FILES) -> list[Path]:
        """Fan out across repos × retrievers, dedupe by absolute path.

        Returns a list of absolute file paths under ``settings.repos_root``,
        sorted by retrieval score (highest first). Best-effort — a failing
        retriever logs a warning and doesn't break the call.
        """
        query = self._build_pseudo_query(text)
        sourcebot = SourcebotRetriever(self.settings)
        ripgrep = RipgrepRetriever(self.settings)

        async def _try(retriever, repo: str):
            try:
                return await retriever.retrieve(repo, query)
            except Exception as e:  # noqa: BLE001 — best-effort retrieval
                log.warning("retriever %s failed on %s: %s", retriever.name, repo, e)
                return None

        coros = []
        for repo in self.settings.repos:
            coros.append(_try(sourcebot, repo))
            coros.append(_try(ripgrep, repo))
        results = await asyncio.gather(*coros)

        # Collect snippets with their per-snippet score, keep best score per path.
        path_score: dict[Path, float] = {}
        for ctx in results:
            if not ctx or not ctx.snippets:
                continue
            for s in ctx.snippets:
                abs_path = (self.settings.repo_path(s.repo) / s.path).resolve()
                if not abs_path.exists():
                    continue
                if abs_path not in path_score or s.score > path_score[abs_path]:
                    path_score[abs_path] = s.score

        ordered = sorted(path_score.items(), key=lambda kv: -kv[1])
        return [p for p, _ in ordered[:max_files]]

    # ---- ask --------------------------------------------------------------

    async def ask(self, inp: AdapterAskInput) -> AdapterAskResult:
        Coder, InputOutput, Model = _import_aider()
        model_name = self._model_name()
        paths = await self._retrieve_paths(inp.query)

        def _sync_ask(fnames: list[str]) -> dict:
            t0 = time.monotonic()
            io = InputOutput(yes=True, pretty=False)
            model = Model(model_name)
            import os
            old = os.getcwd()
            os.chdir(str(self.settings.repos_root))
            try:
                coder = Coder.create(
                    main_model=model,
                    io=io,
                    fnames=fnames or None,
                    edit_format="ask",
                    auto_commits=False,
                    dirty_commits=False,
                )
                coder.run(inp.query)
            finally:
                os.chdir(old)
            elapsed = time.monotonic() - t0
            answer = (
                getattr(coder, "last_assistant_message", None)
                or _last_message(coder)
                or ""
            )
            return {
                "answer": answer,
                "duration_ms": int(elapsed * 1000),
                "tokens_in": getattr(coder, "message_tokens_sent", None),
                "tokens_out": getattr(coder, "message_tokens_received", None),
                "cost_usd": getattr(coder, "total_cost", None),
            }

        try:
            res = await asyncio.to_thread(_sync_ask, [str(p) for p in paths])
        except Exception as e:
            raise RuntimeError(
                f"aider_grounded ask failed: {type(e).__name__}: {e}"
            ) from e

        return AdapterAskResult(
            adapter=self.name,
            answer=res["answer"],
            citations=[],
            metrics=AdapterMetrics(
                duration_ms=res["duration_ms"],
                tokens_in=res["tokens_in"],
                tokens_out=res["tokens_out"],
                cost_usd=res["cost_usd"],
                model=model_name,
                extra={"retrieved_files": [str(p.relative_to(self.settings.repos_root)) for p in paths]},
            ),
        )

    # ---- decompose --------------------------------------------------------

    async def decompose(self, inp: AdapterDecomposeInput) -> AdapterDecomposeResult:
        """Same retrieve-then-ask flow, asking for the JSON Decomposition shape."""
        Coder, InputOutput, Model = _import_aider()
        model_name = self._model_name()
        ticket_blob = _ticket_blob(inp)
        prompt = _DECOMPOSE_PROMPT.format(ticket=ticket_blob)
        paths = await self._retrieve_paths(ticket_blob, max_files=DEFAULT_MAX_FILES)

        def _sync_decompose(fnames: list[str]) -> dict:
            t0 = time.monotonic()
            io = InputOutput(yes=True, pretty=False)
            model = Model(model_name)
            import os
            old = os.getcwd()
            os.chdir(str(self.settings.repos_root))
            try:
                coder = Coder.create(
                    main_model=model,
                    io=io,
                    fnames=fnames or None,
                    edit_format="ask",
                    auto_commits=False,
                    dirty_commits=False,
                )
                coder.run(prompt)
            finally:
                os.chdir(old)
            elapsed = time.monotonic() - t0
            answer = (
                getattr(coder, "last_assistant_message", None)
                or _last_message(coder)
                or ""
            )
            return {
                "answer": answer,
                "duration_ms": int(elapsed * 1000),
                "tokens_in": getattr(coder, "message_tokens_sent", None),
                "tokens_out": getattr(coder, "message_tokens_received", None),
                "cost_usd": getattr(coder, "total_cost", None),
            }

        try:
            res = await asyncio.to_thread(_sync_decompose, [str(p) for p in paths])
        except Exception as e:
            raise RuntimeError(
                f"aider_grounded decompose failed: {type(e).__name__}: {e}"
            ) from e

        try:
            decomp = Decomposition.model_validate(extract_json(res["answer"]))
        except Exception as e:
            raise RuntimeError(
                f"aider_grounded decompose returned unparseable JSON: {e}\n"
                f"--- raw ---\n{res['answer'][:2000]}"
            ) from e

        return AdapterDecomposeResult(
            adapter=self.name,
            decomposition=decomp,
            markdown=res["answer"],
            metrics=AdapterMetrics(
                duration_ms=res["duration_ms"],
                tokens_in=res["tokens_in"],
                tokens_out=res["tokens_out"],
                cost_usd=res["cost_usd"],
                model=model_name,
                extra={"retrieved_files": [str(p.relative_to(self.settings.repos_root)) for p in paths]},
            ),
        )

    # ---- implement --------------------------------------------------------

    async def implement(
        self, inp: AdapterImplementInput, ctx: ImplementContext,
    ) -> AdapterImplementResult:
        """No retrieval prelude — subtasks already name their files.

        We could in theory retrieve *more* files than the subtask names
        (call-sites, tests, helpers) but that gets noisy fast and Aider
        will pull them in itself via /add if it needs them.
        """
        return await super().implement(inp, ctx)


# ---------------------------------------------------------------------------
# Helpers (copied from _aider.py rather than imported to avoid a circular
# dance — they're small).
# ---------------------------------------------------------------------------


def _ticket_blob(inp: AdapterDecomposeInput) -> str:
    parts = []
    if inp.ticket_key:
        parts.append(f"Ticket key: {inp.ticket_key}")
    if inp.ticket_url:
        parts.append(f"Ticket URL: {inp.ticket_url}")
    if inp.ticket_text:
        parts.append(inp.ticket_text)
    if inp.repos:
        parts.append(f"Repos to consider: {', '.join(inp.repos)}")
    return "\n\n".join(parts)


_DECOMPOSE_PROMPT = """Decompose this Jira ticket into a tech decomposition. Use the file
contents you can see (the files I added to chat) to ground every reference.
Never invent paths — only cite files that are visible to you in this chat.

Output exactly one JSON object matching this schema and NOTHING ELSE
(no prose, no markdown fences, no preamble):

{{
  "ticket_key": str | null,
  "ticket_title": str,
  "ticket_url": str | null,
  "overview": str,
  "affected_repos": [str],
  "risks": [str],
  "open_questions": [str],
  "subtasks": [
    {{
      "title": str,
      "description": str,
      "repo": str,
      "files": [str],
      "file_links": [str],
      "acceptance_criteria": [str],
      "estimated_complexity": "small" | "medium" | "large" | "unknown"
    }}
  ],
  "enrichment_model": "",
  "decomposition_model": "aider_grounded"
}}

Ticket:
{ticket}
"""
