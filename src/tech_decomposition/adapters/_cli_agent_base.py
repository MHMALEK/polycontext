"""Shared base class for CLI-backed code-agent adapters.

``cline``, ``cursor-agent``, and ``opencode run`` all have the same shape:

  1. Build a prompt with one of three preambles (ask / decompose / implement).
  2. Spawn the CLI as a subprocess with the prompt as an argument.
  3. Parse the (possibly noisy) stdout into an answer + metrics.

Before this base class each adapter open-coded steps 1 and 2 with copy-paste
preambles. That made it easy to drift — and worse, every new adapter
re-implemented "only ``duration_ms`` is populated" instead of pulling the
token/cost numbers the CLIs already emit.

Subclasses override two small template methods:

  * ``_build_cmd(prompt, kind)`` — the argv list for this CLI.
  * ``_parse_result(stdout, kind)`` — extracts an answer + structured metrics.

Optional ``grounded=True`` injects a Sourcebot+ripgrep retrieval prelude
into ``ask``/``decompose`` prompts via ``_grounding.py``. ``implement``
never grounds — the subtask already names its files.
"""
from __future__ import annotations

import json
import logging
import subprocess
import time
from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

from ..models import Decomposition
from ._grounding import (
    DEFAULT_MAX_FILES,
    format_grounding_block,
    retrieve_context_paths,
)
from ._subprocess import (
    CliFailed,
    CliNotFound,
    CliResult,
    CliTimeout,
    extract_json,
    resolve_binary,
    run_cli,
)
from .base import (
    Adapter,
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


JobKind = Literal["ask", "decompose", "implement"]


# ---------------------------------------------------------------------------
# Preambles — moved here so the three adapters stop drifting from each other.
# ---------------------------------------------------------------------------


ASK_PREAMBLE = """Answer the following question about the codebase under this working
directory. Ground every claim in real code; cite file paths and line ranges.

Question:
"""


DECOMPOSE_PREAMBLE = """Decompose this Jira ticket into a tech decomposition. Use
file-reading tools to ground every reference; never invent paths.

Output ONE JSON object matching this schema and nothing else (no prose, no fences):

{{
  "ticket_key": str | null, "ticket_title": str, "ticket_url": str | null,
  "overview": str, "affected_repos": [str], "risks": [str], "open_questions": [str],
  "subtasks": [
    {{"title": str, "description": str, "repo": str, "files": [str],
      "file_links": [str], "acceptance_criteria": [str],
      "estimated_complexity": "small"|"medium"|"large"|"unknown"}}
  ],
  "enrichment_model": "", "decomposition_model": "{model_tag}"
}}

Ticket:
"""


IMPLEMENT_PREAMBLE = """You are implementing a single subtask inside a clean git
worktree at this working directory. Edit only files inside this directory. Do not
commit, push, or open MRs.

Subtask:
"""


# ---------------------------------------------------------------------------
# Parsed-run shape
# ---------------------------------------------------------------------------


@dataclass
class ParsedAgentRun:
    """What ``_parse_result`` returns. Most fields are optional because
    different CLIs emit different shapes — text-only CLIs leave token/cost
    at None; structured-output CLIs fill them in.
    """

    answer: str
    model: str | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    cost_usd: float | None = None
    tool_calls: int = 0
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Tolerant JSONL event-stream parser
# ---------------------------------------------------------------------------


# Field name candidates the parser scans for. CLI vendors haven't standardised
# on names ("tokens_in" vs "input_tokens" vs "prompt_tokens"); we accept any
# of them. New shapes can be added without breaking older ones.
_TOKEN_IN_KEYS = (
    "tokens_in", "input_tokens", "prompt_tokens", "tokensIn", "inputTokens",
)
_TOKEN_OUT_KEYS = (
    "tokens_out", "output_tokens", "completion_tokens", "tokensOut", "outputTokens",
)
_COST_KEYS = ("cost_usd", "cost", "total_cost", "totalCost")
_MODEL_KEYS = ("model", "model_id", "modelId")
# Keys whose values are the agent's final reply. Intentionally NOT
# including ``"output"`` — that collides with tool-call result fields
# (e.g. opencode's grep emits ``{state:{output:"Found 20 matches..."}}``
# inside tool_use events).
_TEXT_KEYS = ("text", "content", "message", "answer", "say")


def _walk(obj: Any) -> Iterable[Any]:
    """Yield every nested dict/list/leaf in an object (depth-first)."""
    yield obj
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v)


def _find_first(obj: Any, keys: tuple[str, ...]) -> Any:
    """Return the first value found under any of ``keys`` (anywhere in obj)."""
    for node in _walk(obj):
        if isinstance(node, dict):
            for k in keys:
                if k in node and node[k] is not None:
                    return node[k]
    return None


def _coerce_int(v: Any) -> int | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return int(v)
    if isinstance(v, str):
        try:
            return int(v)
        except ValueError:
            return None
    return None


def _coerce_float(v: Any) -> float | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return None
    return None


def _collect_text_chunks(node: Any, out: list[str]) -> None:
    """Walk ``node`` and collect every string value under a key in _TEXT_KEYS.

    Used so we pick up answers nested inside event payloads (e.g. opencode
    emits ``{type:"text", part:{type:"text", text:"..."}}`` — the answer is
    under ``part.text``, not at the top level).
    """
    if isinstance(node, dict):
        for k, v in node.items():
            if k in _TEXT_KEYS and isinstance(v, str) and v.strip():
                out.append(v)
            else:
                _collect_text_chunks(v, out)
    elif isinstance(node, list):
        for v in node:
            _collect_text_chunks(v, out)


def _tokens_block(obj: Any) -> dict | None:
    """Return the first nested ``tokens: {...}`` dict, or None.

    Different CLIs put usage either at the top level (Anthropic-style:
    ``{usage: {input_tokens, output_tokens}}``) or inside a sub-object
    (opencode-style: ``{part: {tokens: {input, output, total}}}``). We
    look for any nested ``{"tokens": {...}}`` first since it's the
    distinctive shape, then fall back to flat keys.
    """
    for node in _walk(obj):
        if isinstance(node, dict):
            t = node.get("tokens")
            if isinstance(t, dict):
                return t
    return None


def parse_jsonl_events(stdout: str) -> ParsedAgentRun | None:
    """Best-effort parser for JSON-event-stream CLI output.

    Both ``cline --json`` and ``opencode run --format json`` emit one
    JSON object per line. Schemas differ across versions, so we walk
    every nested key looking for the fields we care about rather than
    binding to a specific shape.

    Returns ``None`` if no lines parse as JSON — caller should fall
    back to treating stdout as plain text.
    """
    events: list[dict] = []
    text_chunks: list[str] = []
    tool_calls = 0
    tokens_in: int | None = None
    tokens_out: int | None = None
    cost: float | None = None
    model: str | None = None

    for raw in stdout.splitlines():
        line = raw.strip()
        if not line or not (line.startswith("{") or line.startswith("[")):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        events.append(obj)

        # Pull out per-event text chunks (recursive — opencode nests the
        # actual answer under ``part.text``, not at the top level).
        # Skip text collection for tool events so tool inputs/outputs
        # don't leak into the answer.
        etype = str(obj.get("type") or obj.get("event") or obj.get("kind") or "").lower()
        is_tool_event = "tool" in etype or _find_first(obj, ("tool_use", "toolUse")) is not None
        if is_tool_event:
            tool_calls += 1
        else:
            _collect_text_chunks(obj, text_chunks)

        # Per-event usage rollup — keep the largest value seen (final
        # events typically restate the cumulative usage).
        for k in _TOKEN_IN_KEYS:
            v = _coerce_int(_find_first(obj, (k,)))
            if v is not None and (tokens_in is None or v > tokens_in):
                tokens_in = v
        for k in _TOKEN_OUT_KEYS:
            v = _coerce_int(_find_first(obj, (k,)))
            if v is not None and (tokens_out is None or v > tokens_out):
                tokens_out = v
        # opencode shape: ``{tokens: {input, output, total}}``. Bare
        # "input"/"output" are too generic to scan globally, so we only
        # accept them inside a ``tokens`` block.
        tb = _tokens_block(obj)
        if tb is not None:
            v_in = _coerce_int(tb.get("input"))
            if v_in is not None and (tokens_in is None or v_in > tokens_in):
                tokens_in = v_in
            v_out = _coerce_int(tb.get("output"))
            if v_out is not None and (tokens_out is None or v_out > tokens_out):
                tokens_out = v_out
        for k in _COST_KEYS:
            v = _coerce_float(_find_first(obj, (k,)))
            if v is not None and (cost is None or v > cost):
                cost = v
        if model is None:
            m = _find_first(obj, _MODEL_KEYS)
            if isinstance(m, str):
                model = m

    if not events:
        return None

    # Prefer the last "text"-type event's text as the final answer (that's
    # what opencode and cline both emit at the end). Otherwise fall back
    # to the last "final"/"done"/"result" event, then to the joined
    # transcript of every text chunk seen.
    answer = ""
    for obj in reversed(events):
        etype = str(obj.get("type", "")).lower()
        is_final = (
            obj.get("final")
            or obj.get("done")
            or etype in {"text", "result", "final", "answer", "complete", "completion"}
        )
        if is_final:
            chunks: list[str] = []
            _collect_text_chunks(obj, chunks)
            if chunks:
                answer = chunks[-1].strip()
                break
    if not answer:
        answer = "\n".join(text_chunks).strip()

    return ParsedAgentRun(
        answer=answer,
        model=model,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_usd=cost,
        tool_calls=tool_calls,
        extra={"event_count": len(events)},
    )


# ---------------------------------------------------------------------------
# Helpers (formerly duplicated in three adapter files)
# ---------------------------------------------------------------------------


def ticket_blob(inp: AdapterDecomposeInput) -> str:
    parts: list[str] = []
    if inp.ticket_key:
        parts.append(f"Ticket key: {inp.ticket_key}")
    if inp.ticket_url:
        parts.append(f"Ticket URL: {inp.ticket_url}")
    if inp.ticket_text:
        parts.append(inp.ticket_text)
    if inp.repos:
        parts.append(f"Repos to consider: {', '.join(inp.repos)}")
    return "\n\n".join(parts)


def subtask_prompt(inp: AdapterImplementInput) -> str:
    if inp.subtask:
        st = inp.subtask
        out = [f"# {st.title}", "", st.description]
        if st.acceptance_criteria:
            out.append("")
            out.append("Acceptance criteria:")
            out.extend(f"- {ac}" for ac in st.acceptance_criteria)
        return "\n".join(out)
    return inp.free_text or "Implement the requested task."


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class CliAgentAdapter(Adapter):
    """Template-method base. Subclasses set ``name``/``capabilities`` and
    override ``_bin_name``, ``_build_cmd``, ``_parse_result``, and the
    ``_timeout`` property. Everything else is shared.
    """

    capabilities: set[Capability] = {"ask", "decompose", "implement"}

    # Subclass knobs ---------------------------------------------------------

    #: Name of the binary as it appears on PATH.
    _bin_name: str = ""

    #: Tag inserted into the decompose preamble's ``decomposition_model``
    #: field. Lets bake-off scoring tell which adapter produced what.
    _model_tag: str = ""

    #: Whether this instance applies a Sourcebot+ripgrep retrieval prelude
    #: to ``ask``/``decompose`` prompts. Toggled at construction time so a
    #: single class can register as both ``foo`` and ``foo_grounded``.
    _grounded: bool = False

    #: Argv to invoke the binary's version probe during ``health()``.
    #: Empty list disables the probe (binary-on-PATH check only).
    _version_argv: tuple[str, ...] = ("--version",)

    #: How long the version probe can run before we treat the binary as
    #: unhealthy. Independent of the per-call timeout.
    _health_timeout_s: float = 3.0

    def __init__(self, settings, *, grounded: bool = False):
        super().__init__(settings)
        # Per-instance override; class default stays False.
        self._grounded = grounded or self._grounded

    # Required overrides -----------------------------------------------------

    @abstractmethod
    def _bin_override(self) -> str | None:
        """Return the ``settings.<name>_bin`` override (or empty string)."""

    @abstractmethod
    def _timeout(self) -> float:
        """Return the per-call timeout in seconds from settings."""

    @abstractmethod
    def _build_cmd(self, *, prompt: str, kind: JobKind) -> list[str]:
        """Build the argv for this CLI given a prompt.

        Subclasses choose the right flags for the CLI in question (e.g.
        ``cursor-agent -p`` vs ``opencode run`` vs ``cline --json``).
        """

    def _parse_result(self, *, stdout: str, kind: JobKind) -> ParsedAgentRun:
        """Default: take stdout verbatim as the answer, no structured fields.

        Override on subclasses whose CLI emits JSON event streams so we can
        populate token/cost/tool-call counts in metrics.
        """
        return ParsedAgentRun(answer=stdout.strip())

    # Shared implementations -------------------------------------------------

    def _bin(self) -> str:
        return resolve_binary(self._bin_name, override=self._bin_override() or None)

    def health(self) -> dict:
        try:
            bin_path = self._bin()
        except CliNotFound as e:
            return {"ok": False, "reason": str(e)}
        if self._version_argv:
            try:
                proc = subprocess.run(
                    [bin_path, *self._version_argv],
                    capture_output=True,
                    text=True,
                    timeout=self._health_timeout_s,
                )
            except subprocess.TimeoutExpired:
                return {"ok": False, "reason": f"{self._bin_name} --version timed out"}
            except OSError as e:
                return {"ok": False, "reason": f"{self._bin_name} not executable: {e}"}
            if proc.returncode != 0:
                stderr = (proc.stderr or proc.stdout or "").strip()[:200]
                return {
                    "ok": False,
                    "reason": f"{self._bin_name} --version exited {proc.returncode}: {stderr}",
                }
        return {"ok": True}

    async def ask(self, inp: AdapterAskInput) -> AdapterAskResult:
        prompt = await self._assemble_prompt(
            preamble=ASK_PREAMBLE, body=inp.query, kind="ask",
        )
        t = time.monotonic()
        res = await self._invoke(prompt=prompt, cwd=self.settings.repos_root, kind="ask")
        parsed = self._parse_result(stdout=res.stdout, kind="ask")
        return AdapterAskResult(
            adapter=self.name,
            answer=parsed.answer,
            citations=[],
            metrics=self._metrics(parsed, t, res),
        )

    async def decompose(self, inp: AdapterDecomposeInput) -> AdapterDecomposeResult:
        body = ticket_blob(inp)
        preamble = DECOMPOSE_PREAMBLE.format(model_tag=self._model_tag or self.name)
        prompt = await self._assemble_prompt(preamble=preamble, body=body, kind="decompose")
        t = time.monotonic()
        res = await self._invoke(prompt=prompt, cwd=self.settings.repos_root, kind="decompose")
        parsed = self._parse_result(stdout=res.stdout, kind="decompose")
        try:
            decomp = Decomposition.model_validate(extract_json(parsed.answer or res.stdout))
        except Exception as e:
            raise RuntimeError(
                f"{self.name} decompose did not return parseable JSON: {e}\n"
                f"--- raw ---\n{res.stdout[:2000]}"
            ) from e
        return AdapterDecomposeResult(
            adapter=self.name,
            decomposition=decomp,
            markdown=parsed.answer or res.stdout,
            metrics=self._metrics(parsed, t, res),
        )

    async def implement(
        self, inp: AdapterImplementInput, ctx: ImplementContext,
    ) -> AdapterImplementResult:
        # Implement never grounds — subtask already names files; extra
        # retrieval would just dilute the prompt.
        prompt = IMPLEMENT_PREAMBLE + subtask_prompt(inp)
        t = time.monotonic()
        res = await self._invoke(prompt=prompt, cwd=ctx.worktree_path, kind="implement")
        parsed = self._parse_result(stdout=res.stdout, kind="implement")
        return AdapterImplementResult(
            adapter=self.name,
            mr_url=None,
            branch=ctx.branch,
            commits=[],
            diff_summary=(parsed.answer or res.stdout)[-1000:].strip(),
            files_changed=[],
            metrics=self._metrics(parsed, t, res),
        )

    # Internals --------------------------------------------------------------

    async def _assemble_prompt(self, *, preamble: str, body: str, kind: JobKind) -> str:
        """Prepend grounding block (if enabled) and the preamble to the body.

        Stashes the retrieved file list on ``self._last_grounding_paths``
        so ``_metrics`` can surface it as ``extra.retrieved_files``.
        Adapters are instantiated per request (see ``_resolve_adapter``
        in api.py), so per-instance state is per-call.
        """
        self._last_grounding_paths = []
        grounding = ""
        if self._grounded and kind in ("ask", "decompose"):
            try:
                paths = await retrieve_context_paths(
                    body, self.settings, max_files=DEFAULT_MAX_FILES,
                )
                self._last_grounding_paths = paths
                grounding = format_grounding_block(paths, self.settings.repos_root)
            except Exception as e:  # noqa: BLE001 — never block on retrieval
                log.warning("%s: grounding prelude failed: %s", self.name, e)
                grounding = ""
        return grounding + preamble + body

    async def _invoke(self, *, prompt: str, cwd: Path, kind: JobKind) -> CliResult:
        cmd = self._build_cmd(prompt=prompt, kind=kind)
        try:
            return await run_cli(cmd=cmd, cwd=cwd, timeout=self._timeout())
        except CliTimeout as e:
            raise RuntimeError(str(e)) from e
        except CliFailed as e:
            # Surface stdout too — some CLIs (opencode, cline) write the
            # actual failure reason to stdout when --format json is on and
            # stderr stays empty. Truncate each side independently so we
            # always see *something* even when one channel is silent.
            stderr_part = (e.stderr or "").strip()[:500]
            stdout_part = (e.stdout or "").strip()[:500]
            parts = [f"{self.name} failed (exit {e.returncode})"]
            if stderr_part:
                parts.append(f"stderr: {stderr_part}")
            if stdout_part:
                parts.append(f"stdout: {stdout_part}")
            if not stderr_part and not stdout_part:
                parts.append("(both stdout and stderr were empty)")
            raise RuntimeError(" | ".join(parts)) from e

    def _metrics(self, parsed: ParsedAgentRun, t_start: float, res: CliResult) -> AdapterMetrics:
        extra = dict(parsed.extra)
        if res.stderr:
            extra.setdefault("stderr_excerpt", res.stderr[:200])
        if self._grounded:
            extra.setdefault("grounded", True)
            # Surface the file pointers the prelude injected so callers
            # can audit retrieval quality without re-running.
            paths = getattr(self, "_last_grounding_paths", None) or []
            extra["retrieved_files"] = [
                str(p.relative_to(self.settings.repos_root)) if p.is_absolute() else str(p)
                for p in paths
            ]
        # When the parser yielded no answer, surface enough of the raw
        # stdout to debug it without re-running. CLIs sometimes write the
        # final answer to stderr, or emit a JSON shape we don't yet
        # recognise — both look like "empty answer" without this hint.
        if not parsed.answer:
            extra["stdout_len"] = len(res.stdout)
            extra["stderr_len"] = len(res.stderr or "")
            if res.stdout:
                extra["stdout_head"] = res.stdout[:400]
                extra["stdout_tail"] = res.stdout[-400:]
            log.warning(
                "%s produced empty answer; stdout_len=%d stderr_len=%d head=%r",
                self.name, len(res.stdout), len(res.stderr or ""), res.stdout[:200],
            )
        return AdapterMetrics(
            duration_ms=int((time.monotonic() - t_start) * 1000),
            tokens_in=parsed.tokens_in,
            tokens_out=parsed.tokens_out,
            cost_usd=parsed.cost_usd,
            model=parsed.model,
            tool_calls=parsed.tool_calls,
            extra=extra,
        )
