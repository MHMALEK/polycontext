"""Case loading.

Three job types live in three different formats — pragmatic choice, kept
to whatever was easiest to author by hand:

  * **ask**       — ``eval/questions.toml`` (already existed; we just read it)
  * **decompose** — one YAML per case in ``eval/cases/decompose/``
  * **implement** — one YAML per case in ``eval/cases/implement/``

Each loaded case has a stable ``Case`` shape regardless of source format, so
the runner and scorer don't care where it came from.
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml


Job = Literal["ask", "decompose", "implement"]


@dataclass
class Case:
    """A single evaluation case.

    ``input`` is the request body for the adapter call. ``expected`` is
    optional scoring criteria — exact contents differ by job type, see
    ``scorer.py`` for what each kind of rubric expects.
    """

    id: str
    job: Job
    tags: list[str] = field(default_factory=list)
    description: str = ""
    input: dict[str, Any] = field(default_factory=dict)
    expected: dict[str, Any] = field(default_factory=dict)
    source_path: Path | None = None


# ---------------------------------------------------------------------------
# ask — TOML, already in tree
# ---------------------------------------------------------------------------


def load_ask_cases(toml_path: Path) -> list[Case]:
    """Parse the legacy ``questions.toml`` file into ``Case`` objects.

    Each ``[[questions]]`` table becomes one case. Rubric (``expected``) is
    optional — when absent the scorer falls back to rule-checks that don't
    depend on golden answers (non-empty response, has citations, etc.).
    """
    if not toml_path.is_file():
        return []
    data = tomllib.loads(toml_path.read_text())
    out: list[Case] = []
    for q in data.get("questions", []):
        out.append(Case(
            id=q["id"],
            job="ask",
            tags=list(q.get("tags", [])),
            description=q.get("description", ""),
            input={"query": q["text"].strip()},
            expected=q.get("expected", {}),
            source_path=toml_path,
        ))
    return out


# ---------------------------------------------------------------------------
# decompose / implement — YAML, one file per case
# ---------------------------------------------------------------------------


def load_yaml_cases(dir_path: Path, job: Job) -> list[Case]:
    if not dir_path.is_dir():
        return []
    out: list[Case] = []
    for p in sorted(dir_path.glob("*.yaml")) + sorted(dir_path.glob("*.yml")):
        with p.open() as f:
            doc = yaml.safe_load(f) or {}
        out.append(Case(
            id=doc.get("id") or p.stem,
            job=job,
            tags=list(doc.get("tags", [])),
            description=doc.get("description", ""),
            input=dict(doc.get("input", {})),
            expected=dict(doc.get("expected", {})),
            source_path=p,
        ))
    return out


# ---------------------------------------------------------------------------
# Index — load everything from the standard layout under ``eval/``
# ---------------------------------------------------------------------------


def discover_cases(eval_dir: Path) -> list[Case]:
    """Load every case under the standard ``eval/`` layout.

    Layout:
        eval/questions.toml          → ask cases
        eval/cases/decompose/*.yaml  → decompose cases
        eval/cases/implement/*.yaml  → implement cases
    """
    cases: list[Case] = []
    cases.extend(load_ask_cases(eval_dir / "questions.toml"))
    cases.extend(load_yaml_cases(eval_dir / "cases" / "decompose", "decompose"))
    cases.extend(load_yaml_cases(eval_dir / "cases" / "implement", "implement"))
    return cases


def filter_cases(
    cases: list[Case],
    *,
    job: Job | None = None,
    ids: list[str] | None = None,
    tags: list[str] | None = None,
) -> list[Case]:
    """Apply CLI filters: by job, by id allowlist, by tag intersection."""
    out = cases
    if job:
        out = [c for c in out if c.job == job]
    if ids:
        id_set = set(ids)
        out = [c for c in out if c.id in id_set]
    if tags:
        tag_set = set(tags)
        out = [c for c in out if tag_set.intersection(c.tags)]
    return out
