"""Deterministic verification of cited file paths in decompositions."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..config import Settings
from ..models import Decomposition, Subtask


@dataclass
class VerifyResult:
    ok: bool
    verified_files: list[str] = field(default_factory=list)
    missing_files: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _resolve_file(settings: Settings, path: str) -> Path | None:
    """Try repo/path then bare path under repos_root."""
    p = path.strip().replace("\\", "/")
    if not p or p.startswith("http"):
        return None
    # repo-relative: backend-api/src/foo.py
    parts = p.split("/", 1)
    if len(parts) == 2 and parts[0] in settings.repos:
        candidate = settings.repo_path(parts[0]) / parts[1]
        if candidate.is_file():
            return candidate
    # search each repo for suffix match
    for repo in settings.repos:
        candidate = settings.repo_path(repo) / p
        if candidate.is_file():
            return candidate
        # basename fallback
        if "/" in p:
            tail = settings.repo_path(repo) / p.split("/", 1)[-1]
            if tail.is_file():
                return tail
    return None


def verify_decomposition(settings: Settings, decomp: Decomposition) -> VerifyResult:
    """Check that subtask file paths exist on disk."""
    verified: list[str] = []
    missing: list[str] = []
    seen: set[str] = set()

    def check(path: str) -> None:
        if not path or path in seen:
            return
        seen.add(path)
        if _resolve_file(settings, path):
            verified.append(path)
        else:
            missing.append(path)

    for st in decomp.subtasks:
        for f in st.files or []:
            check(f)

    notes: list[str] = []
    if missing:
        notes.append(
            f"{len(missing)} cited file(s) not found under REPOS_ROOT — "
            "may be hallucinated or outside indexed repos"
        )

    return VerifyResult(
        ok=len(missing) == 0,
        verified_files=verified,
        missing_files=missing,
        notes=notes,
    )


def apply_verifier_warnings(decomp: Decomposition, vr: VerifyResult) -> Decomposition:
    """Append missing-file warnings to risks without mutating subtasks."""
    if vr.ok:
        return decomp
    extra = list(decomp.risks or [])
    for path in vr.missing_files[:5]:
        msg = f"Unverified file path (not found on disk): `{path}`"
        if msg not in extra:
            extra.append(msg)
    if len(vr.missing_files) > 5:
        extra.append(f"... and {len(vr.missing_files) - 5} more unverified paths")
    return decomp.model_copy(update={"risks": extra})
