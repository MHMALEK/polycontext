"""Task complexity routing for the tiered pipeline adapter.

Cheap heuristics first — no LLM call. Classifies ask/decompose inputs so the
pipeline can choose prefetch-only synthesis vs bounded agent fallback.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Literal

Job = Literal["ask", "decompose"]


class TaskTier(str, Enum):
    """How much agentic exploration the task likely needs."""

    SIMPLE = "simple"          # symbol lookup, single-file, strong search terms
    ENUMERATION = "enumeration"  # list-all / role inventory style
    TRACE = "trace"            # behavior flow, multi-hop "what happens when"
    COMPLEX = "complex"        # migrations, cross-repo decompose, ambiguous


@dataclass(frozen=True)
class RouteDecision:
    tier: TaskTier
    prefetch_top_k: int
    prefer_single_shot: bool
    allow_agent_fallback: bool
    reason: str


_RX_ENUM = re.compile(
    r"\b(list\s+all|what\s+are\s+(the\s+)?|enumerate|describe\s+each|every\s+)\b",
    re.IGNORECASE,
)
_RX_TRACE = re.compile(
    r"\b(what\s+happens|when\s+(a\s+)?user|flow|trace|walk\s+me\s+through|"
    r"end[\s-]to[\s-]end|sequence|step[\s-]by[\s-]step|will\s+I\s+see|"
    r"when\s+I\s+import|if\s+I\s+(only\s+)?select)\b",
    re.IGNORECASE,
)
_RX_DEEP = re.compile(
    r"\b(stateless|introspection|every\s+REST|JWKS|architecture|call\s+chain)\b",
    re.IGNORECASE,
)
_RX_VALIDATION = re.compile(
    r"\b(regex|validator|validation|pattern|unicode|special\s+char)\b",
    re.IGNORECASE,
)
_RX_MIGRATION = re.compile(
    r"\b(move|port|migrate|cloud\s+function|refactor|extract)\b",
    re.IGNORECASE,
)
_RX_ROLES = re.compile(r"\b(user\s+roles?|rbac|permission)\b", re.IGNORECASE)


def enumeration_boost_terms(query: str) -> list[str]:
    """Extra zoekt terms for list-all / role-inventory questions."""
    terms: list[str] = []
    if _RX_ROLES.search(query):
        terms.extend(["UserRoles", "DATA_", "RolesChecker", "constants.py"])
    if _RX_ENUM.search(query):
        terms.append("StrEnum")
    return terms


def validation_boost_terms(query: str) -> list[str]:
    """Extra zoekt terms for validator / regex lookup questions."""
    terms: list[str] = []
    if _RX_VALIDATION.search(query):
        terms.extend(["validator", "ValidationMode", "regex", "pattern", "IDENTIFIER"])
    if "farm" in query.lower():
        terms.extend(["FarmName", "farm_name", "ReferenceID"])
    if "geojson" in query.lower() or "country" in query.lower():
        terms.extend(["geojson_validator", "ValidationMode", "NodeID"])
    return terms


def retrieval_boost_terms(query: str, tags: list[str] | None) -> list[str]:
    """Merge enumeration + validation boost terms for prefetch."""
    tag_set = {t.lower() for t in (tags or [])}
    out: list[str] = []
    seen: set[str] = set()

    def add(items: list[str]) -> None:
        for t in items:
            low = t.lower()
            if low not in seen:
                seen.add(low)
                out.append(t)

    if "enumeration" in tag_set or _RX_ENUM.search(query):
        add(enumeration_boost_terms(query))
    if "validation" in tag_set or _RX_VALIDATION.search(query):
        add(validation_boost_terms(query))
    if "master-data" in tag_set:
        add(["master_data", "MasterData", "upload", "template"])
    if "behavior" in tag_set:
        add([
            "includeGeolocation",
            "includeDeforestation",
            "GeoSpatialDumpStrategy",
            "DataSharing",
        ])
    return out


def should_escalate_synthesis(
    *, tags: list[str] | None, coverage_sufficient: bool,
) -> bool:
    """Use Pro for single-shot synthesis on hard case types (not agent fallback)."""
    tag_set = {t.lower() for t in (tags or [])}
    if "validation" in tag_set or "master-data" in tag_set:
        return True
    if "cross-repo" in tag_set and not coverage_sufficient:
        return True
    return False


def should_escalate_fallback(*, route: RouteDecision, tags: list[str] | None) -> bool:
    """Use Pro model for agent fallback on complex cross-repo traces only."""
    tag_set = {t.lower() for t in (tags or [])}
    if route.tier == TaskTier.COMPLEX and "cross-repo" in tag_set:
        return True
    return False


_RX_SYNTHESIS_INSUFFICIENT = re.compile(
    r"\b("
    r"not (?:possible|in snippets)|insufficient|cannot determine|can't determine|"
    r"do(?:es)? not contain|snippets do not|not (?:found|present) in the (?:provided )?snippets|"
    r"(?:unable|can't) to (?:answer|determine)|no (?:information|evidence)|"
    r"not provided in the snippets"
    r")\b",
    re.IGNORECASE,
)


def answer_signals_insufficient(text: str) -> bool:
    """True when synthesis explicitly says snippets were not enough."""
    t = (text or "").strip()
    if not t:
        return True
    return bool(_RX_SYNTHESIS_INSUFFICIENT.search(t))


def should_agent_fallback(
    *,
    route: RouteDecision,
    tags: list[str] | None,
    coverage_sufficient: bool,
    synthesis_text: str,
) -> bool:
    """Agent only for trace/complex cases with low coverage AND weak synthesis."""
    if not route.allow_agent_fallback:
        return False
    tag_set = {t.lower() for t in (tags or [])}
    if "validation" in tag_set or "master-data" in tag_set:
        return False
    if route.tier not in (TaskTier.TRACE, TaskTier.COMPLEX):
        return False
    if coverage_sufficient:
        return False
    return answer_signals_insufficient(synthesis_text)


def classify_task(*, query: str, job: Job, tags: list[str] | None = None) -> RouteDecision:
    """Heuristic router — zero LLM cost."""
    q = (query or "").strip()
    tag_set = {t.lower() for t in (tags or [])}

    if job == "decompose":
        top_k = 15
        if "cross-repo" in tag_set or _RX_MIGRATION.search(q):
            return RouteDecision(
                tier=TaskTier.COMPLEX,
                prefetch_top_k=top_k,
                prefer_single_shot=False,
                allow_agent_fallback=True,
                reason="decompose ticket with migration/cross-repo signals",
            )
        return RouteDecision(
            tier=TaskTier.COMPLEX,
            prefetch_top_k=top_k,
            prefer_single_shot=True,
            allow_agent_fallback=True,
            reason="decompose: prefetch + synthesis first, agent if coverage low",
        )

    # --- ask ---
    if "enumeration" in tag_set or (_RX_ENUM.search(q) and "how does" not in q.lower()):
        return RouteDecision(
            tier=TaskTier.ENUMERATION,
            prefetch_top_k=12,
            prefer_single_shot=True,
            allow_agent_fallback=True,
            reason="enumeration-style question",
        )

    if (
        "behavior" in tag_set
        or "architecture" in tag_set
        or _RX_TRACE.search(q)
        or _RX_DEEP.search(q)
    ):
        return RouteDecision(
            tier=TaskTier.TRACE,
            prefetch_top_k=18,
            prefer_single_shot=False,
            allow_agent_fallback=True,
            reason="behavior/architecture trace",
        )

    if _RX_VALIDATION.search(q) or "validation" in tag_set:
        return RouteDecision(
            tier=TaskTier.TRACE,
            prefetch_top_k=18,
            prefer_single_shot=False,
            allow_agent_fallback=True,
            reason="validator/regex lookup",
        )

    if "cross-repo" in tag_set:
        return RouteDecision(
            tier=TaskTier.COMPLEX,
            prefetch_top_k=10,
            prefer_single_shot=False,
            allow_agent_fallback=True,
            reason="cross-repo ask",
        )

    return RouteDecision(
        tier=TaskTier.SIMPLE,
        prefetch_top_k=8,
        prefer_single_shot=True,
        allow_agent_fallback=True,
        reason="default simple ask",
    )
