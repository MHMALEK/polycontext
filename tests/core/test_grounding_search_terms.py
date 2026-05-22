"""General grounding term extraction — not tied to a single eval case."""
from __future__ import annotations

from tech_decomposition.core.grounding import (
    _adjacent_lowercase_pair_variants,
    _expand_with_variants,
    _extract_question_bigrams,
    _extract_search_terms,
)


def test_bigrams_from_plain_english_question():
    pairs = _extract_question_bigrams(
        "what are the current user roles we have in the platform"
    )
    assert "user roles" in pairs


def test_bigrams_from_validation_question():
    pairs = _extract_question_bigrams(
        "Where is Farm Name validated in the frontend?"
    )
    assert any("farm" in p.lower() and "name" in p.lower() for p in pairs)


def test_expand_adjacent_lowercase_to_compound():
    terms = ["user", "roles"]
    variants = _adjacent_lowercase_pair_variants(terms)
    assert "UserRoles" in variants or "user_roles" in variants


def test_expand_title_case_phrase():
    terms = _extract_search_terms("How does Data Sharing geolocation work?")
    expanded = _expand_with_variants(terms)
    assert any("DataSharing" in v or "data_sharing" in v.lower() for v in expanded)
