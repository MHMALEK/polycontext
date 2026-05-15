"""Tests for the shared grounding helpers.

We unit-test the parts that are pure / hermetic — keyword extraction and
the prompt-block formatter. ``retrieve_context_paths`` itself talks to
Sourcebot + ripgrep and is exercised in the integration bake-off rather
than here.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from tech_decomposition.adapters._grounding import (
    build_keyword_query,
    format_grounding_block,
)


def _fake_settings(repos: list[str] | None = None) -> SimpleNamespace:
    return SimpleNamespace(repos=repos or ["repo-a", "repo-b"])


def test_build_keyword_query_extracts_distinctive_tokens() -> None:
    text = "Why does SupplierNode.create_master_data raise IntegrityError in p2_dev?"
    q = build_keyword_query(text, _fake_settings())
    assert "SupplierNode.create_master_data" in q.code_keywords
    assert "IntegrityError" in q.code_keywords
    # Search queries are the first 6 keywords by construction.
    assert q.search_queries == q.code_keywords[:6]
    assert q.suspected_repos == ["repo-a", "repo-b"]


def test_build_keyword_query_skips_stopwords_and_generic() -> None:
    text = "The validation function for this file class"
    q = build_keyword_query(text, _fake_settings())
    # All tokens above are stopwords or generic — keywords should be empty.
    assert q.code_keywords == []


def test_build_keyword_query_dedupes_case_insensitive() -> None:
    text = "FooBar foobar FOOBAR FooBar"
    q = build_keyword_query(text, _fake_settings())
    assert len(q.code_keywords) == 1
    # Preserves first-seen casing.
    assert q.code_keywords[0] == "FooBar"


def test_build_keyword_query_caps_at_twelve() -> None:
    tokens = [f"Symbol{i}" for i in range(30)]
    q = build_keyword_query(" ".join(tokens), _fake_settings())
    assert len(q.code_keywords) == 12


def test_format_grounding_block_renders_relative_paths(tmp_path: Path) -> None:
    repo = tmp_path / "repo-a"
    (repo / "src").mkdir(parents=True)
    f1 = repo / "src" / "a.py"
    f2 = repo / "src" / "b.py"
    f1.touch()
    f2.touch()

    block = format_grounding_block([f1, f2], tmp_path)
    assert "Relevant files retrieved" in block
    assert "- repo-a/src/a.py" in block
    assert "- repo-a/src/b.py" in block
    # Block ends with a blank line so callers can prepend it cleanly.
    assert block.endswith("\n\n")


def test_format_grounding_block_empty_returns_empty(tmp_path: Path) -> None:
    assert format_grounding_block([], tmp_path) == ""


def test_format_grounding_block_falls_back_when_outside_root(tmp_path: Path) -> None:
    outside = Path("/tmp/some/other/place.py")
    block = format_grounding_block([outside], tmp_path)
    assert str(outside) in block
