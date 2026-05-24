"""Per-repo cap + on-disk snippet expansion."""
from __future__ import annotations

from pathlib import Path

from tech_decomposition.core.grounding import (
    GroundingSnippet,
    _sanitize_llm_terms,
    _symbol_candidates,
    apply_per_repo_cap,
    expand_snippets_from_disk,
    normalize_repo_name,
    normalize_sourcebot_snippets,
)


def _snip(repo: str, path: str, start: int | None = None, end: int | None = None, content: str = "x") -> GroundingSnippet:
    return GroundingSnippet(repo=repo, path=path, start_line=start, end_line=end, content=content)


def test_per_repo_cap_keeps_first_n_per_repo_preserving_order():
    snippets = [
        _snip("a", "1.py"),
        _snip("a", "2.py"),
        _snip("b", "1.py"),
        _snip("a", "3.py"),   # 3rd from a — kept
        _snip("a", "4.py"),   # 4th from a — dropped (cap=3)
        _snip("b", "2.py"),
        _snip("c", "1.py"),
    ]
    out = apply_per_repo_cap(snippets, cap=3)
    paths = [(s.repo, s.path) for s in out]
    assert paths == [("a", "1.py"), ("a", "2.py"), ("b", "1.py"), ("a", "3.py"), ("b", "2.py"), ("c", "1.py")]


def test_per_repo_cap_cap_zero_or_negative_disables():
    snippets = [_snip("a", f"{i}.py") for i in range(5)]
    assert apply_per_repo_cap(snippets, cap=0) == snippets
    assert apply_per_repo_cap(snippets, cap=-1) == snippets


def test_per_repo_cap_empty_repo_passes_through():
    # snippets with no repo identity aren't grouped — keep them all so we don't
    # silently drop hits that lost their repo label upstream.
    snippets = [_snip("", "1.py"), _snip("", "2.py"), _snip("", "3.py"), _snip("", "4.py")]
    out = apply_per_repo_cap(snippets, cap=2)
    assert len(out) == 4


def test_expand_snippets_from_disk_widens_window(tmp_path: Path):
    repo_dir = tmp_path / "demo"
    src_dir = repo_dir / "src"
    src_dir.mkdir(parents=True)
    file_lines = [f"line_{i}" for i in range(1, 41)]
    (src_dir / "module.py").write_text("\n".join(file_lines) + "\n")

    snip = _snip("demo", "src/module.py", start=20, end=20, content="line_20")
    out = expand_snippets_from_disk(
        [snip], repos_root=tmp_path, context_lines=5, max_lines=120,
    )
    assert len(out) == 1
    expanded = out[0]
    # ±5 lines around line 20 → lines 15..25 (11 lines total).
    assert expanded.start_line == 15
    assert expanded.end_line == 25
    body = expanded.content.splitlines()
    assert body[0] == "line_15"
    assert body[-1] == "line_25"
    # Hit line preserved verbatim.
    assert "line_20" in expanded.content


def test_expand_snippets_clamps_at_file_boundaries(tmp_path: Path):
    repo_dir = tmp_path / "demo"
    repo_dir.mkdir()
    (repo_dir / "short.py").write_text("a\nb\nc\n")
    snip = _snip("demo", "short.py", start=1, end=1, content="a")
    out = expand_snippets_from_disk(
        [snip], repos_root=tmp_path, context_lines=50, max_lines=120,
    )
    assert out[0].start_line == 1
    assert out[0].end_line == 3
    assert out[0].content.splitlines() == ["a", "b", "c"]


def test_expand_snippets_respects_max_lines_cap(tmp_path: Path):
    repo_dir = tmp_path / "demo"
    repo_dir.mkdir()
    (repo_dir / "big.py").write_text("\n".join(f"l{i}" for i in range(1, 201)) + "\n")
    snip = _snip("demo", "big.py", start=100, end=100, content="l100")
    out = expand_snippets_from_disk(
        [snip], repos_root=tmp_path, context_lines=60, max_lines=20,
    )
    assert (out[0].end_line - out[0].start_line + 1) <= 20
    assert out[0].start_line <= 100 <= out[0].end_line


def test_expand_snippets_skips_missing_files(tmp_path: Path):
    snip = _snip("ghost", "nowhere.py", start=1, end=1, content="orig")
    out = expand_snippets_from_disk(
        [snip], repos_root=tmp_path, context_lines=10, max_lines=120,
    )
    assert out == [snip]


def test_expand_snippets_skips_when_disabled(tmp_path: Path):
    snip = _snip("demo", "x.py", start=10, end=10, content="orig")
    assert expand_snippets_from_disk([snip], repos_root=tmp_path, context_lines=0, max_lines=120) == [snip]


def test_symbol_candidates_picks_real_symbols_skips_english():
    terms = [
        "UserRoles",            # PascalCase ≥ 2 segs — yes
        "NODE_NAME_PATTERN",    # UPPER_SNAKE — yes
        "RolesChecker",         # PascalCase — yes
        "get_identity",         # snake_case_with_two_segments — yes
        "foo.bar",              # dotted — yes
        "user",                 # plain English — no
        "role",                 # plain English — no
        "data viewer",          # multi-word lowercase — no
        "JWT",                  # single uppercase token, no underscore — no (LSP can't match)
    ]
    out = _symbol_candidates(terms, cap=10)
    assert "UserRoles" in out
    assert "NODE_NAME_PATTERN" in out
    assert "RolesChecker" in out
    assert "get_identity" in out
    assert "foo.bar" in out
    assert "user" not in out
    assert "role" not in out
    assert "data viewer" not in out
    assert "JWT" not in out


def test_symbol_candidates_caps_and_dedupes():
    terms = ["UserRoles", "userroles", "USERROLES", "RolesChecker", "AnotherClass", "ThirdClass"]
    out = _symbol_candidates(terms, cap=2)
    assert len(out) == 2
    # Case-insensitive dedup: 'UserRoles', 'userroles', 'USERROLES' collapse to 1.
    assert out[0] == "UserRoles"


def test_sanitize_llm_terms_keeps_real_search_tokens():
    out = _sanitize_llm_terms([
        "UserRoles", "Farm Name", "Data viewer", "rbac.py",
        "NODE_NAME_PATTERN",
    ])
    assert out == ["UserRoles", "Farm Name", "Data viewer", "rbac.py", "NODE_NAME_PATTERN"]


def test_sanitize_llm_terms_drops_python_statements():
    out = _sanitize_llm_terms([
        "class LoadDomain(StrEnum):",
        "MASTER = ",
        "def get_user_identity",
        "from foo import bar",
        "UserRoles",  # the one survivor
    ])
    assert out == ["UserRoles"]


def test_sanitize_llm_terms_drops_test_fixture_symbols():
    out = _sanitize_llm_terms([
        "test_client_data_uploader",
        "conftest",
        "pytest.fixture",
        "MockResponse",
        "TestClient",
        "fixture",
        "UserRoles",
    ])
    assert out == ["UserRoles"]


def test_sanitize_llm_terms_drops_template_strings_and_long_phrases():
    out = _sanitize_llm_terms([
        "{TEST_RATE_LIMIT}/minute",
        "row per country, single source of truth, query by foreign key",  # > 40 chars
        "df.columns: df[",
        "_normalize_excel_date - value after: {v}",
        "UserRoles",
    ])
    assert out == ["UserRoles"]


def test_sanitize_llm_terms_drops_grep_line_numbers_and_multiline():
    out = _sanitize_llm_terms([
        "  126:def authorize",
        "rbac.py\\nconstants.py",
        "Real Term",
    ])
    assert out == ["Real Term"]


def test_normalize_repo_name_picks_known_repo_suffix():
    known = ["traceability", "frontend", "data", "data-cloud-functions"]
    # Prefer the longest match so "data-cloud-functions" doesn't get shortened to "data".
    assert normalize_repo_name("gitlab.com/tract1/application/api/traceability", known_repos=known) == "traceability"
    assert normalize_repo_name("gitlab.com/tract1/application/frontend", known_repos=known) == "frontend"
    assert normalize_repo_name("gitlab.com/tract1/application/data-cloud-functions", known_repos=known) == "data-cloud-functions"
    assert normalize_repo_name("gitlab.com/tract1/application/data", known_repos=known) == "data"


def test_normalize_repo_name_falls_back_to_last_segment_when_no_match():
    assert normalize_repo_name("github.com/foo/bar/unknown-repo", known_repos=["x", "y"]) == "unknown-repo"


def test_normalize_repo_name_empty_inputs():
    assert normalize_repo_name("", known_repos=["x"]) == ""
    assert normalize_repo_name("traceability", known_repos=[]) == "traceability"
    assert normalize_repo_name("a/b/c", known_repos=[""]) == "c"


def test_normalize_sourcebot_snippets_rewrites_only_when_changed():
    known = ["traceability", "frontend"]
    snips = [
        GroundingSnippet(repo="gitlab.com/tract1/application/api/traceability", path="src/x.py", content="..."),
        GroundingSnippet(repo="frontend", path="src/x.ts", content="..."),
        GroundingSnippet(repo="", path="loose.py", content="..."),
    ]
    out = normalize_sourcebot_snippets(snips, known_repos=known)
    assert [s.repo for s in out] == ["traceability", "frontend", ""]
    # Original list not mutated.
    assert snips[0].repo == "gitlab.com/tract1/application/api/traceability"


def test_expand_snippets_finds_file_after_repo_normalization(tmp_path: Path):
    # End-to-end: a sourcebot snippet with the full gitlab URL should NOT
    # resolve on disk; after normalization it should.
    repo_dir = tmp_path / "traceability" / "src" / "common"
    repo_dir.mkdir(parents=True)
    (repo_dir / "constants.py").write_text("\n".join(f"l{i}" for i in range(1, 21)) + "\n")

    raw = GroundingSnippet(
        repo="gitlab.com/tract1/application/api/traceability",
        path="src/common/constants.py",
        start_line=10, end_line=10, content="l10",
    )
    # Pre-normalization: expansion can't find the file.
    pre = expand_snippets_from_disk([raw], repos_root=tmp_path, context_lines=3, max_lines=120)
    assert pre[0].content == "l10"   # unchanged
    # Post-normalization: expansion finds it.
    normed = normalize_sourcebot_snippets([raw], known_repos=["traceability"])
    post = expand_snippets_from_disk(normed, repos_root=tmp_path, context_lines=3, max_lines=120)
    body = post[0].content.splitlines()
    assert body == ["l7", "l8", "l9", "l10", "l11", "l12", "l13"]


def test_expand_snippets_skips_snippets_without_start_line(tmp_path: Path):
    repo_dir = tmp_path / "demo"
    repo_dir.mkdir()
    (repo_dir / "x.py").write_text("a\nb\nc\n")
    snip = _snip("demo", "x.py", start=None, end=None, content="orig")
    out = expand_snippets_from_disk(
        [snip], repos_root=tmp_path, context_lines=10, max_lines=120,
    )
    assert out == [snip]
