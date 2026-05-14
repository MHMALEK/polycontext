"""Registry behavior — name lookup, missing-dep handling, allowlist."""
from __future__ import annotations

import pytest

from tech_decomposition.adapters import get_adapter, list_adapters
from tech_decomposition.adapters import registry as reg
from tech_decomposition.config import Settings


def _settings(**overrides) -> Settings:
    return Settings(**overrides)


def test_unknown_adapter_raises_key_error():
    with pytest.raises(KeyError):
        get_adapter("does-not-exist", _settings())


def test_baseline_is_always_loadable():
    a = get_adapter("baseline", _settings())
    assert a.name == "baseline"
    assert "ask" in a.capabilities
    assert "decompose" in a.capabilities
    assert "implement" not in a.capabilities


def test_list_adapters_reports_every_registered_name():
    items = list_adapters(_settings())
    names = {i["name"] for i in items if i["installed"]}
    # At minimum the in-tree adapters are always installed.
    assert "baseline" in names
    # claude_sdk / aider may report installed=False if their optional deps
    # are missing in the test environment — that's fine, the registry should
    # still list them.
    all_names = {i["name"] for i in items}
    assert {"baseline", "claude_sdk", "aider", "aider_grounded",
            "opencode", "goose", "cursor", "cline", "cline_sdk",
            "tabby", "openhands"}.issubset(all_names)


def test_health_for_missing_dep_reports_reason():
    # If claude-agent-sdk isn't installed, list_adapters should show installed=False
    # with a non-empty reason, not crash.
    items = list_adapters(_settings())
    by_name = {i["name"]: i for i in items}
    if not by_name["claude_sdk"]["installed"]:
        assert by_name["claude_sdk"]["health"]["ok"] is False
        assert by_name["claude_sdk"]["health"]["reason"]


def test_programmatic_register_round_trip():
    reg.register("baseline_alias", "tech_decomposition.adapters._baseline", "BaselineAdapter")
    try:
        a = get_adapter("baseline_alias", _settings())
        # name comes from the class, not the registry key — that's intentional.
        assert a.name == "baseline"
    finally:
        # Clean up so test pollution doesn't bleed into other tests.
        reg._REGISTRY.pop("baseline_alias", None)
