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


def test_cursor_is_loadable():
    # Cursor has no runtime deps beyond the cursor-agent binary, which
    # ``health()`` will report on but the class itself always instantiates.
    a = get_adapter("cursor", _settings())
    assert a.name == "cursor"
    assert {"ask", "decompose", "implement"}.issubset(a.capabilities)


def test_list_adapters_reports_every_registered_name():
    items = list_adapters(_settings())
    all_names = {i["name"] for i in items}
    # The current narrowed adapter set — see registry.py.
    assert all_names == {
        "cline_sdk", "cline_sdk_grounded",
        "opencode", "opencode_grounded",
        "cursor",
    }


def test_health_for_missing_sidecar_reports_reason():
    # When the cline_sdk bridge isn't configured the entry should still
    # list with installed=True but health.ok=False and a useful reason.
    items = list_adapters(_settings())
    by_name = {i["name"]: i for i in items}
    cs = by_name["cline_sdk"]
    if not cs["health"]["ok"]:
        assert cs["health"]["reason"]


def test_programmatic_register_round_trip():
    reg.register("cursor_alias", "tech_decomposition.adapters._cursor", "CursorAdapter")
    try:
        a = get_adapter("cursor_alias", _settings())
        # name comes from the class, not the registry key — that's intentional.
        assert a.name == "cursor"
    finally:
        # Clean up so test pollution doesn't bleed into other tests.
        reg._REGISTRY.pop("cursor_alias", None)
