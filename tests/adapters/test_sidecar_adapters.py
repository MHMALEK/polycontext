"""Health-check tests for Tabby and OpenHands adapters.

These don't talk to real services — they verify the adapter's ``health()``
contract: clear ``ok=False`` with a useful reason when the URL is unset
or unreachable, ``ok=True`` only when the HTTP probe succeeds.

Real end-to-end testing against the containers happens in the bake-off
once ``make adapters`` is running.
"""
from __future__ import annotations

import httpx
import pytest

from tech_decomposition.adapters._openhands import OpenHandsAdapter
from tech_decomposition.adapters._tabby import TabbyAdapter
from tech_decomposition.config import Settings


def _settings(**overrides) -> Settings:
    # Force the unset values explicitly — the user's real .env may have these
    # set, which would silently invalidate the "url not configured" tests.
    defaults = {"tabby_base_url": "", "openhands_base_url": ""}
    defaults.update(overrides)
    return Settings(**defaults)


# ----- Tabby --------------------------------------------------------------


def test_tabby_health_without_url():
    a = TabbyAdapter(_settings())
    h = a.health()
    assert h["ok"] is False
    assert "TABBY_BASE_URL" in h["reason"]


def test_tabby_health_unreachable_url():
    # Port 0 is reserved — guaranteed to fail to connect.
    a = TabbyAdapter(_settings(tabby_base_url="http://127.0.0.1:1"))
    h = a.health()
    assert h["ok"] is False
    assert "unreachable" in h["reason"].lower() or "tabby" in h["reason"].lower()


def test_tabby_capabilities():
    assert TabbyAdapter.capabilities == {"ask", "decompose"}
    assert "implement" not in TabbyAdapter.capabilities


# ----- OpenHands ----------------------------------------------------------


def test_openhands_health_without_url():
    a = OpenHandsAdapter(_settings())
    h = a.health()
    assert h["ok"] is False
    assert "OPENHANDS_BASE_URL" in h["reason"]


def test_openhands_health_unreachable_url():
    a = OpenHandsAdapter(_settings(openhands_base_url="http://127.0.0.1:1"))
    h = a.health()
    assert h["ok"] is False


def test_openhands_capabilities():
    assert OpenHandsAdapter.capabilities == {"ask", "decompose", "implement"}


# ----- Health-OK path with a stub server --------------------------------


@pytest.mark.asyncio
async def test_tabby_health_ok_with_stub(monkeypatch):
    """When ``/v1/health`` returns 200, ``health()`` reports ok."""

    class _Resp:
        status_code = 200

    class _Client:
        def __init__(self, *_, **__): ...
        def __enter__(self): return self
        def __exit__(self, *_): return False
        def get(self, url, *args, **kwargs): return _Resp()

    monkeypatch.setattr(httpx, "Client", _Client)
    a = TabbyAdapter(_settings(tabby_base_url="http://stub"))
    assert a.health()["ok"] is True


@pytest.mark.asyncio
async def test_openhands_health_ok_with_stub(monkeypatch):
    class _Resp:
        status_code = 200

    class _Client:
        def __init__(self, *_, **__): ...
        def __enter__(self): return self
        def __exit__(self, *_): return False
        def get(self, url, *args, **kwargs): return _Resp()

    monkeypatch.setattr(httpx, "Client", _Client)
    a = OpenHandsAdapter(_settings(openhands_base_url="http://stub"))
    assert a.health()["ok"] is True
