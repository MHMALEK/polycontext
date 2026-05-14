"""Contract tests for the Adapter ABC.

Verifies:
  * an adapter that declares no capabilities raises NotSupported on every method
  * subclasses that override only some methods don't accidentally inherit a
    working impl of the others
  * supports() reflects the capabilities ClassVar
"""
from __future__ import annotations

import pytest

from tech_decomposition.adapters import (
    Adapter,
    AdapterAskInput,
    AdapterDecomposeInput,
    AdapterImplementInput,
    NotSupported,
)
from tech_decomposition.adapters.base import ImplementContext


class _EmptyAdapter(Adapter):
    name = "empty"
    capabilities = set()


class _AskOnlyAdapter(Adapter):
    name = "ask_only"
    capabilities = {"ask"}

    async def ask(self, inp):
        return type("R", (), {"adapter": "ask_only", "answer": "ok"})()


@pytest.mark.asyncio
async def test_empty_adapter_raises_not_supported_for_all():
    a = _EmptyAdapter(settings=None)
    with pytest.raises(NotSupported):
        await a.ask(AdapterAskInput(query="x"))
    with pytest.raises(NotSupported):
        await a.decompose(AdapterDecomposeInput(ticket_text="x"))
    with pytest.raises(NotSupported):
        await a.implement(
            AdapterImplementInput(repo="r", free_text="x"),
            ImplementContext(worktree_path="/tmp/x", branch="b", repo="r", base_branch="main"),
        )
    assert not a.supports("ask")
    assert not a.supports("decompose")
    assert not a.supports("implement")


@pytest.mark.asyncio
async def test_partial_override_leaves_others_unsupported():
    a = _AskOnlyAdapter(settings=None)
    assert a.supports("ask")
    assert not a.supports("decompose")
    with pytest.raises(NotSupported):
        await a.decompose(AdapterDecomposeInput(ticket_text="x"))


def test_not_supported_carries_adapter_and_capability():
    e = NotSupported("foo", "implement")
    assert e.adapter == "foo"
    assert e.capability == "implement"
    assert "foo" in str(e) and "implement" in str(e)
