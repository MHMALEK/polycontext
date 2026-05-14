"""pytest config for the adapter bake-off tests.

Adds an ``asyncio_mode = auto`` fallback so individual tests don't need
``@pytest.mark.asyncio`` if pytest-asyncio is installed. The marker is still
used explicitly in places for clarity.
"""
import pytest


def pytest_collection_modifyitems(config, items):
    # Auto-mark any ``async def`` test with asyncio so they actually run.
    for item in items:
        if "asyncio" in item.keywords:
            continue
        if isinstance(item, pytest.Function) and item.obj.__code__.co_flags & 0x100:
            item.add_marker(pytest.mark.asyncio)
