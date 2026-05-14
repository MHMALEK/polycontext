"""Adapter discovery — name → class lookup.

Registration is by **late import**: each adapter module is imported lazily on
first lookup so a missing optional dependency (e.g. ``aider`` not installed)
doesn't take down the whole API. ``health()`` reports the failure if anyone
asks for that adapter specifically.

To add a new adapter:
    1. Implement ``Adapter`` in a new ``_yourname.py`` module.
    2. Add an entry to ``_REGISTRY`` below.
"""
from __future__ import annotations

import importlib
from typing import Callable

from .base import Adapter


# name → (module, classname). Late-imported on demand.
_REGISTRY: dict[str, tuple[str, str]] = {
    # Phase 1: baseline + in-process Python adapters
    "baseline": ("tech_decomposition.adapters._baseline", "BaselineAdapter"),
    "claude_sdk": ("tech_decomposition.adapters._claude_sdk", "ClaudeSDKAdapter"),
    "aider": ("tech_decomposition.adapters._aider", "AiderAdapter"),
    # Experiment: Aider with cross-repo retrieval prelude (Sourcebot + ripgrep)
    "aider_grounded": ("tech_decomposition.adapters._aider_grounded", "AiderGroundedAdapter"),
    # Phase 2: CLI-backed adapters
    "opencode": ("tech_decomposition.adapters._opencode", "OpenCodeAdapter"),
    "goose": ("tech_decomposition.adapters._goose", "GooseAdapter"),
    "cursor": ("tech_decomposition.adapters._cursor", "CursorAdapter"),
    "cline": ("tech_decomposition.adapters._cline", "ClineAdapter"),
    "cline_sdk": ("tech_decomposition.adapters._cline_sdk", "ClineSDKAdapter"),
    # Phase 3: HTTP sidecars (Docker — see compose.adapters.yaml)
    "tabby": ("tech_decomposition.adapters._tabby", "TabbyAdapter"),
    "openhands": ("tech_decomposition.adapters._openhands", "OpenHandsAdapter"),
}


def _load_class(name: str) -> type[Adapter]:
    if name not in _REGISTRY:
        raise KeyError(f"unknown adapter: {name!r}. Known: {sorted(_REGISTRY)}")
    module_name, class_name = _REGISTRY[name]
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    if not issubclass(cls, Adapter):
        raise TypeError(f"{module_name}.{class_name} is not an Adapter subclass")
    return cls


def get_adapter(name: str, settings) -> Adapter:
    """Instantiate the named adapter with the given Settings.

    Raises:
        KeyError: name is not in the registry.
        ImportError: adapter's optional dependency is missing — caller can
            translate this to a 501/503.
    """
    cls = _load_class(name)
    return cls(settings)


def list_adapters(settings) -> list[dict]:
    """Best-effort listing of every registered adapter and what it supports.

    Adapters whose optional deps are missing are reported with ``installed=False``
    and ``health.ok=False`` rather than omitted, so the UI can show a greyed-out
    entry and tell the user how to enable it.
    """
    out: list[dict] = []
    for name in _REGISTRY:
        try:
            cls = _load_class(name)
        except ImportError as e:
            out.append({
                "name": name,
                "installed": False,
                "capabilities": [],
                "description": "",
                "health": {"ok": False, "reason": f"import failed: {e}"},
            })
            continue
        try:
            instance = cls(settings)
            health = instance.health()
        except Exception as e:  # noqa: BLE001 — health is best-effort
            health = {"ok": False, "reason": f"{type(e).__name__}: {e}"}
        out.append({
            "name": cls.name,
            "installed": True,
            "capabilities": sorted(cls.capabilities),
            "description": cls.description,
            "health": health,
        })
    return out


def register(name: str, module: str, classname: str) -> None:
    """Programmatic registration — useful for tests and plugins."""
    _REGISTRY[name] = (module, classname)
