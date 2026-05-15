"""Adapter discovery — name → class lookup.

Registration is by **late import**: each adapter module is imported lazily on
first lookup so a missing optional dependency doesn't take down the whole API.
``health()`` reports the failure if anyone asks for that adapter specifically.

To add a new adapter:
    1. Implement ``Adapter`` in a new ``_yourname.py`` module.
    2. Add an entry to ``_REGISTRY`` below.

Current set (post-narrowing): the bake-off has converged on Cline (SDK) and
OpenCode as the strongest agents, with Cursor kept as a frozen secondary.
Each primary has a ``_grounded`` sibling that pre-loads Sourcebot+ripgrep
file pointers into the prompt.

The legacy ``/ask`` endpoint (Sourcebot retrieval + Gemini answer) still
exists in ``api.py`` and is what the UI calls when no adapter is selected —
it is not exposed through the adapter registry.
"""
from __future__ import annotations

import importlib

from .base import Adapter


# name → (module, classname). Late-imported on demand.
_REGISTRY: dict[str, tuple[str, str]] = {
    # Primary: Cline via Node SDK bridge (real token/cost telemetry).
    "cline_sdk": ("tech_decomposition.adapters._cline_sdk", "ClineSDKAdapter"),
    "cline_sdk_grounded": ("tech_decomposition.adapters._cline_sdk", "ClineSDKGroundedAdapter"),
    # Primary: OpenCode terminal agent (provider-agnostic).
    "opencode": ("tech_decomposition.adapters._opencode", "OpenCodeAdapter"),
    "opencode_grounded": ("tech_decomposition.adapters._opencode", "OpenCodeGroundedAdapter"),
    # Secondary (kept, frozen): Cursor Agent CLI.
    "cursor": ("tech_decomposition.adapters._cursor", "CursorAdapter"),
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
