"""Pluggable agent backends behind a uniform interface.

Each adapter implements some subset of (ask, decompose, implement). The
registry exposes them by name so the API can route requests to any installed
backend. See ``base.Adapter`` for the contract and ``registry`` for discovery.
"""
from .base import (
    Adapter,
    AdapterAskInput,
    AdapterAskResult,
    AdapterDecomposeInput,
    AdapterDecomposeResult,
    AdapterImplementInput,
    AdapterImplementResult,
    AdapterMetrics,
    Capability,
    NotSupported,
)
from .registry import get_adapter, list_adapters

__all__ = [
    "Adapter",
    "AdapterAskInput",
    "AdapterAskResult",
    "AdapterDecomposeInput",
    "AdapterDecomposeResult",
    "AdapterImplementInput",
    "AdapterImplementResult",
    "AdapterMetrics",
    "Capability",
    "NotSupported",
    "get_adapter",
    "list_adapters",
]
