"""Adapter API client.

Two backends, same shape:

  * **TestClient backend** (default) — instantiates the FastAPI app in-process
    via ``fastapi.testclient.TestClient``. No server required. Fast.
  * **HTTP backend**                  — hits a running ``make dev`` instance
    over httpx. Use this when you want to exercise the real network stack
    (Docker, reverse proxy, ...).

Backend selection via ``--base-url`` on the CLI:
    omitted      → TestClient
    http://...   → real HTTP

The two return identical dict shapes so the runner doesn't have to care.
"""
from __future__ import annotations

import logging
from typing import Any, Protocol

import httpx

log = logging.getLogger(__name__)


class AdapterClient(Protocol):
    async def list_adapters(self) -> list[dict[str, Any]]: ...
    async def call(self, adapter: str, job: str, body: dict[str, Any]) -> dict[str, Any]: ...
    async def aclose(self) -> None: ...


# ---------------------------------------------------------------------------
# TestClient backend — in-process, no server required
# ---------------------------------------------------------------------------


class InProcessClient:
    """Backs adapter calls via httpx + ASGI transport.

    Earlier versions of this client used ``fastapi.testclient.TestClient``,
    which is synchronous and internally bounces requests through an anyio
    portal. That portal leaked event-loop state across calls when the runner
    drove it from its own asyncio loop — successive calls would crash with
    ``RuntimeError: Event loop is closed``. ASGITransport is the documented
    fix: a fully-async transport that talks ASGI directly to the FastAPI app
    without spinning up a server or a portal.
    """

    def __init__(self):
        from tech_decomposition.api import app
        self._app = app
        transport = httpx.ASGITransport(app=app)
        # base_url is required by httpx for a relative path; the value is
        # irrelevant since ASGITransport ignores it.
        self._client = httpx.AsyncClient(transport=transport, base_url="http://testserver", timeout=900.0)

    async def list_adapters(self) -> list[dict[str, Any]]:
        resp = await self._client.get("/v1/adapters")
        resp.raise_for_status()
        return resp.json()["adapters"]

    async def call(self, adapter: str, job: str, body: dict[str, Any]) -> dict[str, Any]:
        resp = await self._client.post(f"/v1/adapters/{adapter}/{job}", json=body)
        try:
            payload = resp.json()
        except ValueError:
            payload = {"detail": resp.text}
        return _coerce_response(resp.status_code, payload)

    async def aclose(self) -> None:
        await self._client.aclose()


# ---------------------------------------------------------------------------
# Real HTTP backend — for testing against a running server
# ---------------------------------------------------------------------------


class HttpClient:
    def __init__(self, base_url: str, *, timeout: float = 900.0):
        self._client = httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=timeout)

    async def list_adapters(self) -> list[dict[str, Any]]:
        resp = await self._client.get("/v1/adapters")
        resp.raise_for_status()
        return resp.json()["adapters"]

    async def call(self, adapter: str, job: str, body: dict[str, Any]) -> dict[str, Any]:
        resp = await self._client.post(f"/v1/adapters/{adapter}/{job}", json=body)
        try:
            payload = resp.json()
        except ValueError:
            payload = {"detail": resp.text}
        return _coerce_response(resp.status_code, payload)

    async def aclose(self) -> None:
        await self._client.aclose()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _coerce_response(status: int, payload: Any) -> dict[str, Any]:
    """Normalize success/error into one shape the runner can branch on.

    Success:
        {"ok": True, "status": 200, "run_id": "...", "result": {...}}
    Error (any 4xx/5xx):
        {"ok": False, "status": 502, "error": "...detail..."}
    """
    if status >= 400:
        detail = ""
        if isinstance(payload, dict):
            detail = str(payload.get("detail") or payload)
        return {"ok": False, "status": status, "error": detail}
    if isinstance(payload, dict):
        return {
            "ok": True,
            "status": status,
            "run_id": payload.get("run_id"),
            "result": payload.get("result", payload),
        }
    return {"ok": True, "status": status, "result": payload}


def build_client(base_url: str | None) -> AdapterClient:
    """Pick a backend by inspecting ``base_url`` (None → TestClient)."""
    if base_url:
        return HttpClient(base_url)
    return InProcessClient()
