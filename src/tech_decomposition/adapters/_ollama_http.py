"""Shared Ollama HTTP helpers for ``sourcebot_ollama*`` adapters.

Supports:
  - Local daemon (``http://localhost:11434``)
  - Docker host bridge (``host.docker.internal`` → ``localhost`` on the Python host)
  - Ollama Cloud direct API (``https://ollama.com`` + ``OLLAMA_API_KEY``)
"""
from __future__ import annotations

from typing import Any

import httpx

from ..config import Settings


def ollama_api_root(settings: Settings) -> str:
    """Base URL without ``/v1`` or ``/api/chat`` suffix."""
    url = (settings.ollama_base_url or "").strip()
    if not url:
        return ""
    if url.startswith("https://") or "ollama.com" in url:
        return url.rstrip("/").removesuffix("/v1")
    return url.replace("host.docker.internal", "localhost").rstrip("/").removesuffix("/v1")


def ollama_request_headers(settings: Settings) -> dict[str, str]:
    key = (getattr(settings, "ollama_api_key", None) or "").strip()
    if key:
        return {"Authorization": f"Bearer {key}"}
    return {}


def ollama_is_cloud(settings: Settings) -> bool:
    root = ollama_api_root(settings)
    return root.startswith("https://") or "ollama.com" in root


async def ollama_chat(
    settings: Settings,
    *,
    system: str,
    user: str,
    num_ctx: int = 32768,
    temperature: float = 0.2,
) -> dict[str, Any]:
    """POST ``/api/chat`` and return content + token counts."""
    root = ollama_api_root(settings)
    if not root:
        raise RuntimeError("OLLAMA_BASE_URL not set")
    if not (settings.ollama_model or "").strip():
        raise RuntimeError("OLLAMA_MODEL not set")
    if ollama_is_cloud(settings) and not ollama_request_headers(settings):
        raise RuntimeError(
            "OLLAMA_API_KEY required when OLLAMA_BASE_URL points at Ollama Cloud (https://ollama.com)"
        )

    url = root + "/api/chat"
    body = {
        "model": settings.ollama_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {"temperature": temperature, "num_ctx": num_ctx},
    }
    timeout = float(settings.agent_node_timeout_seconds or 600)
    async with httpx.AsyncClient(timeout=timeout) as c:
        r = await c.post(url, json=body, headers=ollama_request_headers(settings))
    if r.status_code >= 400:
        raise RuntimeError(f"ollama /api/chat -> {r.status_code}: {r.text[:500]}")
    data = r.json()
    msg = data.get("message") or {}
    return {
        "content": msg.get("content", ""),
        "prompt_eval_count": data.get("prompt_eval_count"),
        "eval_count": data.get("eval_count"),
        "total_duration": data.get("total_duration"),
    }


def ollama_health_check(settings: Settings) -> dict:
    """Sync probe for adapter ``health()``."""
    if not (settings.ollama_base_url or "").strip():
        return {"ok": False, "reason": "OLLAMA_BASE_URL not set"}
    if not (settings.ollama_model or "").strip():
        return {"ok": False, "reason": "OLLAMA_MODEL not set"}
    if ollama_is_cloud(settings) and not ollama_request_headers(settings):
        return {"ok": False, "reason": "OLLAMA_API_KEY required for Ollama Cloud (https://ollama.com)"}
    root = ollama_api_root(settings)
    try:
        with httpx.Client(timeout=10.0) as c:
            r = c.get(root + "/api/tags", headers=ollama_request_headers(settings))
        if r.status_code >= 400:
            return {"ok": False, "reason": f"ollama /api/tags -> {r.status_code}"}
    except httpx.HTTPError as e:
        return {"ok": False, "reason": f"ollama unreachable ({root}): {e}"}
    out: dict = {"ok": True}
    if ollama_is_cloud(settings):
        out["mode"] = "ollama_cloud"
    return out
