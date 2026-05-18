"""Minimal MCP streamable-HTTP client for Serena (https://github.com/oraios/serena).

Calls Serena tools directly over JSON-RPC so the grounding pipeline can use
LSP-backed semantic search alongside Sourcebot. Specifically targeted at
adapters that can't take MCP themselves (gemini direct, sourcebot's chat) —
adapters that DO support MCP register Serena per-request in agent-node and
should not call this.

Why a hand-rolled client instead of the ``mcp`` Python package: we only need
``search_for_pattern`` here, and the official client adds session-management
boilerplate around stateful HTTP that this single-shot use doesn't need.
"""
from __future__ import annotations

import json
from typing import Any

import httpx


class SerenaError(RuntimeError):
    """Any non-recoverable error from Serena (bad URL, JSON-RPC error, timeout)."""


def _parse_sse_or_json(body: str) -> dict[str, Any]:
    """Serena's streamable-HTTP transport returns either plain JSON or SSE-style
    ``data: { ... }`` frames depending on whether the tool result streamed."""
    text = body.strip()
    if text.startswith("event:") or text.startswith("data:"):
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("data:"):
                payload = line[len("data:"):].strip()
                if payload and payload != "[DONE]":
                    return json.loads(payload)
        return {}
    return json.loads(text)


class SerenaMcpClient:
    """One-call client: initialize, send tool call, close.

    Sessions in Serena's streamable-HTTP transport are stateful (Mcp-Session-Id
    header tracks them) but we only need single-tool-call semantics for grounding,
    so we initialize fresh per call. The cost is ~50ms per init — acceptable.
    """

    def __init__(self, *, url: str, api_key: str = "", timeout: float = 20.0):
        if not url.strip():
            raise SerenaError("serena url is empty")
        self.url = url.strip()
        self.timeout = timeout
        self._headers_base: dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if api_key.strip():
            self._headers_base["Authorization"] = f"Bearer {api_key.strip()}"

    async def call(self, tool: str, args: dict[str, Any]) -> str:
        """Call ``tool`` with ``args`` and return its text content (joined)."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            session_id = await self._init(client)
            return await self._tool_call(client, session_id, tool, args)

    async def _post(
        self, client: httpx.AsyncClient, payload: dict[str, Any], *, session_id: str | None,
    ) -> tuple[dict[str, Any], str | None]:
        headers = dict(self._headers_base)
        if session_id:
            headers["mcp-session-id"] = session_id
        r = await client.post(self.url, json=payload, headers=headers)
        if r.status_code >= 400:
            raise SerenaError(f"serena HTTP {r.status_code}: {r.text[:300]}")
        sid = r.headers.get("mcp-session-id") or session_id
        body = r.text
        if not body:
            return ({}, sid)
        return (_parse_sse_or_json(body), sid)

    async def _init(self, client: httpx.AsyncClient) -> str:
        out, sid = await self._post(
            client,
            {
                "jsonrpc": "2.0", "id": 1, "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "tech-decomposition-grounding", "version": "1"},
                },
            },
            session_id=None,
        )
        if "error" in out:
            raise SerenaError(f"serena initialize error: {out['error']}")
        if not sid:
            raise SerenaError("serena did not return a session id on initialize")
        # The MCP spec requires the client to send `notifications/initialized`
        # before any tool calls; Serena enforces this.
        await self._post(
            client,
            {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}},
            session_id=sid,
        )
        return sid

    async def _tool_call(
        self, client: httpx.AsyncClient, session_id: str, tool: str, args: dict[str, Any],
    ) -> str:
        out, _ = await self._post(
            client,
            {
                "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                "params": {"name": tool, "arguments": args},
            },
            session_id=session_id,
        )
        if "error" in out:
            raise SerenaError(f"serena tools/call error ({tool}): {out['error']}")
        result = out.get("result") or {}
        if result.get("isError"):
            content = result.get("content") or []
            msg = content[0].get("text", "") if content else ""
            raise SerenaError(f"serena tool {tool!r} returned isError: {msg[:300]}")
        parts: list[str] = []
        for block in result.get("content") or []:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts)
