"""Shared fixtures for adapter tests."""
from __future__ import annotations

from typing import Any

import httpx
import pytest

from tech_decomposition.config import Settings


@pytest.fixture
def repos_root(tmp_path):
    (tmp_path / "backend-api").mkdir()
    return tmp_path


def settings(**overrides) -> Settings:
    return Settings(**overrides)


@pytest.fixture
def ollama_cline_settings(repos_root) -> Settings:
    return settings(
        agent_node_url="http://agent-node.test:13100",
        ollama_base_url="http://host.docker.internal:11434/v1",
        ollama_model="cline-qwen:7b",
        cline_sdk_use_ollama=True,
        repos_root=repos_root,
    )


@pytest.fixture
def sourcebot_ollama_settings(repos_root) -> Settings:
    return settings(
        sourcebot_url="http://sourcebot.test:3000",
        sourcebot_api_key="test-key",
        ollama_base_url="http://host.docker.internal:11434/v1",
        ollama_model="qwen2.5:7b",
        repos_root=repos_root,
    )


class _MockHttpResponse:
    def __init__(self, *, status_code: int = 200, json_data: dict | None = None, text: str = ""):
        self.status_code = status_code
        self._json = json_data or {}
        self.text = text or str(self._json)

    def json(self) -> dict:
        return self._json


class MockAsyncHttpClient:
    """Records POST bodies and returns a configured JSON payload."""

    posts: list[dict[str, Any]]
    response_json: dict[str, Any]
    status_code: int

    def __init__(
        self,
        *,
        response_json: dict | None = None,
        status_code: int = 200,
        timeout: float | None = None,
    ):
        self.timeout = timeout
        self.response_json = response_json or {"ok": True, "answer": "mock adapter answer"}
        self.status_code = status_code
        self.posts = []

    async def __aenter__(self) -> MockAsyncHttpClient:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def post(self, url: str, json: dict | None = None) -> _MockHttpResponse:
        self.posts.append({"url": url, "json": json})
        return _MockHttpResponse(status_code=self.status_code, json_data=self.response_json)


class MockSyncHttpClient:
    """For adapter ``health()`` sync GET probes."""

    gets: list[str]
    get_status: int
    get_json: dict

    def __init__(self, *, get_status: int = 200, get_json: dict | None = None, timeout: float = 3.0):
        self.timeout = timeout
        self.get_status = get_status
        self.get_json = get_json or {"ok": True}
        self.gets = []

    def __enter__(self) -> MockSyncHttpClient:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def get(self, url: str) -> _MockHttpResponse:
        self.gets.append(url)
        return _MockHttpResponse(status_code=self.get_status, json_data=self.get_json)


@pytest.fixture
def patch_async_http(monkeypatch):
    """Replace httpx.AsyncClient in a target module with MockAsyncHttpClient."""

    def _patch(module_path: str) -> MockAsyncHttpClient:
        client = MockAsyncHttpClient()

        def factory(*_args, **_kwargs):
            return client

        monkeypatch.setattr(f"{module_path}.httpx.AsyncClient", factory)
        return client

    return _patch


@pytest.fixture
def patch_sync_http(monkeypatch):
    def _patch(module_path: str, **kwargs) -> MockSyncHttpClient:
        client = MockSyncHttpClient(**kwargs)

        def factory(*_args, **_kwargs):
            return client

        monkeypatch.setattr(f"{module_path}.httpx.Client", factory)
        return client

    return _patch
