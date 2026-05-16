from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    gemini_api_key: str = ""
    # Optional credentials for the model registry (core/llm_registry.py). Each provider
    # is opt-in: only required if a stage's model spec names that provider.
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    openrouter_api_key: str = ""
    # Generic OpenAI-compatible endpoint (LiteLLM proxy, vLLM, Ollama, etc.).
    custom_llm_base_url: str = ""
    custom_llm_api_key: str = ""

    repos_root: Path = Path("./repos")
    repos: Annotated[list[str], NoDecode] = Field(default_factory=lambda: [
        "backend-api", "frontend-webapp",
    ])

    gitlab_base_url: str = "https://gitlab.com"
    gitlab_projects: Annotated[dict[str, str], NoDecode] = Field(default_factory=lambda: {
        "backend-api": "your-company/backend-api",
        "frontend-webapp": "your-company/frontend-webapp",
    })

    sourcebot_url: str = ""
    sourcebot_api_key: str = ""
    # Base URL for Sourcebot as seen by agent-node when it proxies ``/api/chat/blocking``.
    # Empty ⇒ use ``sourcebot_url``. Set ``http://127.0.0.1:<port>`` only when FastAPI runs in
    # Docker and agent-node runs on the host. When both services use compose.adapters.yaml,
    # Compose sets ``http://sourcebot:3000``.
    sourcebot_url_for_agent_node: str = ""
    # Used for Sourcebot HTTP (search + agent-node `/adapters/sourcebot/ask` → blocking chat).
    # Low values produce fast 502 timeouts from agent-node during long reasoning turns.
    sourcebot_timeout_seconds: float = 300.0
    # Fail closed: when true, grounded retrieval requires Sourcebot to be
    # configured and reachable; otherwise the pipeline raises.
    grounding_require_sourcebot: bool = True
    # When true, ``ask_sourcebot`` uses only ``POST /api/chat/blocking`` and does
    # not fall back to MCP ``ask_codebase`` if the primary endpoint 404s.
    sourcebot_disable_mcp_fallback: bool = False

    enrich_model: str = "gemini-2.5-pro"
    decompose_model: str = "gemini-2.5-pro"

    retrieval_max_hits: int = 40
    retrieval_snippet_lines: int = 8
    decompose_max_context_chars: int = 80_000
    # Optional open-source cross-encoder reranker for grounding snippets.
    grounding_reranker_model: str = ""
    grounding_reranker_max_candidates: int = 48

    output_dir: Path = Path("./outputs")

    # Default when routing ``ask`` inside the Python stack (see clients/sourcebot):
    #   auto       = try Sourcebot /api/chat/blocking, MCP on 404, else local agent.
    #   sourcebot  = require Sourcebot (no local agent fallback).
    #   local      = local Pydantic AI agent only.
    ask_via: Literal["auto", "sourcebot", "local"] = "auto"

    # ----- adapters / agent-node ------------------------------------------------
    gitlab_token: str = ""
    git_author_name: str = ""
    git_author_email: str = ""
    enabled_adapters: Annotated[list[str], NoDecode] = Field(default_factory=list)

    # Node service (services/agent-node) — Cursor, Cline, Claude Code, Gemini, OpenAI Agents,
    # OpenCode, Sourcebot proxy.
    agent_node_url: str = "http://127.0.0.1:13100"
    agent_node_timeout_seconds: float = 600.0

    # Cursor Cloud API key (also used by @cursor/sdk local runs).
    cursor_api_key: str = ""
    cursor_sdk_model: str = "composer-2"
    # Claude Code (@anthropic-ai/claude-agent-sdk) default model when using ``claude_code`` adapter.
    claude_code_model: str = "claude-sonnet-4-5"

    # Gemini via agent-node (`@google/genai` — https://googleapis.github.io/js-genai/).
    gemini_sdk_model: str = "gemini-2.5-pro"
    # If empty, decompose uses ``decompose_model``.
    gemini_sdk_decompose_model: str = ""
    gemini_sdk_timeout_seconds: float = 600.0

    # OpenAI Agents SDK via agent-node (`@openai/agents` — https://github.com/openai/openai-agents-js).
    openai_agents_sdk_model: str = "gpt-4.1"
    openai_agents_sdk_decompose_model: str = ""
    openai_agents_sdk_timeout_seconds: float = 600.0

    # OpenCode via agent-node (`@opencode-ai/sdk` — https://opencode.ai/docs/sdk/).
    opencode_sdk_model: str = "anthropic/claude-sonnet-4-20250514"
    # When set (or OPENCODE_BASE_URL on agent-node), use client-only mode against that server URL.
    opencode_sdk_base_url: str = ""
    opencode_sdk_structured_retry_count: int = 2

    @field_validator("enabled_adapters", mode="before")
    @classmethod
    def _split_enabled(cls, v):
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return v

    @field_validator("repos", mode="before")
    @classmethod
    def _split_repos(cls, v):
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return v

    @field_validator("gitlab_projects", mode="before")
    @classmethod
    def _parse_projects(cls, v):
        if isinstance(v, dict):
            return v
        if not v:
            return {}
        out: dict[str, str] = {}
        for entry in str(v).split(","):
            entry = entry.strip()
            if not entry:
                continue
            if "=" not in entry:
                raise ValueError(f"GITLAB_PROJECTS entry must be repo=path, got: {entry!r}")
            name, path = entry.split("=", 1)
            out[name.strip()] = path.strip()
        return out

    def repo_path(self, name: str) -> Path:
        return self.repos_root / name


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
