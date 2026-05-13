from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    gemini_api_key: str = ""
    # Optional credentials for the model registry (core/models.py). Each provider
    # is opt-in: only required if a stage's model spec names that provider.
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    openrouter_api_key: str = ""
    # Generic OpenAI-compatible endpoint (LiteLLM proxy, vLLM, Ollama, etc.).
    custom_llm_base_url: str = ""
    custom_llm_api_key: str = ""

    repos_root: Path = Path("/Users/mohammadhosseinmalek/tract-projects/aider-experiments")
    repos: Annotated[list[str], NoDecode] = Field(default_factory=lambda: [
        "traceability", "frontend", "data-cloud-functions", "data",
    ])

    gitlab_base_url: str = "https://gitlab.com"
    gitlab_projects: Annotated[dict[str, str], NoDecode] = Field(default_factory=lambda: {
        "traceability": "tract1/application/api/traceability",
        "frontend": "tract1/application/frontend",
        "data-cloud-functions": "tract1/application/data-cloud-functions",
        "data": "tract1/application/data",
    })

    jira_base_url: str = ""
    jira_email: str = ""
    jira_api_token: str = ""

    sourcebot_url: str = ""
    sourcebot_api_key: str = ""
    sourcebot_timeout_seconds: float = 15.0
    # When true, ``ask_sourcebot`` uses only ``POST /api/chat/blocking`` and does
    # not fall back to MCP ``ask_codebase`` if the primary endpoint 404s.
    sourcebot_disable_mcp_fallback: bool = False

    serena_url: str = ""
    serena_timeout_seconds: float = 20.0
    serena_max_results_per_query: int = 10

    enrich_model: str = "gemini-2.5-flash"
    decompose_model: str = "gemini-2.5-pro"

    retrieval_max_hits: int = 40
    retrieval_snippet_lines: int = 8
    decompose_max_context_chars: int = 80_000

    output_dir: Path = Path("./outputs")

    # Default for ``tech-decomposition --ask``:
    #   auto       = try Sourcebot /api/chat/blocking, MCP on 404, else local agent.
    #   sourcebot  = require Sourcebot (no local agent fallback).
    #   local      = local Pydantic AI agent only.
    ask_via: Literal["auto", "sourcebot", "local"] = "auto"

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
