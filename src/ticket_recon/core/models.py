"""Model registry: ``provider:name`` string → pydantic-ai Model instance.

Supports Gemini, Anthropic, OpenAI natively, and any OpenAI-compatible
provider (notably OpenRouter and LiteLLM proxies) via base_url override.
Adding a new provider = one new branch in ``get_model`` and one row in
``PRICING_PER_M_USD``. Imports are lazy so missing optional libs don't
break startup.

Examples
--------
    get_model("gemini:gemini-2.5-pro")
    get_model("anthropic:claude-sonnet-4-5")
    get_model("openai:gpt-4o")
    get_model("openrouter:anthropic/claude-opus-4-5")
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from ..config import Settings


class UnknownProviderError(ValueError):
    pass


class MissingApiKeyError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------

# Per-million-token prices in USD. Keys are matched against the *full*
# "provider:model" string by substring, so "gemini-2.5-pro" matches
# "gemini:gemini-2.5-pro" and "openrouter:google/gemini-2.5-pro".
PRICING_PER_M_USD: dict[str, dict[str, float]] = {
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    "claude-opus-4": {"input": 15.00, "output": 75.00},
    "claude-sonnet-4": {"input": 3.00, "output": 15.00},
    "claude-haiku-4": {"input": 1.00, "output": 5.00},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}


def _price_for(spec: str) -> dict[str, float] | None:
    for key, prices in PRICING_PER_M_USD.items():
        if key in spec:
            return prices
    return None


def estimate_cost_usd(spec: str, input_tokens: int, output_tokens: int) -> float | None:
    """Estimate cost given a ``provider:model`` spec or a bare model name."""
    p = _price_for(spec)
    if not p:
        return None
    return round(
        (input_tokens / 1_000_000) * p["input"]
        + (output_tokens / 1_000_000) * p["output"],
        6,
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@dataclass
class ModelSpec:
    """Parsed form of a ``provider:name`` config string."""

    provider: str
    name: str

    @classmethod
    def parse(cls, spec: str) -> "ModelSpec":
        spec = spec.strip()
        if not spec:
            raise ValueError("empty model spec")
        if ":" in spec:
            provider, name = spec.split(":", 1)
            return cls(provider=provider.strip().lower(), name=name.strip())
        # Bare names — infer provider from prefix for backwards compat.
        if spec.startswith("gemini"):
            return cls(provider="gemini", name=spec)
        if spec.startswith("claude"):
            return cls(provider="anthropic", name=spec)
        if spec.startswith("gpt") or spec.startswith("o1") or spec.startswith("o3"):
            return cls(provider="openai", name=spec)
        raise UnknownProviderError(
            f"cannot infer provider from {spec!r}; use 'provider:name' form"
        )


def get_model(spec: str, settings: Settings):
    """Return a pydantic-ai Model instance for the given spec.

    Raises ``UnknownProviderError`` for unknown providers and
    ``MissingApiKeyError`` if the relevant credential isn't set.
    """
    ms = ModelSpec.parse(spec)

    if ms.provider == "gemini":
        from pydantic_ai.models.gemini import GeminiModel

        if not settings.gemini_api_key:
            raise MissingApiKeyError("GEMINI_API_KEY required for provider 'gemini'")
        os.environ["GEMINI_API_KEY"] = settings.gemini_api_key
        return GeminiModel(ms.name)

    if ms.provider == "anthropic":
        from pydantic_ai.models.anthropic import AnthropicModel

        if not settings.anthropic_api_key:
            raise MissingApiKeyError("ANTHROPIC_API_KEY required for provider 'anthropic'")
        os.environ["ANTHROPIC_API_KEY"] = settings.anthropic_api_key
        return AnthropicModel(ms.name)

    if ms.provider == "openai":
        from pydantic_ai.models.openai import OpenAIModel

        if not settings.openai_api_key:
            raise MissingApiKeyError("OPENAI_API_KEY required for provider 'openai'")
        os.environ["OPENAI_API_KEY"] = settings.openai_api_key
        return OpenAIModel(ms.name)

    if ms.provider == "openrouter":
        # OpenRouter speaks OpenAI-compatible API.
        from pydantic_ai.models.openai import OpenAIModel
        from pydantic_ai.providers.openai import OpenAIProvider

        if not settings.openrouter_api_key:
            raise MissingApiKeyError("OPENROUTER_API_KEY required for provider 'openrouter'")
        provider = OpenAIProvider(
            api_key=settings.openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
        )
        return OpenAIModel(ms.name, provider=provider)

    if ms.provider in ("openai-compat", "custom"):
        # Generic OpenAI-compatible endpoint (LiteLLM proxy, vLLM, Ollama, etc.).
        from pydantic_ai.models.openai import OpenAIModel
        from pydantic_ai.providers.openai import OpenAIProvider

        base_url = settings.custom_llm_base_url
        api_key = settings.custom_llm_api_key or "sk-noop"
        if not base_url:
            raise MissingApiKeyError("CUSTOM_LLM_BASE_URL required for provider 'openai-compat'")
        provider = OpenAIProvider(api_key=api_key, base_url=base_url)
        return OpenAIModel(ms.name, provider=provider)

    raise UnknownProviderError(f"unknown provider: {ms.provider!r}")
