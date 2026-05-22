"""OpenAI-compatible chat/completions for RAG adapters (OpenRouter, Groq, etc.)."""
from __future__ import annotations

from typing import Any

import httpx

from ..config import Settings


def resolve_rag_api(settings: Settings) -> tuple[str, str, str]:
    """Return ``(base_url, api_key, model)`` for hosted free/cheap generation.

    Uses ``RAG_*`` overrides, then ``CUSTOM_LLM_*``, then provider keys.
    """
    base = (settings.rag_openai_base_url or settings.custom_llm_base_url or "").strip()
    model = (settings.rag_openai_model or settings.custom_llm_model or "").strip()
    key = (settings.rag_openai_api_key or "").strip()
    if not key:
        key = (settings.openrouter_api_key or "").strip()
    if not key:
        key = (settings.custom_llm_api_key or settings.openai_api_key or "").strip()
    return base.rstrip("/"), key, model


def openai_is_configured(settings: Settings) -> bool:
    base, key, model = resolve_rag_api(settings)
    return bool(base and key and model)


async def openai_chat(
    settings: Settings,
    *,
    system: str,
    user: str,
    temperature: float = 0.2,
    max_tokens: int = 4096,
) -> dict[str, Any]:
    base, api_key, model = resolve_rag_api(settings)
    if not base:
        raise RuntimeError("RAG_OPENAI_BASE_URL or CUSTOM_LLM_BASE_URL not set")
    if not api_key:
        raise RuntimeError(
            "Set RAG_OPENAI_API_KEY, OPENROUTER_API_KEY, CUSTOM_LLM_API_KEY, or OPENAI_API_KEY"
        )
    if not model:
        raise RuntimeError("RAG_OPENAI_MODEL or CUSTOM_LLM_MODEL not set")

    url = base + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if "openrouter.ai" in base:
        headers["HTTP-Referer"] = "https://github.com/tract/tech-decomposition"
        headers["X-Title"] = "tech-decomposition"

    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    timeout = float(settings.agent_node_timeout_seconds or 600)
    async with httpx.AsyncClient(timeout=timeout) as c:
        r = await c.post(url, json=body, headers=headers)
    if r.status_code >= 400:
        raise RuntimeError(f"chat/completions -> {r.status_code}: {r.text[:800]}")
    data = r.json()
    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    usage = data.get("usage") or {}
    return {
        "content": msg.get("content", "") or "",
        "prompt_eval_count": usage.get("prompt_tokens"),
        "eval_count": usage.get("completion_tokens"),
        "model": data.get("model") or model,
    }


def openai_health_check(settings: Settings) -> dict:
    base, api_key, model = resolve_rag_api(settings)
    if not base:
        return {"ok": False, "reason": "RAG_OPENAI_BASE_URL or CUSTOM_LLM_BASE_URL not set"}
    if not api_key:
        return {"ok": False, "reason": "no API key (OPENROUTER/CUSTOM/OPENAI/RAG_OPENAI)"}
    if not model:
        return {"ok": False, "reason": "RAG_OPENAI_MODEL or CUSTOM_LLM_MODEL not set"}
    try:
        with httpx.Client(timeout=15.0) as c:
            r = c.get(
                base + "/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
        if r.status_code >= 400:
            return {"ok": False, "reason": f"GET /models -> {r.status_code}"}
    except httpx.HTTPError as e:
        return {"ok": False, "reason": f"API unreachable ({base}): {e}"}
    provider = "openrouter" if "openrouter" in base else "openai_compat"
    return {"ok": True, "mode": provider, "model": model, "base_url": base}
