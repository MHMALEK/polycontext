"""Live OpenRouter model catalog fetcher with in-process TTL cache.

OpenRouter exposes its full model catalog at ``GET /api/v1/models`` — ~350
entries with id, context_length, pricing (per-token, in USD), modality, etc.
The model picker in the UI uses this to show what's actually available right
now without us having to hand-maintain a list.

Caches the result for one hour to avoid hammering OpenRouter on every UI
page load.

Returned shape is normalized to match the curated catalog in
``_models_catalog.py`` (id, name, provider, in/out per-M USD, context_k,
note) so the frontend doesn't have to special-case live vs curated.
"""
from __future__ import annotations

import time
from typing import Any

import httpx


_OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
_CACHE_TTL_SECONDS = 3600  # 1 hour

# Module-level cache: (timestamp, models_list)
_cache: tuple[float, list[dict]] | None = None


def _normalize(raw: dict) -> dict:
    """Map OpenRouter's raw shape to our UI-friendly shape.

    Note: the OpenRouter ``id`` (e.g. ``deepseek/deepseek-v3.2``) doesn't
    include the ``openrouter/`` provider prefix that the OpenCode SDK
    expects. We add it here so the id is directly usable by callers.
    """
    pricing = raw.get("pricing") or {}
    out: dict[str, Any] = {
        # The id callers will pass to the opencode adapter is "openrouter/<id>".
        # OpenRouter's own id is the bare model — we prefix it for convenience.
        "id": f"openrouter/{raw.get('id', '')}",
        "name": raw.get("name") or raw.get("id", ""),
        "provider": "openrouter",
    }
    # Pricing in OpenRouter's API is per-token USD as strings. Convert to
    # per-million for display (the cents per million is much more legible).
    try:
        prompt = float(pricing.get("prompt", 0)) * 1_000_000
        completion = float(pricing.get("completion", 0)) * 1_000_000
        if prompt > 0:
            out["in_per_m_usd"] = round(prompt, 4)
        if completion > 0:
            out["out_per_m_usd"] = round(completion, 4)
    except (TypeError, ValueError):
        pass
    ctx = raw.get("context_length")
    if isinstance(ctx, (int, float)) and ctx > 0:
        out["context_k"] = round(ctx / 1000)
    desc = (raw.get("description") or "").strip()
    if desc:
        # Truncate hard so the UI tooltip stays sane.
        out["note"] = desc[:200] + ("…" if len(desc) > 200 else "")
    return out


def fetch_openrouter_models(*, force_refresh: bool = False) -> list[dict]:
    """Return OpenRouter's live model catalog, cached for 1h.

    On network failure returns an empty list and the cache is left untouched.
    """
    global _cache
    now = time.monotonic()
    if not force_refresh and _cache is not None:
        ts, models = _cache
        if (now - ts) < _CACHE_TTL_SECONDS:
            return models
    try:
        with httpx.Client(timeout=httpx.Timeout(15.0, connect=5.0)) as client:
            r = client.get(_OPENROUTER_MODELS_URL)
            if r.status_code != 200:
                return _cache[1] if _cache else []
            raw_list = (r.json() or {}).get("data") or []
    except (httpx.HTTPError, ValueError):
        return _cache[1] if _cache else []
    models = [_normalize(m) for m in raw_list if isinstance(m, dict) and m.get("id")]
    # Sort by name for stable UI ordering.
    models.sort(key=lambda m: (m.get("name") or "").lower())
    _cache = (now, models)
    return models
