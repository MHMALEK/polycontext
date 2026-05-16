"""Extract a JSON object from noisy agent output (shared by adapters)."""
from __future__ import annotations

import json
import re

_JSON_OBJECT_RE = re.compile(r"\{(?:[^{}]|\{[^{}]*\})*\}", re.DOTALL)


def extract_json(text: str) -> dict:
    """Pull the largest top-level JSON object out of a possibly-noisy answer."""
    stripped = re.sub(r"```(?:json)?\s*", "", text)
    stripped = stripped.replace("```", "")
    candidates = _JSON_OBJECT_RE.findall(stripped)
    candidates.sort(key=len, reverse=True)
    for c in candidates:
        try:
            obj = json.loads(c)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise ValueError("no JSON object found in CLI output")
