"""Extract a JSON object from noisy agent output (shared by adapters).

We can't trust the model to emit a clean JSON-only response — it routinely
adds markdown fences, prose, or multiple objects. This module finds the
longest *brace-balanced, string-aware* JSON object in the text and returns
it as a dict.

History: a regex-only extractor used to live here. It worked for shallow
objects but mis-matched on the ``Decomposition`` shape because that has
nested objects inside arrays (``"subtasks": [{...}, {...}]``), and the
regex could only handle one level of ``{}`` nesting. When the outer
match failed, ``findall`` returned each subtask as its own top-level
object and the "largest" picked was a single Subtask — surfaced
upstream as ``ValidationError: 4 validation errors for Decomposition``
in adapters or as a silent fall-through to LLM repair in the
``decomposition_structurer``.

Verified by capturing real Gemini decompose output (8.4 kB Decomposition
with 6 subtasks): the old regex returned a single ``{title, description,
repo, ...}`` dict; the new walker returns the full ``{query, overview,
affected_repos, subtasks, ...}``.
"""
from __future__ import annotations

import json
import re


def _find_balanced_json_objects(text: str) -> list[str]:
    """Walk ``text`` and yield every top-level ``{...}`` whose braces and
    string literals balance. Handles nested objects/arrays and escaped
    quotes inside strings.

    Returns candidates in document order (not sorted by size).
    """
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        depth = 0
        in_str = False
        escape = False
        start = i
        while i < n:
            ch = text[i]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        out.append(text[start : i + 1])
                        i += 1
                        break
            i += 1
        else:
            # Hit end of text without closing — abandon this candidate.
            break
    return out


def extract_json(text: str) -> dict:
    """Pull the largest top-level JSON object out of a possibly-noisy answer.

    Strips ``` markdown fences first, then walks the text for brace-balanced
    candidates, parses each, and returns the largest one that's a dict.
    """
    stripped = re.sub(r"```(?:json)?\s*", "", text)
    stripped = stripped.replace("```", "")
    candidates = _find_balanced_json_objects(stripped)
    candidates.sort(key=len, reverse=True)
    for c in candidates:
        try:
            obj = json.loads(c)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise ValueError("no JSON object found in CLI output")
