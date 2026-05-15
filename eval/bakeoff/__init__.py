"""Adapter bake-off harness.

A separate, self-contained tool for comparing tech-decomposition's agent
adapters head-to-head. Lives outside ``src/tech_decomposition/`` on purpose —
this is evaluation infrastructure, not part of the runtime app.

Run via:
    uv run python -m eval.bakeoff.cli run --job ask --adapters cline_sdk,opencode
    uv run python -m eval.bakeoff.cli report runs/eval-<timestamp>

See ``eval/README.md`` for the full usage guide.
"""
__version__ = "0.1.0"
