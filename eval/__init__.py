"""Adapter bake-off — evaluation harness for tech-decomposition adapters.

Intentionally lives outside ``src/tech_decomposition/``. The harness reads
the same FastAPI surface external clients would and writes its results to
its own ``outputs/`` directory. Nothing under here is imported by the app
at runtime.
"""
