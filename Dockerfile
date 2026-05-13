# syntax=docker/dockerfile:1.7
# Two-stage build: install deps with uv into a venv, then copy into a slim
# runtime image. Keeps the final image small (~150 MB) and reproducible from
# uv.lock.

FROM python:3.13-slim AS builder

# Install uv (fast Python package installer).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy only the files needed to resolve deps, so layer caches well.
COPY pyproject.toml uv.lock ./
COPY src ./src

# Install into a project-local venv that we'll copy out later.
ENV UV_PROJECT_ENVIRONMENT=/app/.venv UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev


FROM python:3.13-slim AS runtime

# ripgrep is required by RipgrepRetriever; git is needed if you ever want
# the local agent (experimental) to clone snapshots on the fly.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ripgrep git ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for runtime — small hardening win.
RUN useradd -u 10001 -m -s /usr/sbin/nologin app
WORKDIR /app

COPY --from=builder /app /app
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Outputs (markdown answers, metrics JSONL) live under /app/outputs by
# default. Mount a volume here in compose so they survive container restarts.
RUN mkdir -p /app/outputs && chown -R app:app /app

USER app
EXPOSE 8000

# Healthcheck hits the FastAPI /health endpoint.
HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

# Default: run the FastAPI server. Override at runtime for one-shot CLI
# invocations, e.g.:
#     docker run --rm ticket-recon ticket-recon ask "..."
CMD ["uvicorn", "ticket_recon.api:app", "--host", "0.0.0.0", "--port", "8000"]
