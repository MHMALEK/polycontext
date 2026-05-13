# syntax=docker/dockerfile:1.7
# Three-stage build:
#   1. ui-builder    — npm install + vite build → web/dist
#   2. py-builder    — uv install Python deps into a venv
#   3. runtime       — slim image with the venv + UI build + ripgrep + git
#
# Result: one image, ~1.1 GB, with everything needed to serve the API + UI.

FROM node:22-alpine AS ui-builder
WORKDIR /web
# Install JS deps first so this layer caches well across UI source changes.
COPY web/package.json web/package-lock.json* ./
RUN npm install --no-audit --no-fund
COPY web ./
RUN npm run build


FROM python:3.13-slim AS py-builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY src ./src
ENV UV_PROJECT_ENVIRONMENT=/app/.venv UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev


FROM python:3.13-slim AS runtime
# ripgrep for RipgrepRetriever; git for the experimental local agent.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ripgrep git ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -u 10001 -m -s /usr/sbin/nologin app
WORKDIR /app

# Python app + venv from py-builder.
COPY --from=py-builder /app /app
# UI bundle from ui-builder; FastAPI mounts /app/web/dist at /ui at startup.
COPY --from=ui-builder /web/dist /app/web/dist

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UI_DIST_DIR=/app/web/dist

RUN mkdir -p /app/outputs && chown -R app:app /app

USER app
EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

# Default: run the FastAPI server. The UI is served from this same process at
# /ui. Override at runtime for one-shot CLI invocations, e.g.:
#     docker run --rm ticket-recon ticket-recon ask "..."
CMD ["uvicorn", "ticket_recon.api:app", "--host", "0.0.0.0", "--port", "8000"]
