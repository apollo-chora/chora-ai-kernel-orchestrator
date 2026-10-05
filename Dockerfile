# Multi-stage build for chora-ai-kernel-orchestrator (Python LangGraph).
# Cloud-neutral: NATS JetStream + MinIO + Postgres, no cloud-provider deps.

FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS build

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Project metadata first for layer caching.
COPY pyproject.toml uv.lock ./

# Install the runtime deps (no dev, no project) into the venv. The project
# itself is imported from /app/src via PYTHONPATH at runtime (NOT installed
# into site-packages) so the prompt-registry testdata/ path resolution keeps
# working.
RUN uv sync --no-dev --no-install-project

# -----------------------------------------------------------------------------
# Runtime
# -----------------------------------------------------------------------------
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app

# The venv with the installed runtime deps.
COPY --from=build /app/.venv /app/.venv

# Sources + runtime config (prompt baselines, agent configs, guardrail map).
COPY src /app/src
COPY config /app/config
COPY testdata /app/testdata

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080 \
    SERVICE_NAME=chora-ai-kernel-orchestrator

EXPOSE 8080

CMD ["uvicorn", "chora_ai_kernel_orchestrator.main:app", "--host", "0.0.0.0", "--port", "8080"]
