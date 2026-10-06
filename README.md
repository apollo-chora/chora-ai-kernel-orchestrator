# chora-ai-kernel-orchestrator

## About

`chora-ai-kernel-orchestrator` is a Python service that coordinates Chora AI agent workflows with FastAPI, LangGraph, NATS JetStream, PostgreSQL, and OpenTelemetry. It runs the kennel runtime that parks and resumes agent work, dispatches enabled crew lanes, drains the durable outbox, and exposes the HTTP routes used for health checks, Growth-Edge review resume, and the prompt catalogue.

## Quick start

Requires Python 3.13 and `uv`. Dependencies and the project definition are in `pyproject.toml` and `uv.lock`.

Clone the repository and install the project with its development dependencies:

```sh
git clone https://github.com/apollo-chora/chora-ai-kernel-orchestrator.git
cd chora-ai-kernel-orchestrator

uv sync --all-extras
```

For a local HTTP-only process, no database or NATS configuration is required. The application can be imported and started with:

```sh
uv run uvicorn chora_ai_kernel_orchestrator.main:app --host 0.0.0.0 --port 8080
```

With the default configuration, the crew lanes remain disabled. Check the service:

```sh
curl -s http://localhost:8080/healthz
curl -s http://localhost:8080/readyz
```

A container image can also be built from the repository root:

```sh
docker build -t chora-ai-kernel-orchestrator .
docker run --rm -p 8080:8080 chora-ai-kernel-orchestrator
```

The container listens on port `8080` and starts Uvicorn with `chora_ai_kernel_orchestrator.main:app`.

## Usage

The service exposes four HTTP route groups:

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/healthz` | Liveness check |
| `GET` | `/readyz` | Readiness check |
| `POST` | `/v1/orchestrator/weakness/{upload_id}/resume` | Resume a Growth-Edge HITL review |
| `GET` | `/v1/prompt-registry/agents/{agent_id}/versions` | List prompt versions for an agent |
| `GET` | `/v1/prompt-registry/agents/{agent_id}/resolve` | Resolve the prompt version for an agent and optional tenant |
| `GET` | `/v1/prompt-registry/agents/{agent_id}/versions/{version}` | Get one prompt version and its segments |

The weakness resume endpoint requires `X-Tenant-Id`. Its JSON request body contains a bounded review decision:

```json
{
  "action": "confirm",
  "edges": [
    {
      "proposed_edge_id": "edge-1",
      "decision": "accept"
    }
  ],
  "added_struggles": [],
  "selected_outputs": []
}
```

The allowed top-level actions are `confirm` and `reiterate`. The route reconstructs the LangGraph checkpoint thread from the tenant ID and upload ID. If the analyser crew is not configured, the route returns `503`.

Prompt-registry routes read the repository configured by the FastAPI lifespan. An unconfigured catalogue returns `503`; an unknown agent or version returns `404`.

The application is configured through environment variables. The most important settings are:

| Variable | Purpose | Default |
| --- | --- | --- |
| `PORT` | HTTP listen port | `8080` |
| `CHORA_AI_KERNEL_PG_DSN` | PostgreSQL DSN used by LangGraph/checkpoint and registry paths | unset |
| `CHORA_AI_KERNEL_PG_DSN_SECRET_ID` | Secret identifier used when the PostgreSQL DSN is resolved through the secrets adapter | unset |
| `NATS_URL` | NATS JetStream connection URL | unset |
| `S3_ENDPOINT` | MinIO/S3 endpoint used by object-store adapters | unset |
| `S3_ACCESS_KEY_ID` | MinIO/S3 access key | unset |
| `S3_SECRET_ACCESS_KEY` | MinIO/S3 secret key | unset |
| `S3_SECURE` | Use TLS for the MinIO/S3 client | unset |
| `QGEN_IMAGE_GCS_BUCKET` | Bucket name for rendered qgen images | unset |
| `QGEN_CREW_ENABLED` | Enable the QGen dispatch lane | off |
| `OE_GRADING_CREW_ENABLED` | Enable the OE grading lane | off |
| `WEAKNESS_ANALYSER_ENABLED` | Enable the Growth-Edge weakness analyser lane | off |
| `PROMPT_PROMOTION_AUDIT_ENABLED` | Enable prompt-promotion audit processing | off |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OTLP trace collector endpoint | implementation default |
| `CHORA_AGENT_GUARDRAIL_MAPPING_PATH` | Override the agent-to-guardrail-tier YAML mapping | bundled config |
| `CHORA_GUARDRAIL_BLOCKLIST` | Comma-separated local screener patterns | built-in list |
| `CHORA_SAFESEARCH_FAIL_CLOSED` | Image-gate failure mode | fail open |

The dispatch subscribers use NATS JetStream subjects and durable consumers. The kennel runtime coordinates completion handling, dead-letter processing, park reaping, and one Postgres outbox drain. Dispatch-related configuration is controlled by the lane enable flags and `NATS_URL`.

## Development

Run the formatting, dependency, type, lint, and test checks used by the repository:

```sh
uv sync --all-extras
uv run ruff check .
uv run mypy src/chora_ai_kernel_orchestrator
uv run pytest
```

Coverage is configured in `pyproject.toml` with a minimum reported coverage threshold of 85 percent.

The repository's GitHub Actions workflow also checks:

```sh
uv run ruff check .
uv run mypy src/chora_ai_kernel_orchestrator
uv run pytest
```

Integration tests use the `integration` pytest marker. Tests that require external infrastructure skip when their dependency is not available, while the normal unit-test suite is designed to run without a broker or database.

The project layout is:

```text
src/chora_ai_kernel_orchestrator/
  main.py                  FastAPI application entrypoint and lifespan
  adapter/
    checkpointer/          LangGraph checkpoint persistence
    gcs/                   MinIO/S3-backed image and object handling
    http/                  FastAPI handlers and prompt-registry routes
    modelarmor/            Guardrail port and local screener
    postgres/              PostgreSQL runtime adapters
    pubsub/                NATS transport, subscribers, outbox, park and completion handling
    secrets/               DSN/secret resolution
  domain/
    agent_dispatch/        Dispatch state and parking policies
    agents/                Agent definitions
    ai_assist_crew/        AI-assist workflow state and registry
    oe_grading_crew/       OE grading state and scoring
    prompt_registry/       Prompt catalogue and resolution logic
    qgen_crew/              QGen workflow state
    registry/               Agent and capability metadata
    state/                  Shared workflow state
    weakness_analyser_crew/ Growth-Edge analysis and review state
  orchestrators/           LangGraph workflows and crew runners
  observability/           Logging and tracing
config/
  agent-guardrail-mapping.yaml
  PII_Closure_Map.yaml
testdata/                   Vendored prompt and agent fixtures
migrations/                 PostgreSQL schema migrations
Dockerfile                  Container build
pyproject.toml              Python package, dependencies, tooling and test configuration
uv.lock                     Locked dependency graph
```

The project declares the console script `chora-ai-kernel-orchestrator`, which invokes `chora_ai_kernel_orchestrator.main:run`. The Docker image instead starts the FastAPI application with Uvicorn.
