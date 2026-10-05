# chora-ai-kernel-orchestrator

The Chora AI Kernel orchestrator — the Python LangGraph half of the hybrid
kernel (Tier 2 D5). It is the **kennel**: the always-on runtime that parks,
resumes, and dispatches AI agent runs across the Chora crews (qgen, OE
grading, weakness analysis, and the single-agent fold lanes).

The service is **cloud-neutral**: the message bus is **NATS JetStream**, the
object store is **MinIO** (S3-compatible), the DSN comes from the
environment, and traces go to a local **OpenTelemetry Collector**. Nothing
talks to Google Cloud.

## What it does

- **HTTP API** (port `8080`): liveness/readiness probes, the Growth-Edge
  HITL resume route, and the O+ prompt-catalogue read API.
- **Dispatch lanes** (NATS JetStream): each crew consumes its request
  subject, parks the run on an agent dispatch, and resumes it from the
  checkpoint when the agent completes. All lanes are env-gated and default
  off.
- **Outbox drain**: a single dispatcher drains `ai_kernel_outbox_events`
  (Postgres) to NATS with retry + deadletter semantics.
- **Prompt registry**: the baseline seed spec + revision catalogue, sliced
  from the vendored golden fixtures under `testdata/prompt_baselines/`.
- **Guardrail**: a tier-aware local screener (the cloud-neutral stand-in for
  Cloud Model Armor) + a local SafeSearch gate (stand-in for Cloud Vision).

## Runtime

Python 3.13, FastAPI + uvicorn, LangGraph (Postgres checkpointer), NATS
JetStream (`nats-py`), MinIO (`minio`), OpenTelemetry (OTLP). Dependencies
are managed with **uv** (`pyproject.toml` + `uv.lock`).

## Local Docker deployment

The service is designed to run under the `chora-stack` compose (Postgres 18,
NATS, MinIO, OTEL collector). To run it standalone:

```bash
docker build -t chora-ai-kernel-orchestrator .
docker run -d --name ai-kernel -p 8080:8080 \
  -e CHORA_AI_KERNEL_PG_DSN=postgres://chora:chora@postgres:5432/chora_ai_kernel?sslmode=disable \
  -e NATS_URL=nats://nats:4222 \
  -e S3_ENDPOINT=http://minio:9000 \
  -e S3_ACCESS_KEY_ID=chora -e S3_SECRET_ACCESS_KEY=chora \
  chora-ai-kernel-orchestrator
```

With no env set, the service still boots (HTTP-only): the checkpointer
falls back to in-memory, the kennel runtime + lanes stay off, and `/readyz`
reports `ready`. Set the lane enable flags + `NATS_URL` + the DSN to bring
the dispatch lanes up.

### Environment variables

| Variable | Purpose |
| --- | --- |
| `PORT` | HTTP port (default `8080`) |
| `CHORA_AI_KERNEL_PG_DSN` | Postgres DSN (checkpointer, outbox, prompt registry) |
| `NATS_URL` | NATS JetStream connection URL (kennel runtime + lanes) |
| `S3_ENDPOINT` / `S3_ACCESS_KEY_ID` / `S3_SECRET_ACCESS_KEY` | MinIO object store (image upload + weakness download) |
| `QGEN_IMAGE_GCS_BUCKET` | image bucket name (the `gs://` scheme is retained as the canonical object URI) |
| `QGEN_CREW_ENABLED` / `OE_GRADING_CREW_ENABLED` / `WEAKNESS_ANALYSER_ENABLED` / `PROMPT_PROMOTION_AUDIT_ENABLED` | lane enable flags (default off) |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OTLP collector endpoint (tracing) |
| `CHORA_AGENT_GUARDRAIL_MAPPING_PATH` | override the vendored guardrail tier map |
| `CHORA_GUARDRAIL_BLOCKLIST` | comma-separated local guardrail blocklist |
| `CHORA_SAFESEARCH_FAIL_CLOSED` | `1` to block every image in the weakness gate (default: fail open, logged) |

## Development

```bash
uv sync --all-extras
uv run pytest
```

The suite is ~1800 tests (unit + integration). Integration tests that need a
live Postgres / the eval harness skip when their dependency is absent.

## Layout

```
src/chora_ai_kernel_orchestrator/
  main.py                  FastAPI entrypoint + lifespan (the kennel)
  adapter/
    pubsub/                NATS JetStream transport (publisher, consumer loops, outbox)
    gcs/                   MinIO-backed image upload + blob download
    modelarmor/            local guardrail screener (tier-aware)
    weakness/              SafeSearch gate + screener adapter
    secrets/               env-backed DSN resolver
    http/                  FastAPI handlers + prompt-registry router
    checkpointer/          LangGraph Postgres saver (lazy)
  domain/                  prompt registry, agent dispatch, crews
  orchestrators/           the LangGraph crew graphs
  observability/           logging + OTLP tracing
config/                    agent-guardrail-mapping.yaml (vendored)
testdata/                  prompt baselines + agent configs (vendored fixtures)
migrations/                Postgres migrations (outbox, park ledger, prompt registry)
tests/                     unit + integration suite
```

## Notes on the cloud-neutral port

- **Pub/Sub → NATS JetStream.** The wire contract is unchanged: the envelope
  rides as message headers, the destination is the row's topic (a NATS
  subject), and ack-after-processing + the callback-timeout + nack-on-failure
  discipline is preserved.
- **Cloud Storage → MinIO.** The `gs://` URI scheme is retained as the
  canonical object reference (the FE sends `gs://` URIs); the adapter
  translates to a MinIO bucket/key and mints presigned GET URLs.
- **Cloud Model Armor / Cloud Vision → local substitutes.** There is no
  drop-in local equivalent for a managed moderation API, so the guardrail
  port is backed by a minimal, configurable, tier-aware local screener and
  the image gate fails open (loudly) by default. The `Screener` /
  `SafeSearchPort` seams and the service's public API are unchanged; a
  deployment that needs production-grade screening should front these with a
  local moderation service and re-point the port.
- **Secret Manager → env.** The DSN comes from `CHORA_AI_KERNEL_PG_DSN`.
- **Cloud Trace → OTLP.** Traces go to the local OTEL collector.
