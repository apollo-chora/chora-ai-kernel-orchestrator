# chora-ai-kernel-orchestrator

> The AI kernel: a Python service that runs the kennel runtime, parking and
> resuming agent work, dispatching the enabled crew lanes over NATS JetStream, and
> draining the durable outbox.

FastAPI for HTTP, LangGraph for the workflow graphs, PostgreSQL for checkpoints and
state, NATS JetStream for dispatch, and OpenTelemetry for traces.

## What it is

The orchestrator is the runtime that the `qgen-*`, `oe-*`, `companion-*` and
`kg-explorer` services plug into. It does four jobs:

- **Runs the kennel** — it owns the agent-dispatch state machine: park a dispatch,
  wait for a completion event, resume it, reap a park that overran, and route a
  failure to the dead-letter path.
- **Dispatches crew lanes** — four lanes are wired, each a set of roles subscribed
  to durable JetStream consumers. A lane is inert until its enable flag is set.
- **Drains the outbox** — one PostgreSQL outbox drain runs alongside the
  subscribers.
- **Serves three HTTP surfaces** — health, the Growth-Edge review resume, and the
  prompt catalogue.

In the stack it is built from `../chora-ai-kernel-orchestrator` and listens on
`ai-kernel-orchestrator:8080` **inside the Compose network only** — it publishes no
host port and exposes only `8080`.

### The crew lanes

**Seven** lanes are implemented; `compose.yml` enables **four** of them. `GET
/readyz` reports the live state of each enabled one.

| Lane | Roles | Enable flag | Default subscription | In the stack |
| --- | --- | --- | --- | --- |
| `qgen` | `qgen_generate`, `qgen_critique`, `qgen_render` | `QGEN_CREW_ENABLED` | — | on |
| `oe_grading` | `oe_evaluate`, `oe_moderate` | `OE_GRADING_CREW_ENABLED` | — | on |
| `weakness_analyser` | `companion_extract`, `companion_diagnose` | `WEAKNESS_ANALYSER_ENABLED` | — | on |
| `kg_exploration` | `kg_explore` | `KG_EXPLORATION_ENABLED` | `chora.consumption.concept_suggestion.requested.v1` | on |
| `companion_reflection` | `concept_suggestion` | `COMPANION_REFLECTION_ENABLED` | `chora.consumption.goal_knowledge.synthesis_requested.v1` | off |
| `companion_turn` | `companion_turn` | `COMPANION_TURN_ENABLED` | `chora.consumption.companion_turn.requested.v1` | off |
| `dose_recommendation` | `dose_recommendation` | `DOSE_RECOMMENDATION_ENABLED` | `chora.consumption.dose_recommendation.requested.v1` | off |

The first three are lane wirings under `adapter/pubsub/`; the last four are
single-agent lanes registered from `single_agent_lanes_wiring.py`, each with its own
`*_SUBSCRIPTION` override. Eight roles are subscribed in the stack's configuration.

A lane whose flag is unset never subscribes, so the crew service behind it receives
nothing and its queue stays empty — silence, not an error. The three disabled lanes
are implemented but unused by this stack; enabling one needs the extra inbound
traffic that `chora-consumption` would send.

### Readiness is a gate, not a liveness check

`GET /readyz` returns **503** rather than 200 in two cases, both with a JSON body
naming the cause:

| Body | Meaning |
| --- | --- |
| `{"status":"not_ready","reason":"guardrail_unconfigured"}` | the Model Armor guardrail could not be built from the environment |
| `{"status":"not_ready","reason":"kennel_runtime_not_ready"}` | the kennel runtime did not come up |

A process that serves `/healthz` but fails `/readyz` is up but not usable, and the
lanes that need the screener will not have started.

## Prerequisites

| Route | What you need |
| --- | --- |
| In the stack | A working `chora-stack` deployment. PostgreSQL, NATS and object storage come from it. |
| Standalone (HTTP only) | Python **3.13** and `uv`. No database or broker: the lanes stay disabled. |
| Standalone (lanes) | Plus **PostgreSQL 18 or newer** and a NATS JetStream instance. |
| Any database you supply | **PostgreSQL 18 or newer.** Migration `0060_uuidv7_pk_defaults.up.sql` raises `0060: uuidv7() is not available on this server; PostgreSQL 18+ is required`. The stack's `pgvector/pgvector:0.8.7-pg18` satisfies it. |

## Credentials

This service holds **no model-provider credentials**. It does not call Gemini or
Exa; it dispatches work over NATS and the crew services make the model calls. There
is no API key to obtain for it.

What it needs to start with its lanes active:

| Value | Where it comes from | Variable |
| --- | --- | --- |
| PostgreSQL password | the stack's `POSTGRES_PASSWORD`, carried into the DSN as the `chora_ai_kernel_app_rw` role | `CHORA_AI_KERNEL_PG_DSN` |
| NATS endpoint | the stack's NATS service; no credential in the local configuration | `NATS_URL` |
| Object-storage credentials | the stack's Garage access key and secret (`CHORA_S3_ACCESS_KEY_ID` / `CHORA_S3_SECRET_KEY`) | `S3_ACCESS_KEY_ID`, `S3_SECRET_ACCESS_KEY` |

Generate the database password with `openssl rand -hex 16` if you are starting from
nothing. Where each stack credential comes from, and how to obtain the provider
keys the *crew* services need, is documented once in
[`chora-stack/README.md`](../chora-stack/README.md#credentials).

One caveat worth knowing: `compose.yml` passes `CHORA_LOCAL_KEK` to this service,
but **no code in this repository reads it**. Treat it as unused here rather than as
a secret you must provision for the orchestrator.

## Run it in the stack

`compose.yml` does four things:

1. Builds the image from the sibling checkout at `../chora-ai-kernel-orchestrator`.
2. Runs `ai-kernel-migrate` first — a `postgres:18.6-alpine` one-shot applying this
   repository's migrations to `chora_ai_kernel`.
3. Starts the service once `postgres` is healthy, `nats` is healthy, `nats-init`
   has completed, and the migrate one-shot has completed successfully.
4. Publishes no host port, and adds `extra_hosts: localhost:host-gateway` so the
   object-store client can resolve `localhost:9000` (the origin baked into the
   image URLs it mints) from inside the container.

```sh
cd chora-stack
docker compose up -d ai-kernel-orchestrator
docker compose ps ai-kernel-orchestrator
```

It sets, among others: `PORT`, `CHORA_AI_KERNEL_PG_DSN`, `NATS_URL`,
`CHORA_SOURCE_PROJECT`, `OTEL_EXPORTER_OTLP_ENDPOINT`, `CREATION_BASE_URL`
(default `http://creation:8080`), `KROKI_ENDPOINT` (default
`http://kroki:8000`), the Garage S3 settings, and all four lane flags set to
`"true"`.

Because there is no host port, probe it from inside the network:

```sh
docker compose exec -T nats wget -qO- http://ai-kernel-orchestrator:8080/readyz
```

## Run it standalone

HTTP only — no database, no broker, no lanes:

```sh
git clone https://github.com/apollo-chora/chora-ai-kernel-orchestrator.git
cd chora-ai-kernel-orchestrator
uv sync --all-extras
uv run uvicorn chora_ai_kernel_orchestrator.main:app --host 0.0.0.0 --port 8080
```

With the default configuration every lane flag is off, so the process serves HTTP
and subscribes to nothing. `CHORA_AI_KERNEL_PG_DSN` and `NATS_URL` plus a lane flag
are only the infrastructure half: the lane builders also need a usable guardrail,
and startup aborts when an enabled lane cannot be built. Set the flag, start the
process, then read `readyz` to see whether the lane actually came up.

From a container instead:

```sh
docker build -t chora-ai-kernel-orchestrator .
docker run --rm -p 8080:8080 chora-ai-kernel-orchestrator
```

The image starts Uvicorn on `chora_ai_kernel_orchestrator.main:app`. The package
also declares a console script, `chora-ai-kernel-orchestrator`, which invokes
`chora_ai_kernel_orchestrator.main:run` — a different entry point from the image's.

Development checks, as the repository's CI runs them:

```sh
uv run ruff check .
uv run mypy src/chora_ai_kernel_orchestrator
uv run pytest
```

`pyproject.toml` sets an 85 percent coverage floor. Tests needing external
infrastructure carry the `integration` marker and skip when their dependency is
absent, so the unit suite runs with no broker and no database.

## Configuration

Every setting is an environment variable; there is no `.env.example` in this
repository, so the table below is the reference.

| Variable | Default | Purpose |
| --- | --- | --- |
| `PORT` | `8080` | HTTP listen port |
| `CHORA_AI_KERNEL_PG_DSN` | unset | PostgreSQL DSN for LangGraph checkpoints, the registry and the outbox |
| `CHORA_AI_KERNEL_PG_DSN_SECRET_ID` | unset | secret identifier used when the DSN is resolved through the secrets adapter |
| `NATS_URL` | unset | NATS JetStream URL; without it no lane starts |
| `QGEN_CREW_ENABLED` | off | enable the `qgen` lane |
| `OE_GRADING_CREW_ENABLED` | off | enable the `oe_grading` lane |
| `WEAKNESS_ANALYSER_ENABLED` | off | enable the `weakness_analyser` lane |
| `KG_EXPLORATION_ENABLED` | off | enable the `kg_exploration` lane |
| `COMPANION_REFLECTION_ENABLED` | off | enable the `companion_reflection` lane |
| `COMPANION_TURN_ENABLED` | off | enable the `companion_turn` lane |
| `DOSE_RECOMMENDATION_ENABLED` | off | enable the `dose_recommendation` lane |
| `KG_EXPLORATION_SUBSCRIPTION`, `COMPANION_REFLECTION_SUBSCRIPTION`, `COMPANION_TURN_SUBSCRIPTION`, `DOSE_RECOMMENDATION_SUBSCRIPTION` | the defaults in the lane table | per-lane subject override |
| `COMPANION_TURN_TENANT_INFLIGHT_CAP`, `DOSE_RECOMMENDATION_TENANT_INFLIGHT_CAP` | unset | per-tenant in-flight cap for those two lanes |
| `CREATION_BASE_URL` | unset (stack: `http://creation:8080`) | how the qgen crew reads a job's uploaded source material |
| `KROKI_ENDPOINT` | unset (stack: `http://kroki:8000`) | renders the Mermaid half of an image spec |
| `QGEN_IMAGE_GCS_BUCKET` | unset | bucket for rendered qgen images |
| `S3_ENDPOINT` | unset | object-store endpoint used from inside the network (stack: `http://garage:3900`) |
| `S3_ACCESS_KEY_ID`, `S3_SECRET_ACCESS_KEY` | unset | object-store credentials |
| `S3_SECURE` | unset | TLS for the object-store client |
| `S3_FORCE_PATH_STYLE` | unset | path-style addressing (required by Garage) |
| `S3_REGION` | unset (stack: `us-east-1`) | region the client signs for. Leaving it unset makes minio-py resolve the bucket region with a live request against `S3_PUBLIC_ENDPOINT` — which is why the stack sets both this and the `extra_hosts` alias |
| `S3_PUBLIC_ENDPOINT` | unset (stack: `http://localhost:9000`) | the **browser-facing** origin baked into minted image URLs; must differ from `S3_ENDPOINT` |
| `CHORA_ENVIRONMENT` | unset | environment label |
| `CHORA_PUBSUB_PROJECT` | unset | Pub/Sub project for the GCP transport |
| `CHORA_MODELARMOR_PROJECT`, `CHORA_MODELARMOR_LOCATION` | unset | Model Armor templates project and location |
| `CHORA_AGENT_GUARDRAIL_MAPPING_PATH` | bundled `config/agent-guardrail-mapping.yaml` | override the agent-to-guardrail-tier mapping |
| `CHORA_AGENT_GUARDRAIL_MAPPING` | unset | inline override of the same mapping |
| `CHORA_GUARDRAIL_BLOCKLIST` | built-in list | comma-separated local screener patterns |
| `CHORA_SAFESEARCH_FAIL_CLOSED` | fail open | image-gate failure mode |
| `QGEN_PROGRESS_STREAM_ENABLED` | unset | emit qgen progress on the stream |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | implementation default | OTLP trace collector |

`S3_ENDPOINT` and `S3_PUBLIC_ENDPOINT` are not interchangeable. The first is how the
service reaches the bucket from inside the container; the second is the origin a
browser must be able to resolve, and it is what gets baked into a minted URL. The
stack sets them to `http://garage:3900` and `http://localhost:9000` respectively,
and the `extra_hosts` entry exists precisely so the in-container client can still
reach the public origin when probing a bucket's region.

## Verify it works

### It is up, and the kennel knows its lanes

```sh
docker compose exec -T nats wget -qO- http://ai-kernel-orchestrator:8080/healthz
docker compose exec -T nats wget -qO- http://ai-kernel-orchestrator:8080/readyz
```

`healthz` returns the process state:

```json
{"status":"ok","time":"..."}
```

`readyz` returns the process state **and** the live kennel block — the roles it has
and, per lane, whether it started and whether it failed. That is the check that
matters for this service:

```json
{
  "status": "ready",
  "kennel": {
    "roles": ["companion_diagnose", "companion_extract", "kg_explore",
              "oe_evaluate", "oe_moderate", "qgen_critique",
              "qgen_generate", "qgen_render"],
    "lanes": {
      "kg_exploration":    {"started": true, "failed": false, "detail": ""},
      "oe_grading":        {"started": true, "failed": false, "detail": ""},
      "qgen":              {"started": true, "failed": false, "detail": ""},
      "weakness_analyser": {"started": true, "failed": false, "detail": ""}
    }
  }
}
```

A lane with `started: false` is disabled or lost its subscription; `failed: true`
carries the reason in `detail`. Compare against the four lanes listed above — all
four should read `started: true` in the stack.

### Its migrations landed

```sh
docker compose exec -T postgres psql -U chora_ai_kernel_migrate -d chora_ai_kernel \
  -c "select count(*) from schema_migrations;"
```

The count matches the 20 forward migration files listed under
[Migrations](#migrations).

### Its HTTP routes answer

```sh
docker compose exec -T nats wget -qO- \
  http://ai-kernel-orchestrator:8080/v1/prompt-registry/agents/qgen_generate/versions \
  | head -c 200
```

Read the status, not just the body. Two non-200 answers mean different things:
`503` is the registry never being wired during the lifespan, and `404` is it being
wired but holding no version for that agent id. Both are failures of this check; a
`200` with a JSON listing is the pass.

## Troubleshooting

- **`curl http://localhost:8080/...` from the host fails** — expected. This service
  publishes no host port. Probe from inside the network with
  `docker compose exec -T nats wget -qO- http://ai-kernel-orchestrator:8080/readyz`.
- **`readyz` shows a lane `started: false`** — its enable flag is unset, or
  `NATS_URL` is missing. A disabled lane is silent: the crew service behind it
  simply never receives a dispatch.
- **`readyz` shows a lane `failed: true`** — the `detail` field carries the reason.
  Check `docker compose logs ai-kernel-orchestrator` for the same message in full.
- **`readyz` answers `503` with `guardrail_unconfigured`** — the Model Armor
  guardrail could not be built from the environment. In the stack the in-process
  `LocalScreener` satisfies it; a standalone run needs that configuration, and
  without it readiness never passes and every screener-dependent lane refuses to
  build.
- **`readyz` answers `503` with `kennel_runtime_not_ready`** — the runtime did not
  come up; the logs name the failing step.
- **The migrate one-shot cannot find its role or database on a re-used volume** —
  `postgres-init/00-create-roles.sh` and `01-create-databases.sh` run only when
  PostgreSQL initializes its data directory for the first time. On a volume where
  they did not complete, the `chora_ai_kernel` database and the
  `chora_ai_kernel_migrate` / `_app_rw` roles are missing, and starting only this
  service will not create them.
- **Image uploads fail outside the stack** — the bucket named by
  `QGEN_IMAGE_GCS_BUCKET` must exist. The stack's `garage-init` one-shot creates it;
  anywhere else you create it yourself, with the `S3_*` credentials allowed to write.
- **The migrate one-shot fails with `0060: uuidv7() is not available on this
  server`** — the database server is older than PostgreSQL 18. The container image
  being `postgres:18.6-alpine` does not make the *server* 18.
- **A document-grounded generation fails and blames the document** — check
  `CREATION_BASE_URL`. Without it the qgen crew cannot read the job's extracted
  source material, and the author sees an error that misattributes the cause to the
  upload.
- **A candidate that requests an image aborts the whole generation** — a job
  carrying any image spec requires a renderer. Without `KROKI_ENDPOINT` the crew
  raises before the per-spec fail-soft path, so the stack runs `kroki` and
  `kroki-mermaid` and defaults the endpoint rather than leaving it optional.
- **`POST /v1/orchestrator/weakness/{upload_id}/resume` returns `503`** — the
  weakness analyser crew is not configured, so there is no checkpoint thread to
  resume.
- **The prompt-registry routes return `503`** — the catalogue was not wired during
  the FastAPI lifespan. An unknown agent or version returns `404` instead.
- **The resume route returns `400`** — the body's top-level `action` must be
  `confirm` or `reiterate`, and `X-Tenant-Id` is required.
- **Minted image URLs point at a host the browser cannot reach** — `S3_PUBLIC_ENDPOINT`
  is wrong. It must be the origin the browser resolves, not `S3_ENDPOINT`.
- **`ALTER` on a populated `agent_decision_embeddings.embedding` discards vectors** —
  pgvector cannot cast between widths. Migration `0063` is a pure schema change only
  because the table was empty; a populated deployment needs a re-embed pass instead.

## API surface

Six routes: two health checks, one review resume, and three prompt-catalogue reads.

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/healthz` | Liveness |
| `GET` | `/readyz` | Readiness, including the live kennel lane state |
| `POST` | `/v1/orchestrator/weakness/{upload_id}/resume` | Resume a Growth-Edge HITL review. Requires `X-Tenant-Id` |
| `GET` | `/v1/prompt-registry/agents/{agent_id}/versions` | List prompt versions for an agent |
| `GET` | `/v1/prompt-registry/agents/{agent_id}/resolve` | Resolve the prompt version for an agent, optionally per tenant |
| `GET` | `/v1/prompt-registry/agents/{agent_id}/versions/{version}` | One prompt version and its segments |

The resume body carries a bounded review decision:

```json
{
  "action": "confirm",
  "edges": [{"proposed_edge_id": "edge-1", "decision": "accept"}],
  "added_struggles": [],
  "selected_outputs": []
}
```

The route reconstructs the LangGraph checkpoint thread from the tenant ID and the
upload ID, so the pair must match the one the analyser parked.

## Architecture

```text
src/chora_ai_kernel_orchestrator/
  main.py                  FastAPI application, lifespan wiring, lane startup
  adapter/
    agent_io/              dispatch state and the wire shapes
    checkpointer/          LangGraph checkpoint persistence
    creation/              reads a job's uploaded source material over HTTP
    events/                event bus helpers, closure subscriber wiring
    gcs/                   object-store image and blob handling
    http/                  FastAPI handlers and prompt-registry routes
    kroki/                 Mermaid renderer client
    modelarmor/            guardrail port and the in-process local screener
    pg/, postgres/         PostgreSQL adapters and pool plumbing
    pubsub/                NATS transport, lane wiring, outbox, park and completion
    registry/              agent registry
    secrets/               DSN and secret resolution
    weakness/              weakness-evidence plumbing
  domain/
    agent_dispatch/        dispatch state and parking policy
    agents/                agent definitions
    ai_assist_crew/        AI-assist workflow state
    oe_grading_crew/       OE grading state and scoring
    prompt_registry/       prompt catalogue and resolution
    qgen_crew/             qgen workflow state
    registry/              agent and capability metadata
    state/                 shared workflow state
    weakness_analyser_crew/Growth-Edge analysis and review state
  orchestrators/           LangGraph graphs and crew runners
  observability/           logging and tracing
  ops/                     operational helpers
config/
  agent-guardrail-mapping.yaml
  PII_Closure_Map.yaml
testdata/                  vendored prompt and agent fixtures
migrations/                PostgreSQL schema
```

The lane wiring lives in `adapter/pubsub/` — one module per lane, each taking its
enable flag and subscription name and binding a durable consumer. `main.py` decides
which lanes to start from the flags and records the result, which is what `readyz`
reads back.

## Events

Dispatch runs over NATS JetStream. Each enabled role gets a **durable consumer** on
its own subject, and the kennel runtime owns the lifecycle around it:

- **completion handling** — a dispatcher publishes a completion event, the kennel
  matches it to the parked dispatch and resumes the graph;
- **park reaping** — a dispatch that overran its park window is reaped rather than
  left waiting;
- **dead-letter processing** — a failed dispatch routes to the dead-letter path
  rather than being retried forever;
- **one outbox drain** — a single PostgreSQL outbox drain runs alongside the
  subscribers, matching every other service in the stack's pattern.

The subjects and consumer names follow the platform taxonomy
(`chora.<service>.<aggregate>.<event_type>.v<N>`). The four single-agent lanes bind
`chora.consumption.*` subjects by default — for instance
`chora.consumption.companion_turn.requested.v1` for the `companion_turn` lane — and
each has a `*_SUBSCRIPTION` override so a deployment can move it without a rebuild.
The exact subject per lane is in the table under [The crew lanes](#the-crew-lanes).

## Migrations

Twenty forward migration files, applied by `scripts/migrate.sh` in the
`ai-kernel-migrate` one-shot against the `chora_ai_kernel` database:

`0001_initial`, `0002_agent_registry_and_langgraph`, `0003_outbox`,
`0004_inbox_idempotency`, `0005_grant_outbox_explicit`,
`0006_langgraph_checkpoint_grants`, `0007_prompt_override_registry`,
`0008_prompt_override_promotion`, `0009_prompt_catalogue_baselines`,
`0010_prompt_version_label_dedup`, `0055_ai_assist_inflight_registry`,
`0056_agent_dispatch_parks`, `0057_companion_chat_sessions_schema`,
`0058_companion_chat_sessions_tables`, `0059_kernel_rls_fail_closed`,
`0060_uuidv7_pk_defaults`, `0061_langgraph_checkpoint_tables`,
`0062_langgraph_checkpoint_grants`, `0063_embedding_dimensions_1024`,
`9999_grant_app_roles`.

The filenames mix two conventions: `0001`–`0006` and `9999` are plain `.sql`, while
`0007`–`0010` and `0055`–`0063` carry the `.up.sql` suffix. Both are forward
migrations — the runner's rule is "every `*.sql` except `*.down.sql`", so the suffix
does not change which files run, only how they are named.

Two carry hard requirements:

- **`0060_uuidv7_pk_defaults`** switches primary-key defaults from
  `gen_random_uuid()` to PostgreSQL 18's native `uuidv7()`, and raises
  `0060: uuidv7() is not available on this server; PostgreSQL 18+ is required`
  otherwise.
- **`0063_embedding_dimensions_1024`** widens
  `agent_decision_embeddings.embedding` to the embedding route's native width. It is
  a pure schema change only because that table is empty — pgvector cannot cast
  between widths, so `ALTER` on a populated column would discard the stored vectors
  and require a re-embed pass.

The numbering jumps from `0010` to `0055`; that is the repository's history, not a
gap you need to fill. Every `*.down.sql` sibling is a destructive rollback the
runner never applies — it takes every `*.sql` except `*.down.sql`, records a SHA-256
checksum per file in `schema_migrations`, and refuses a file whose contents changed
after it was applied. A session-scoped advisory lock serialises concurrent
migrators.
