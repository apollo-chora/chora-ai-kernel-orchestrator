# `adapter/modelarmor` — Cloud Model Armor screener (ADR-152)

Canonical Python adapter for **Cloud Model Armor** — the runtime
guardrail primitive that supersedes the deprecated `chora-guardrail`
service per
[`docs/architecture/adrs/adr-152-chora-guardrail-superseded-by-cloud-model-armor.md`](../../../../../../../docs/architecture/adrs/adr-152-chora-guardrail-superseded-by-cloud-model-armor.md).

| | |
|---|---|
| **Package** | `chora_ai_kernel_orchestrator.adapter.modelarmor` |
| **SDK** | `google-cloud-modelarmor` (>=0.6.0, GA 2026) |
| **Go counterpart** | `libs/chora-go-common/modelarmor/` (authored in parallel — keep shapes in sync) |
| **Replaces** | Legacy `adapter/http/guardrail_client.py` (deleted at cutover; see ADR-152 §"Code consequences") |

## Surface

```python
from chora_ai_kernel_orchestrator.adapter.modelarmor import (
    new_screener,  # async factory → concrete ModelArmorScreener
    Screener,  # Protocol (port)
    ScreenRequest,
    ScreenResult,
    Verdict,  # ALLOW | BLOCK | INSPECT_ONLY
    FilterHit,
    StubScreener,  # for unit tests
)
```

### Verdict semantics

| Verdict | Meaning | Legacy chora-guardrail equivalent |
|---|---|---|
| `ALLOW` | No filter matched — pass through. | `allow` |
| `BLOCK` | At least one filter matched **and** the template's `invocation_result == SUCCESS` (template chose to enforce). | `refuse` |
| `INSPECT_ONLY` | Filter(s) matched but the template ran in inspect-only mode (`invocation_result == PARTIAL` / unspecified). Surface the event downstream; **do not** gate the LLM call. | NEW |

Rewrite is no longer emitted by the runtime guardrail — content rewrite
is a content-author responsibility post-ADR-152.

### Verdict derivation edge cases

- No `sanitization_result` on the response → `BLOCK`
  (`reason="model_armor_no_sanitization_result"`). Security wins
  (Principle Conflict Resolution #1).
- `filter_match_state == NO_MATCH_FOUND` and no per-filter `MATCH_FOUND`
  → `ALLOW`.
- Top-level `MATCH_FOUND` + `invocation_result == SUCCESS` → `BLOCK`.
- Top-level `MATCH_FOUND` + `invocation_result == PARTIAL` (or
  `UNSPECIFIED` / missing) → `INSPECT_ONLY`.
- `MATCH_FOUND` + unrecognised `invocation_result` value → `BLOCK`.
- The SDK response **does not** include the template's
  `enforcement_type`, so we use `invocation_result` as the proxy. If you
  need a stricter (always-BLOCK on MATCH_FOUND) policy, wrap the
  screener at the LangGraph node and map `INSPECT_ONLY → BLOCK` there.

## Usage from the LangGraph guardrail node

`adapter/langgraph/graph.py::guardrail_screen_node` depends on
:class:`ModelArmorGuardrailPort` directly — template name resolution +
direction routing is internal to the port:

```python
from chora_ai_kernel_orchestrator.adapter.modelarmor import (
    GuardrailScreenInput,
    ModelArmorGuardrailPort,
)


async def guardrail_screen_node(state, *, guardrail: ModelArmorGuardrailPort) -> dict:
    payload = GuardrailScreenInput(
        tenant_id=state["tenant_id"],
        gcid=state["gcid"],
        agent_id=state["agent_id"],
        content=state["prompt"],
        direction="input",  # or "output" for post-LLM screening
    )
    result = await guardrail.screen(payload)
    return {
        "guardrail_decision": result.verdict.value,
        "guardrail_explanation": result.reason,
    }
```

## OpenTelemetry

Every call emits a span on tracer `chora_kernel.modelarmor` with
attributes:

| Attribute | Source |
|---|---|
| `chora.tenant_id` | `ScreenRequest.tenant_id` |
| `chora.agent_id` | `ScreenRequest.agent_id` |
| `chora.gcid` | `ScreenRequest.gcid` |
| `chora.model_armor.template_name` | `ScreenRequest.template_name` |
| `chora.model_armor.method` | `sanitize_user_prompt` / `sanitize_model_response` |
| `chora.model_armor.verdict` | `Verdict.value` |
| `chora.model_armor.latency_ms` | client-side measured ms |
| `chora.model_armor.filter_count` | `len(result.filters)` |

These follow the OpenInference / OpenLLMetry conventions used elsewhere
in this service (see `observability/tracing.py`).

## Testing

```bash
python -m pytest \
  src/chora_ai_kernel_orchestrator/adapter/modelarmor/tests/ -q
```

Tests are stub-driven — no cloud credentials required. Integration tests
that hit the real Model Armor API live under
`tests/poc/w3-iter4-chaos/` (per ADR-152 §"Test surface" — repurposed as
the M14 regression suite).

## Configuration

No inline config. `ModelArmorGuardrailPort.from_env()` reads these env
vars at composition root:

| Env var | Default | Source |
|---|---|---|
| `CHORA_MODELARMOR_PROJECT`  | `chora-489812`           | Terraform / Secret Manager |
| `CHORA_MODELARMOR_LOCATION` | `us-central1`            | Terraform (per ADR-148) |
| `CHORA_ENVIRONMENT`         | `dev`                    | Terraform (`dev`/`staging`/`prod`) |
| `CHORA_AGENT_GUARDRAIL_MAPPING_PATH` | bundled `chora-contracts/yaml/agent-guardrail-mapping.yaml` | Terraform override (optional) |

Per-agent template **tiers** (strict / balanced / permissive) live in
`chora-contracts/yaml/agent-guardrail-mapping.yaml` keyed by `agent_id`.
The port composes the full template name at request time:

```
projects/{project}/locations/{location}/templates/chora-guardrail-{tier}-{env}
```

Unknown agents fall back to the YAML's `default.template_tier`
(`strict` per ADR-152 — security wins; never permissive).
