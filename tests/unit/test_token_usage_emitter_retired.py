"""The orchestrator's legacy TokenUsageLedger emit lane is RETIRED.

Incident 2026-07-23 (sibling of CHO-2221). Between 2026-07-17 and
2026-07-23 the DLQ ``chora.dlq.observability.token_usage.recorded.v1``
accumulated 220 quarantined events. Every one sampled (68/68) carried:

  - ``model_id`` EMPTY  -> chora-observability's consumer refused it with
    ``token_usage_consumer: model_id required`` (correctly: a ledger row
    with no model cannot be priced), retried 5x, then dead-lettered.
  - ``cost_micros`` == 0.
  - an idempotency key of the shape
    ``token_usage.{tenant}.{invocation}.{qgen_question|qgen_critic}.attempt_N``
    which ONLY this producer emits.

Root cause: ``qgen_crew._trace_row`` writes ``engine_resource`` into the
trace row only ``if engine_resource:`` and NO production graph node ever
passed one, so ``qgen_crew_runner`` read ``""`` and
``proto_wire_encoder.encode_token_usage_recorded`` mapped that empty
string onto ``model_id`` (field 5). 100% of this lane's events were
therefore unpriceable. The unit tests never caught it because
``test_handle_started_engine_resource_propagates`` HAND-FILLS
``engine_resource`` into the trace row it feeds the runner.

Resolution: retire the lane rather than stamp a model into it.

  1. Per ADR-163 chora-model-gateway is the SOLE producer of
     ``chora.observability.token_usage.recorded.v1`` post-cutover, and
     ``events/observability/token_usage.proto`` already records that this
     writer is "on its retirement path".
  2. The qgen agents reach the LLM THROUGH the gateway
     (``CHORA_GATEWAY_ENDPOINT=gateway.chora.site:443``), which meters the
     call and stamps the real model. ``QGenBatchRunner._emit_token_usage``
     says so in its own docstring: "the gateway also meters each LLM
     Invoke, so this is additive cost attribution".
  3. So stamping a value here would DOUBLE-COUNT every qgen call, and the
     only value available (a ``gke://`` engine resource) is not a model
     and cannot be priced.

These tests pin the retirement so the lane cannot be reintroduced by
accident.
"""

from __future__ import annotations

import importlib
import inspect

import pytest

from chora_ai_kernel_orchestrator.orchestrators import qgen_crew_runner
from chora_ai_kernel_orchestrator.orchestrators.qgen_crew_runner import (
    QGenBatchRunner,
    QGenCrewRunner,
)

RETIRED_TOPIC = "chora.observability.token_usage.recorded.v1"


@pytest.mark.parametrize("runner_cls", [QGenCrewRunner, QGenBatchRunner])
def test_runner_no_longer_accepts_a_token_usage_emitter(runner_cls: type) -> None:
    """Neither runner takes a token_usage_emitter port any more."""
    params = inspect.signature(runner_cls.__init__).parameters
    assert "token_usage_emitter" not in params, (
        f"{runner_cls.__name__} still accepts token_usage_emitter; the "
        "orchestrator must not produce token_usage events (ADR-163: the "
        "Model Gateway is the sole producer)."
    )


@pytest.mark.parametrize("runner_cls", [QGenCrewRunner, QGenBatchRunner])
def test_runner_has_no_token_usage_emit_method(runner_cls: type) -> None:
    """The per-hop emit helpers are gone, not merely unwired."""
    leftovers = [n for n in dir(runner_cls) if "token_usage" in n]
    assert leftovers == [], f"{runner_cls.__name__} still defines {leftovers}"


def test_runner_module_does_not_reference_the_retired_topic() -> None:
    """No module-level constant, and the literal is not left lying around."""
    assert not hasattr(qgen_crew_runner, "TOPIC_OBSERVABILITY_TOKEN_USAGE")
    source = inspect.getsource(qgen_crew_runner)
    assert RETIRED_TOPIC not in source, "qgen_crew_runner still carries the retired token_usage topic literal"


def test_token_usage_outbox_writer_module_is_deleted() -> None:
    """The producer itself is removed, so nothing can re-wire it."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("chora_ai_kernel_orchestrator.adapter.pubsub.token_usage_outbox_writer")


def test_proto_wire_encoder_no_longer_encodes_token_usage() -> None:
    """The empty-model_id encoder that produced the quarantined events is gone."""
    encoder = importlib.import_module("chora_ai_kernel_orchestrator.adapter.pubsub.proto_wire_encoder")
    assert not hasattr(encoder, "encode_token_usage_recorded")


def test_qgen_crew_wiring_builds_no_token_usage_writer() -> None:
    """The composition root must not construct the retired producer."""
    wiring = importlib.import_module("chora_ai_kernel_orchestrator.adapter.pubsub.qgen_crew_wiring")
    source = inspect.getsource(wiring)
    assert "TokenUsageLedgerOutboxWriter" not in source
    assert "token_usage_emitter" not in source
