"""Tests for the JetStream transport's durable-name derivation.

The durable name is the JetStream consumer identity, so it must key off the
ORIGINAL subject: ``_dlq.`` is a transport prefix chora-common/eventbus puts on
the subject a consumer dead-lettered, and deriving from it would give the DLQ
reaper a different durable from every other consumer on the same subject.
"""

from __future__ import annotations

from chora_ai_kernel_orchestrator.adapter.pubsub.nats import NatsConsumerLoop


def _loop(subject: str) -> NatsConsumerLoop:
    return NatsConsumerLoop(url="nats://unit-test/never-dialled", subjects=[subject], subscriber=None)


def test_a_plain_subject_derives_the_durable_by_replacing_dots() -> None:
    loop = _loop("chora.consumption.weakness_doc.uploaded.v1")
    assert loop._durable_name("chora.consumption.weakness_doc.uploaded.v1") == (  # noqa: SLF001
        "chora-ai-kernel-chora-consumption-weakness_doc-uploaded-v1"
    )


def test_the_dlq_transport_prefix_is_stripped_before_deriving_the_durable() -> None:
    loop = _loop("_dlq.chora.ai_kernel.agent_dispatch.oe_evaluate_requested.v1")
    assert loop._durable_name("_dlq.chora.ai_kernel.agent_dispatch.oe_evaluate_requested.v1") == (  # noqa: SLF001
        "chora-ai-kernel-chora-ai_kernel-agent_dispatch-oe_evaluate_requested-v1"
    )
