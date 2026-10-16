"""Tests for the JetStream transport's durable-name derivation.

The durable name is the JetStream consumer identity, so it must key off the
ORIGINAL subject: ``_dlq.`` is a transport prefix chora-common/eventbus puts on
the subject a consumer dead-lettered, and deriving from it would give the DLQ
reaper a different durable from every other consumer on the same subject.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from chora_ai_kernel_orchestrator.adapter.pubsub.nats import (
    DEFAULT_MAX_DELIVER,
    NatsConsumerLoop,
    NatsMessage,
)


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


class _MsgWithMetadataProperty:
    """Stands in for ``nats.aio.msg.Msg``: ``metadata`` is a property."""

    def __init__(self, num_delivered: int) -> None:
        self._num_delivered = num_delivered

    @property
    def metadata(self) -> object:
        return SimpleNamespace(num_delivered=self._num_delivered)


class _MsgWithoutMetadata:
    """Stands in for a double with no JetStream metadata at all."""


def test_delivery_count_reads_the_jetstream_metadata_property() -> None:
    """REGRESSION: nats-py's ``Msg.metadata`` is a property, not a method.

    The first version of the retry ceiling read ``msg.metadata()``. On a real
    nats-py message that raises ``TypeError`` ('Metadata' object is not
    callable); the broad fallback swallowed it and returned 1, so every real
    message reported a delivery count of 1 and neither the redelivery ceiling
    nor the per-attempt logging ever fired.
    """
    assert NatsMessage(_MsgWithMetadataProperty(1)).delivery_count == 1
    assert NatsMessage(_MsgWithMetadataProperty(3)).delivery_count == 3
    assert NatsMessage(_MsgWithMetadataProperty(5)).delivery_count == 5


def test_delivery_count_falls_back_to_one_without_metadata() -> None:
    assert NatsMessage(_MsgWithoutMetadata()).delivery_count == 1


def test_the_ceiling_is_normalised_before_it_validates_the_schedule() -> None:
    """``max_deliver=0`` means 'use the default', so it must not reject a schedule."""
    loop = NatsConsumerLoop(
        url="nats://unit-test/never-dialled",
        subjects=["chora.test"],
        subscriber=None,
        max_deliver=0,
    )
    assert loop._max_deliver == DEFAULT_MAX_DELIVER  # noqa: SLF001


def test_a_schedule_longer_than_the_ceiling_is_refused() -> None:
    with pytest.raises(ValueError, match="backoff entries exceeds"):
        NatsConsumerLoop(
            url="nats://unit-test/never-dialled",
            subjects=["chora.test"],
            subscriber=None,
            max_deliver=2,
            backoff_s=(1.0, 2.0, 5.0),
        )


def test_an_empty_schedule_is_refused() -> None:
    """An empty schedule NAKs immediately — the unbounded redelivery this loop stops."""
    with pytest.raises(ValueError, match="must not be empty"):
        NatsConsumerLoop(
            url="nats://unit-test/never-dialled",
            subjects=["chora.test"],
            subscriber=None,
            backoff_s=(),
        )


class _MsgWithSequence:
    """Stands in for a JetStream message carrying broker metadata."""

    def __init__(self, stream_name: str, stream_seq: int, consumer_seq: int) -> None:
        self._md = SimpleNamespace(
            stream=stream_name,
            sequence=SimpleNamespace(stream=stream_seq, consumer=consumer_seq),
        )

    @property
    def metadata(self) -> object:
        return self._md


def test_the_dlq_id_is_stable_across_redeliveries() -> None:
    """The consumer sequence increments per delivery; the stream sequence does not.

    A DLQ id that included the consumer sequence would change on every
    redelivery, so a replay after a lost ACK would not be recognised as a
    duplicate of the copy already written.
    """
    first = _MsgWithSequence(stream_name="CHORA_EVENTS", stream_seq=42, consumer_seq=7)
    redelivered = _MsgWithSequence(stream_name="CHORA_EVENTS", stream_seq=42, consumer_seq=8)
    loop = _loop("chora.test")
    headers_first = loop._build_dlq_headers(  # noqa: SLF001
        first, NatsMessage(first), durable="chora-ai-kernel-chora-test", cause=RuntimeError("boom")
    )
    headers_again = loop._build_dlq_headers(  # noqa: SLF001
        redelivered, NatsMessage(redelivered), durable="chora-ai-kernel-chora-test", cause=RuntimeError("boom")
    )
    assert (
        headers_first["Nats-Msg-Id"]
        == headers_again["Nats-Msg-Id"]
        == ("CHORA_EVENTS.42.dlq.chora-ai-kernel-chora-test")
    )
