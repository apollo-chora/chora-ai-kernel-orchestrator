"""NATS JetStream transport for the AI Kernel orchestrator.

Cloud-neutral replacement for Google Cloud Pub/Sub. The kennel runtime and
every dispatch lane publish and consume through NATS JetStream:

* ``NatsPublisher`` publishes an ``OutboxRow`` to a NATS subject (the row's
  ``topic``) with the envelope as message headers — the same wire contract
  Pub/Sub attributes gave subscribers (filter on tenant_id / traceparent
  without parsing the payload).
* ``NatsMessage`` wraps a ``nats.aio.msg.Msg`` to the synchronous
  ``_MessageLike`` protocol the subscribers already speak (data / attributes
  / ack / nack). ack/nack are fire-and-forget on the running loop; the
  inbox-idempotency store is the durability backstop for at-least-once.
* ``NatsConsumerLoop`` is the JetStream pull-consumer loop that replaces the
  Pub/Sub StreamingPull loops: one durable pull consumer per subject, a fetch
  pump per consumer, and retry/backoff/dead-letter discipline matching the
  Go ``chora-common/eventbus`` pattern (MaxDeliver, AckWait, Backoff,
  DLQSubject).

Connection is lazy: nothing opens a socket until ``start()`` so unit tests
and ``/healthz`` stay cheap.

## The dead-letter contract

The DLQ transfer is **at-least-once**, and a terminal DLQ failure requires
operator intervention. Both are deliberate, and both are visible:

* The DLQ copy is published before the original is ACKed, so a crash between
  the two leaves the original to be redelivered and the DLQ copy written
  twice. Consumers of a DLQ subject must be idempotent.
* ``NatsMessage.ack`` schedules the ACK without awaiting the broker's
  confirmation, so the ``nats.consumer.dead_lettered`` log means "scheduled",
  not "settled". This matches ``chora-common/eventbus``, which also does not
  confirm the ACK.
* At the redelivery ceiling the broker will not deliver the message again, so
  if every DLQ publish attempt fails the message is neither ACKed nor
  redelivered. ``nats.consumer.dlq_publish_failed`` is the operator signal
  for that case; there is no automatic recovery from it.

What is guaranteed is that a message is never ACKed before its DLQ copy is on
the broker, and that a transient DLQ publish failure is retried on this loop's
own backoff schedule rather than abandoned.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any, Protocol

logger = logging.getLogger(__name__)

ENV_NATS_URL = "NATS_URL"

#: Transport prefix ``chora-common/eventbus/bus.go`` ``DLQSubject`` puts on the
#: original subject when it dead-letters a message. An infrastructure address,
#: not a domain event name.
_DLQ_SUBJECT_PREFIX = "_dlq."

#: Default redelivery ceiling before DLQ routing (``chora-common/eventbus``
#: ``ConsumerConfig.MaxDeliver``).
DEFAULT_MAX_DELIVER = 5

#: Default AckWait in seconds — how long a delivered message may be unacked
#: before redelivery (``chora-common/eventbus`` ``ConsumerConfig.AckWait``).
DEFAULT_ACK_WAIT_S = 30.0

#: Default per-attempt backoff schedule in seconds
#: (``chora-common/eventbus`` ``ConsumerConfig.Backoff``).
DEFAULT_BACKOFF_S: tuple[float, ...] = (1.0, 2.0, 5.0, 10.0)


class _SubscriberLike(Protocol):
    async def handle_message(self, msg: Any) -> None: ...


def _fire_and_forget(coro: Any) -> None:
    """Schedule an async ack/nack on the running loop.

    The subscribers speak a synchronous ack/nack interface (the Pub/Sub
    shape). NATS acks are coroutines; scheduling them keeps that interface
    unchanged. The inbox-idempotency store is the durability backstop if an
    ack is ever lost to a connection drop mid-flight.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:  # pragma: no cover — no loop; nothing to schedule on
        coro.close()
        return
    task = loop.create_task(coro)

    def _log_error(t: asyncio.Task[Any]) -> None:
        if not t.cancelled() and t.exception() is not None:
            logger.warning("nats.ack_nack_failed", exc_info=t.exception())

    task.add_done_callback(_log_error)


def _broker_sequence_id(msg: Any) -> str:
    """The broker's own identity for a delivered message, or ``""``.

    The JetStream metadata's stream name and stream sequence. Used as the DLQ
    message id when the producer set no ``Chora-Event-Id``, so a replay after
    a lost ACK is still recognised as a duplicate of the DLQ copy already
    written. Empty when the message carries no readable metadata, which means
    no broker identity is available either.
    """
    try:
        md = msg.metadata
    except Exception:  # fail-loud-exempt: no readable metadata means no broker
        # identity either, so the DLQ copy carries no dedup id.
        return ""
    seq = getattr(md, "sequence", None)
    stream_seq = getattr(seq, "stream", 0) if seq is not None else 0
    if not stream_seq:
        return ""
    # The CONSUMER sequence increments on every redelivery, so it must not be
    # part of the id: a replay after a lost ACK would then look like a new
    # message. The STREAM sequence is the message's own stable identity.
    stream_name = getattr(md, "stream", "")
    return f"{stream_name}.{stream_seq}" if stream_name else f"{stream_seq}"


class NatsMessage:
    """Wrap a ``nats.aio.msg.Msg`` to the sync ``_MessageLike`` protocol."""

    def __init__(self, msg: Any) -> None:
        self._msg = msg

    @property
    def data(self) -> bytes:
        return self._msg.data

    @property
    def attributes(self) -> dict[str, str]:
        headers = getattr(self._msg, "headers", None) or {}
        return {str(k): str(v) for k, v in headers.items()}

    @property
    def subject(self) -> str:
        return str(getattr(self._msg, "subject", "") or "")

    @property
    def delivery_count(self) -> int:
        """Number of times this message has been delivered (1-indexed).

        Read from the JetStream message metadata; falls back to 1 when
        metadata is unavailable (e.g. in unit tests with a stub message).
        """
        try:
            md = self._msg.metadata
        except Exception:  # fail-loud-exempt: no readable delivery count means
            # the retry ceiling cannot be enforced for this message, so the
            # only safe default is the first delivery; this read also runs
            # inside the failure handler, where re-raising would mask the
            # original error.
            return 1
        if md is None:
            return 1
        return int(getattr(md, "num_delivered", 0) or 0)

    @property
    def headers(self) -> dict[str, str]:
        """All headers as a flat dict (last value wins)."""
        return self.attributes

    def ack(self) -> None:
        _fire_and_forget(self._msg.ack())

    def nack(self) -> None:
        _fire_and_forget(self._msg.nak())

    def nak_with_delay(self, delay_s: float) -> None:
        """NAK with a redelivery delay in seconds.

        JetStream will not redeliver the message until the delay has elapsed.
        A delay of 0 or less falls back to an immediate NAK.
        """
        if delay_s > 0:
            _fire_and_forget(self._msg.nak(delay=delay_s))
        else:
            _fire_and_forget(self._msg.nak())


class NatsConsumerLoop:
    """JetStream pull-consumer loop replacing a Pub/Sub StreamingPull.

    One durable pull consumer per subject, one fetch pump per consumer. Each
    fetched message is wrapped in :class:`NatsMessage` and handed to the
    subscriber's ``handle_message`` under the callback timeout.

    On failure the loop implements the same retry/backoff/dead-letter
    discipline as the Go ``chora-common/eventbus``:

    * Under the ``max_deliver`` ceiling: NAK with a delay from the
      ``backoff_s`` schedule (clamped to the last entry once exhausted).
    * At the ceiling: republish to the DLQ subject (``_dlq.<subject>``
      unless ``dlq_subject`` is set), stamp ``Chora-Dlq-Delivery-Count`` and
      other ``Chora-Dlq-*`` headers, then ACK the original only after the
      DLQ write succeeds.
    """

    def __init__(
        self,
        *,
        url: str,
        subjects: Iterable[str],
        subscriber: _SubscriberLike,
        callback_timeout_s: float = 60.0,
        durable_prefix: str = "chora-ai-kernel",
        subject_prefix: str = "",
        fetch_batch: int = 1,
        fetch_timeout_s: float = 5.0,
        connect_timeout_s: float = 10.0,
        max_deliver: int = DEFAULT_MAX_DELIVER,
        ack_wait_s: float = DEFAULT_ACK_WAIT_S,
        backoff_s: Iterable[float] = DEFAULT_BACKOFF_S,
        dlq_subject: str = "",
    ) -> None:
        if not (url or "").strip():
            raise ValueError("NatsConsumerLoop: url required")
        subs = [s.strip() for s in subjects if (s or "").strip()]
        if not subs:
            raise ValueError(
                "NatsConsumerLoop: at least one subject is required; a loop "
                "with nothing to listen on starts clean and parks every run forever"
            )
        backoff = tuple(float(d) for d in backoff_s)
        if not backoff:
            raise ValueError(
                "NatsConsumerLoop: backoff_s must not be empty; an empty "
                "schedule NAKs immediately, which is the unbounded redelivery "
                "this loop exists to stop"
            )
        # Normalise the ceiling BEFORE validating against it: a caller passing
        # max_deliver=0 means "use the default", not "reject every schedule".
        max_deliver = max_deliver if max_deliver > 0 else DEFAULT_MAX_DELIVER
        if len(backoff) > max_deliver:
            raise ValueError(f"NatsConsumerLoop: {len(backoff)} backoff entries exceeds max_deliver {max_deliver}")
        for i, d in enumerate(backoff):
            if d <= 0:
                raise ValueError(f"NatsConsumerLoop: backoff[{i}] must be positive, got {d}")
        self._url = url.strip()
        self._subjects = subs
        self._subscriber = subscriber
        self._callback_timeout_s = callback_timeout_s if callback_timeout_s and callback_timeout_s > 0 else 60.0
        self._durable_prefix = durable_prefix
        self._subject_prefix = (subject_prefix or "").strip()
        self._fetch_batch = fetch_batch
        self._fetch_timeout_s = fetch_timeout_s
        self._connect_timeout_s = connect_timeout_s
        self._max_deliver = max_deliver
        self._ack_wait_s = ack_wait_s if ack_wait_s > 0 else DEFAULT_ACK_WAIT_S
        self._backoff_s = backoff
        self._dlq_subject = (dlq_subject or "").strip()

        self._nc: Any = None
        self._tasks: list[asyncio.Task[None]] = []
        self._stopped: bool | None = None

    @property
    def subjects(self) -> list[str]:
        return list(self._subjects)

    @property
    def url(self) -> str:
        return self._url

    def _full_subject(self, subject: str) -> str:
        return f"{self._subject_prefix}{subject}" if self._subject_prefix else subject

    def _durable_name(self, subject: str) -> str:
        """JetStream durable name for a bound subject.

        The durable keys off the ORIGINAL subject, not the ``_dlq.``-prefixed
        one: ``_dlq.`` is a transport prefix ``chora-common/eventbus`` puts on
        the subject a consumer dead-lettered, so it says nothing about which
        consumer is listening. Deriving from it would also make the durable
        name differ from every other consumer on the same original subject.
        """
        base = subject[len(_DLQ_SUBJECT_PREFIX) :] if subject.startswith(_DLQ_SUBJECT_PREFIX) else subject
        return f"{self._durable_prefix}-{base.replace('.', '-')}"

    def _dlq_subject_for(self, subject: str) -> str:
        """The DLQ subject for a bound subject.

        Uses the explicitly configured ``dlq_subject`` if set, otherwise
        derives the conventional ``_dlq.<subject>`` per
        ``chora-common/eventbus/bus.go`` ``DLQSubject``.
        """
        if self._dlq_subject:
            return self._dlq_subject
        return f"{_DLQ_SUBJECT_PREFIX}{self._full_subject(subject)}"

    async def start(self) -> None:
        if self._stopped is not None:
            raise RuntimeError("NatsConsumerLoop: already started")
        self._stopped = False
        self._nc = await self._connect()
        js = self._nc.jetstream()
        for subject in self._subjects:
            full = self._full_subject(subject)
            durable = self._durable_name(subject)
            try:
                from nats.js.api import AckPolicy, ConsumerConfig

                config = ConsumerConfig(
                    durable_name=durable,
                    ack_policy=AckPolicy.EXPLICIT,
                    max_deliver=self._max_deliver,
                    ack_wait=self._ack_wait_s,
                    backoff=list(self._backoff_s),
                )
                sub = await js.pull_subscribe(full, durable=durable, config=config)
            except Exception:
                logger.exception("nats.consumer.pull_subscribe_failed", extra={"subject": full})
                await self._drain_tasks()
                await self._close_connection()
                raise
            self._tasks.append(asyncio.create_task(self._pump(full, sub, durable), name=f"nats-pump-{subject}"))
            logger.info("nats.consumer.started", extra={"subject": full, "durable": durable})

    async def _connect(self) -> Any:
        import nats  # lazy — keeps the test path SDK-free

        return await nats.connect(
            self._url,
            connect_timeout=self._connect_timeout_s,
            reconnect_time_wait=2.0,
            max_reconnect_attempts=10,
        )

    async def _pump(self, subject: str, sub: Any, durable: str) -> None:
        while not self._stopped:
            try:
                msgs = await sub.fetch(batch=self._fetch_batch, timeout=self._fetch_timeout_s)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # nats-py raises TimeoutError on an idle fetch; anything else
                # is a real transport failure — back off and retry.
                if _is_fetch_timeout(exc):
                    continue
                logger.exception(
                    "nats.consumer.fetch_failed",
                    extra={"subject": subject, "err": f"{type(exc).__name__}: {exc}"},
                )
                await asyncio.sleep(1.0)
                continue
            for msg in msgs:
                if self._stopped:
                    break
                await self._handle(msg, durable=durable)

    async def _handle(self, msg: Any, *, durable: str) -> None:
        wrapped = NatsMessage(msg)
        try:
            await asyncio.wait_for(
                self._subscriber.handle_message(wrapped),
                self._callback_timeout_s,
            )
        except Exception as exc:
            delivered = wrapped.delivery_count
            if delivered >= self._max_deliver:
                logger.error(
                    "nats.consumer.dead_lettering",
                    extra={
                        "subject": wrapped.subject,
                        "delivery_count": delivered,
                        "err": f"{type(exc).__name__}: {exc}",
                    },
                )
                await self._dead_letter(msg, wrapped, durable, exc)
            else:
                delay = self._retry_delay(delivered)
                if delivered <= 1:
                    logger.exception(
                        "nats.consumer.callback_failed",
                        extra={
                            "subject": wrapped.subject,
                            "delivery_count": delivered,
                            "delay_s": delay,
                        },
                    )
                else:
                    logger.warning(
                        "nats.consumer.callback_failed_retry",
                        extra={
                            "subject": wrapped.subject,
                            "delivery_count": delivered,
                            "delay_s": delay,
                            "err": f"{type(exc).__name__}: {exc}",
                        },
                    )
                wrapped.nak_with_delay(delay)

    async def _dead_letter(self, msg: Any, wrapped: NatsMessage, durable: str, cause: BaseException) -> None:
        """Republish a message to the DLQ subject, then ACK the original.

        The original is acknowledged only after the DLQ write succeeds. If
        the write fails after retries the original is left unsettled and
        redelivered on this loop's backoff schedule, so the DLQ attempt is
        retried rather than abandoned.
        """
        dlq_subject = self._dlq_subject_for(wrapped.subject)
        headers = self._build_dlq_headers(msg, wrapped, durable, cause)

        schedule = [1.0, 2.0, 5.0, 10.0]
        last_err: BaseException | None = None

        for attempt in range(len(schedule) + 1):
            try:
                js = self._nc.jetstream()
                await js.publish(dlq_subject, payload=msg.data, headers=headers)
            except Exception as exc:  # fail-loud-exempt: the DLQ publish is retried
                # on the same schedule the Go eventbus uses, and only the
                # terminal failure is logged below, so a per-attempt log would
                # repeat one operator-visible event several times.
                last_err = exc
                if attempt == len(schedule):
                    break
                await asyncio.sleep(schedule[attempt])
                continue
            # The DLQ copy is on the broker: settle the original. An ACK
            # failure is NOT a publish failure and must not re-run the loop,
            # or the DLQ copy is written a second time.
            wrapped.ack()
            logger.info(
                "nats.consumer.dead_lettered",
                extra={
                    "subject": wrapped.subject,
                    "dlq_subject": dlq_subject,
                    "delivery_count": wrapped.delivery_count,
                },
            )
            return

        logger.error(
            "nats.consumer.dlq_publish_failed",
            extra={
                "subject": wrapped.subject,
                "dlq_subject": dlq_subject,
                "err": f"{type(last_err).__name__}: {last_err}",
            },
        )
        # Leave the original unsettled rather than ACKing a message whose DLQ
        # copy was never written, and redeliver it on this loop's own schedule
        # instead of leaving the retry to the broker's AckWait. At the ceiling
        # the broker may still drop it, which is why the error above is the
        # operator signal and not a guarantee of recovery.
        wrapped.nak_with_delay(schedule[-1])

    def _build_dlq_headers(self, msg: Any, wrapped: NatsMessage, durable: str, cause: BaseException) -> dict[str, str]:
        """Build the header set for a DLQ copy of the original message.

        Copies all original headers, removes the broker message id (so the
        DLQ copy is not suppressed as a duplicate), and stamps the
        ``Chora-Dlq-*`` headers the Go eventbus uses
        (``chora-common/eventbus/jetstream.go`` ``dlqHeaders``).
        """
        src = getattr(msg, "headers", None) or {}
        hdr: dict[str, str] = {}
        for k, v in src.items():
            if isinstance(v, (list, tuple)):
                hdr[str(k)] = str(v[-1]) if v else ""
            else:
                hdr[str(k)] = str(v)

        hdr.pop("Nats-Msg-Id", None)
        event_id = hdr.get("Chora-Event-Id", "")
        if event_id:
            hdr["Nats-Msg-Id"] = f"{event_id}.dlq.{durable}"
        else:
            # No application event id: fall back to the broker's own identity
            # for this delivery, so a replay after a lost ACK is still
            # suppressed as a duplicate of the copy already written.
            broker_id = _broker_sequence_id(msg)
            if broker_id:
                hdr["Nats-Msg-Id"] = f"{broker_id}.dlq.{durable}"

        hdr["Chora-Dlq-Source-Subject"] = wrapped.subject
        hdr["Chora-Dlq-Consumer"] = durable
        hdr["Chora-Dlq-Delivery-Count"] = str(wrapped.delivery_count)
        hdr["Chora-Dlq-Failed-At"] = datetime.now(tz=UTC).isoformat()
        hdr["Chora-Dlq-Reason"] = str(cause)
        return hdr

    def _retry_delay(self, delivered: int) -> float:
        """The backoff interval for the given (1-based) delivery attempt.

        Clamps to the last entry once the schedule is exhausted, and returns
        0 when no schedule is configured (caller then uses an immediate NAK).
        """
        if not self._backoff_s:
            return 0.0
        idx = max(delivered - 1, 0)
        idx = min(idx, len(self._backoff_s) - 1)
        return self._backoff_s[idx]

    async def stop(self) -> None:
        if self._stopped is not None and not self._stopped:
            return
        self._stopped = True
        await self._drain_tasks()
        await self._close_connection()

    async def _drain_tasks(self) -> None:
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []

    async def _close_connection(self) -> None:
        if self._nc is not None:
            try:
                await self._nc.close()
            except Exception:
                logger.exception("nats.consumer.close_failed")
            self._nc = None


def _is_fetch_timeout(exc: BaseException) -> bool:
    """True when an exception is nats-py's idle-fetch timeout (not a failure)."""
    if isinstance(exc, asyncio.TimeoutError):
        return True
    name = type(exc).__name__
    return name in {"TimeoutError", "nats.errors.TimeoutError"}


__all__ = [
    "DEFAULT_ACK_WAIT_S",
    "DEFAULT_BACKOFF_S",
    "DEFAULT_MAX_DELIVER",
    "ENV_NATS_URL",
    "NatsConsumerLoop",
    "NatsMessage",
]
