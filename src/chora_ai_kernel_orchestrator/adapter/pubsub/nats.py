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
  pump per consumer, and the same callback-timeout + nack-on-failure
  discipline the StreamingPull loops had.

Connection is lazy: nothing opens a socket until ``start()`` so unit tests
and ``/healthz`` stay cheap.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable
from typing import Any, Protocol

logger = logging.getLogger(__name__)

ENV_NATS_URL = "NATS_URL"


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

    def ack(self) -> None:
        _fire_and_forget(self._msg.ack())

    def nack(self) -> None:
        _fire_and_forget(self._msg.nak())


class NatsConsumerLoop:
    """JetStream pull-consumer loop replacing a Pub/Sub StreamingPull.

    One durable pull consumer per subject, one fetch pump per consumer. Each
    fetched message is wrapped in :class:`NatsMessage` and handed to the
    subscriber's ``handle_message`` under the callback timeout; a raise or a
    timeout NACKs so JetStream redelivers (the subscriber owns the happy-path
    ack/nack, exactly as under StreamingPull).
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
    ) -> None:
        if not (url or "").strip():
            raise ValueError("NatsConsumerLoop: url required")
        subs = [s.strip() for s in subjects if (s or "").strip()]
        if not subs:
            raise ValueError(
                "NatsConsumerLoop: at least one subject is required; a loop "
                "with nothing to listen on starts clean and parks every run forever"
            )
        self._url = url.strip()
        self._subjects = subs
        self._subscriber = subscriber
        self._callback_timeout_s = callback_timeout_s if callback_timeout_s and callback_timeout_s > 0 else 60.0
        self._durable_prefix = durable_prefix
        self._subject_prefix = (subject_prefix or "").strip()
        self._fetch_batch = fetch_batch
        self._fetch_timeout_s = fetch_timeout_s
        self._connect_timeout_s = connect_timeout_s

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

    async def start(self) -> None:
        if self._stopped is not None:
            raise RuntimeError("NatsConsumerLoop: already started")
        self._stopped = False
        self._nc = await self._connect()
        js = self._nc.jetstream()
        for subject in self._subjects:
            full = self._full_subject(subject)
            durable = f"{self._durable_prefix}-{subject.replace('.', '-')}"
            try:
                sub = await js.pull_subscribe(full, durable=durable)
            except Exception:
                logger.exception("nats.consumer.pull_subscribe_failed", extra={"subject": full})
                await self._drain_tasks()
                await self._close_connection()
                raise
            self._tasks.append(asyncio.create_task(self._pump(full, sub), name=f"nats-pump-{subject}"))
            logger.info("nats.consumer.started", extra={"subject": full, "durable": durable})

    async def _connect(self) -> Any:
        import nats  # lazy — keeps the test path SDK-free

        return await nats.connect(
            self._url,
            connect_timeout=self._connect_timeout_s,
            reconnect_time_wait=2.0,
            max_reconnect_attempts=10,
        )

    async def _pump(self, subject: str, sub: Any) -> None:
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
                await self._handle(msg)

    async def _handle(self, msg: Any) -> None:
        wrapped = NatsMessage(msg)
        try:
            await asyncio.wait_for(
                self._subscriber.handle_message(wrapped),
                self._callback_timeout_s,
            )
        except Exception:
            logger.exception("nats.consumer.callback_failed")
            # The subscriber owns ack/nack; landing here is a marshaling /
            # timeout failure, so NACK and let JetStream redeliver.
            wrapped.nack()

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
    "ENV_NATS_URL",
    "NatsConsumerLoop",
    "NatsMessage",
]
