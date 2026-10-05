"""Coverage for the LazyPostgresSaver proxy + close_checkpointer helper.

We don't actually connect to Postgres — the test injects a fake builder
that mimics the `from_conn_string(dsn)` contextmanager interface so the
proxy's open / close / __getattr__ paths are exercised.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any

import pytest

from chora_ai_kernel_orchestrator.adapter.checkpointer.factory import (
    LazyPostgresSaver,
    close_checkpointer,
)


@dataclass
class _FakeRawConn:
    """Stand-in for the raw psycopg connection wrapped by PostgresSaver.

    The reconnect-on-stale path in `_ensure_open` inspects `inner.conn.closed`
    (or `inner._conn.closed`) to detect a server-side idle drop. The real
    PostgresSaver exposes this via either attribute depending on version.
    """

    closed: bool = False


@dataclass
class _FakeInner:
    """Stand-in for the real PostgresSaver.

    `put` mirrors the `BaseCheckpointSaver.put` signature (4 args) so
    delegation through the LazyPostgresSaver subclass works correctly.
    The post-2026-05-17 refactor turned the proxy into a proper
    `BaseCheckpointSaver` subclass; we no longer rely on `__getattr__`
    to forward arbitrary method names.

    `conn` exposes the raw psycopg connection so the stale-detection
    path in `_ensure_open` can introspect liveness. Tests flip
    `conn.closed = True` to simulate Cloud SQL idle drop.
    """

    name: str = "FakePostgresSaver"
    closed: bool = False
    written: list[Any] = field(default_factory=list)
    conn: _FakeRawConn = field(default_factory=_FakeRawConn)

    def put(
        self,
        config: Any,
        checkpoint: Any,
        metadata: Any,
        new_versions: Any,
    ) -> Any:
        self.written.append(config)
        return config


class _FakeBuilder:
    """Mimics PostgresSaver.from_conn_string."""

    def __init__(self) -> None:
        self.opened_count = 0
        self.exited_count = 0
        self.dsns_seen: list[str] = []
        self.last_inner: _FakeInner | None = None

    @contextlib.contextmanager
    def from_conn_string(self, dsn: str) -> Any:
        self.opened_count += 1
        self.dsns_seen.append(dsn)
        inner = _FakeInner()
        self.last_inner = inner
        try:
            yield inner
        finally:
            self.exited_count += 1
            inner.closed = True


def test_lazy_open_and_delegate() -> None:
    builder = _FakeBuilder()
    saver = LazyPostgresSaver(dsn="postgresql://x:5432/y", builder=builder)

    # Not yet opened.
    assert saver.opened is False
    assert builder.opened_count == 0

    # First put() opens lazily via _ensure_open() inside the override.
    saver.put("k1", {"v": 1, "id": "cp"}, {"source": "loop"}, {})
    assert saver.opened is True
    assert builder.opened_count == 1

    # Second call reuses the same inner.
    saver.put("k2", {"v": 1, "id": "cp"}, {"source": "loop"}, {})
    assert builder.opened_count == 1


def test_explicit_open_then_close() -> None:
    builder = _FakeBuilder()
    saver = LazyPostgresSaver(dsn="postgresql://x:5432/y", builder=builder)

    inner = saver.open()
    assert inner.name == "FakePostgresSaver"
    assert builder.opened_count == 1

    # Repeated open is idempotent.
    saver.open()
    assert builder.opened_count == 1

    # Close runs the contextmanager exit.
    saver.close()
    assert builder.exited_count == 1
    assert saver.opened is False

    # Idempotent close.
    saver.close()
    assert builder.exited_count == 1


def test_close_checkpointer_helper_handles_inmemory() -> None:
    """In-memory savers ignore close — the helper must be defensive."""
    from langgraph.checkpoint.memory import InMemorySaver

    saver = InMemorySaver()
    # No explosion on a saver without `close`.
    close_checkpointer(saver)


def test_close_checkpointer_helper_calls_close_when_present() -> None:
    """When the saver exposes close(), the helper invokes it."""
    builder = _FakeBuilder()
    saver = LazyPostgresSaver(dsn="postgresql://x:5432/y", builder=builder)
    saver.open()
    close_checkpointer(saver)
    assert builder.exited_count == 1


def test_close_checkpointer_helper_swallows_exceptions() -> None:
    """Best-effort close — exceptions don't propagate."""

    class _Boom:
        def close(self) -> None:
            raise RuntimeError("kaboom")

    # No raise.
    close_checkpointer(_Boom())


@pytest.mark.parametrize("dsn", ["", "   ", "\t\n"])
def test_factory_blank_dsn_returns_inmemory(dsn: str, monkeypatch: pytest.MonkeyPatch) -> None:
    from langgraph.checkpoint.memory import InMemorySaver

    from chora_ai_kernel_orchestrator.adapter.checkpointer.factory import (
        build_async_checkpointer_from_env,
        build_checkpointer_from_env,
    )

    monkeypatch.setenv("CHORA_AI_KERNEL_PG_DSN", dsn)
    assert isinstance(build_checkpointer_from_env(), InMemorySaver)
    assert isinstance(build_async_checkpointer_from_env(), InMemorySaver)


def test_async_factory_returns_postgres_when_dsn_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHORA_AI_KERNEL_PG_DSN", "postgresql://localhost:5432/x")
    from chora_ai_kernel_orchestrator.adapter.checkpointer.factory import (
        build_async_checkpointer_from_env,
    )

    saver = build_async_checkpointer_from_env()
    cls_name = type(saver).__name__
    # AsyncPostgresSaver from_conn_string returns an _AsyncGeneratorContextManager
    # in current langgraph; assert it is NOT InMemorySaver (i.e., Postgres
    # branch was taken).
    assert "InMemory" not in cls_name


def test_thread_id_validates_blank_inputs() -> None:
    from chora_ai_kernel_orchestrator.adapter.checkpointer.factory import (
        build_thread_id,
    )

    with pytest.raises(ValueError):
        build_thread_id(tenant_id="", workflow_id="ai-assist", run_id="r")
    with pytest.raises(ValueError):
        build_thread_id(tenant_id="t", workflow_id="", run_id="r")
    with pytest.raises(ValueError):
        build_thread_id(tenant_id="t", workflow_id="ai-assist", run_id="")


# ---------------------------------------------------------------------------
# CHECKPOINTER-STALE-CONN coverage — handoff doc 2026-05-17 ~23:21 SGT.
#
# Cloud SQL drops idle psycopg connections after ~10 min server-side;
# LazyPostgresSaver cached the inner saver indefinitely + never validated
# before reuse, so the next `aget_tuple` raised `psycopg.OperationalError:
# the connection is closed` and the AI-Assist job stuck QUEUED forever.
#
# Fix combo: (A) append libpq keepalive params to DSN, (B) liveness probe
# + reopen in `_ensure_open`. Both small + isolated; ship together.
# ---------------------------------------------------------------------------


def test_open_appends_libpq_keepalive_params_to_dsn_without_query() -> None:
    """DSN without `?` gets keepalives appended via `?`."""
    builder = _FakeBuilder()
    saver = LazyPostgresSaver(
        dsn="postgresql://user:pass@10.0.0.1:5432/chora_ai_kernel",
        builder=builder,
    )
    saver.open()
    seen = builder.dsns_seen[-1]
    assert "keepalives=1" in seen
    assert "keepalives_idle=60" in seen
    assert "keepalives_interval=10" in seen
    assert "keepalives_count=3" in seen
    # Used the `?` separator (no pre-existing query string).
    assert seen.count("?") == 1
    assert "?keepalives=1" in seen


def test_open_appends_libpq_keepalive_params_to_dsn_with_existing_query() -> None:
    """DSN with `?` gets keepalives appended via `&`."""
    builder = _FakeBuilder()
    saver = LazyPostgresSaver(
        dsn="postgresql://u:p@h:5432/db?sslmode=require",
        builder=builder,
    )
    saver.open()
    seen = builder.dsns_seen[-1]
    # Original query preserved + new params appended after `&`.
    assert "sslmode=require" in seen
    assert "&keepalives=1" in seen
    assert seen.count("?") == 1  # exactly one query separator


def test_ensure_open_reopens_on_stale_connection() -> None:
    """Cloud SQL idle drop: inner.conn.closed flips True → _ensure_open reopens."""
    builder = _FakeBuilder()
    saver = LazyPostgresSaver(dsn="postgresql://x:5432/y", builder=builder)

    saver.put("k1", {"v": 1, "id": "cp"}, {"source": "loop"}, {})
    assert builder.opened_count == 1
    first_inner = builder.last_inner
    assert first_inner is not None

    # Simulate server-side idle drop — the raw conn is now closed.
    first_inner.conn.closed = True

    # Next call should detect the stale conn + reopen.
    saver.put("k2", {"v": 1, "id": "cp"}, {"source": "loop"}, {})
    assert builder.opened_count == 2
    # And the new inner is a different instance, with a fresh live conn.
    assert builder.last_inner is not first_inner
    assert builder.last_inner is not None
    assert builder.last_inner.conn.closed is False


def test_ensure_open_skips_reopen_when_inner_is_live() -> None:
    """No spurious reopens — healthy inner is reused across calls."""
    builder = _FakeBuilder()
    saver = LazyPostgresSaver(dsn="postgresql://x:5432/y", builder=builder)

    saver.put("k1", {"v": 1, "id": "cp"}, {"source": "loop"}, {})
    saver.put("k2", {"v": 1, "id": "cp"}, {"source": "loop"}, {})
    saver.put("k3", {"v": 1, "id": "cp"}, {"source": "loop"}, {})
    assert builder.opened_count == 1


def test_ensure_open_treats_missing_conn_attribute_as_live() -> None:
    """Defensive: when the inner has no `conn` / `_conn` (e.g. InMemorySaver,
    test fakes that pre-date this change), assume live to avoid spurious
    reopens. The reconnect path is gated on a KNOWN-closed signal."""

    class _NoConnInner:
        def put(self, *_args: Any, **_kwargs: Any) -> Any:
            return None

    @contextlib.contextmanager
    def _build(_dsn: str) -> Any:
        yield _NoConnInner()

    class _NoConnBuilder:
        opened_count = 0

        def from_conn_string(self, dsn: str) -> Any:
            self.opened_count += 1
            return _build(dsn)

    builder = _NoConnBuilder()
    saver = LazyPostgresSaver(dsn="postgresql://x:5432/y", builder=builder)

    saver.put("k1", {"v": 1, "id": "cp"}, {"source": "loop"}, {})
    saver.put("k2", {"v": 1, "id": "cp"}, {"source": "loop"}, {})
    # One open — the missing-conn case must NOT trigger reopen loops.
    assert builder.opened_count == 1
