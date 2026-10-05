"""LazyPostgresSaver must subclass BaseCheckpointSaver (LangGraph 0.6.x).

Without this, `graph.compile(checkpointer=lazy_saver)` raises::

    TypeError: Invalid checkpointer provided. Expected `BaseCheckpointSaver`,
    `True`, `False`, or `None`. Received LazyPostgresSaver.

per `langgraph/types.py::ensure_valid_checkpointer`. The proxy must therefore:

1. ``isinstance(saver, BaseCheckpointSaver)`` returns ``True``
2. All public BaseCheckpointSaver methods (sync + async) delegate to the
   inner saver, opening lazily on first call
3. ``serde`` is inherited from the inner saver once opened (falls back to
   the default ``JsonPlusSerializer`` before open)
4. ``close()`` / ``open()`` / ``opened`` proxy public API preserved
"""

from __future__ import annotations

import builtins
import contextlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import pytest
from langgraph.checkpoint.base import BaseCheckpointSaver

from chora_ai_kernel_orchestrator.adapter.checkpointer.factory import (
    LazyPostgresSaver,
)

# ---------------------------------------------------------------------------
# Fakes — mimic the inner PostgresSaver surface the proxy must delegate to.
# ---------------------------------------------------------------------------


@dataclass
class _CallLog:
    """Records the method call name + args/kwargs for delegation assertion."""

    name: str
    args: tuple[Any, ...]
    kwargs: dict[str, Any]


@dataclass
class _FakeInnerSaver:
    """A stand-in for langgraph PostgresSaver used in delegation tests.

    Records every (a)method call to ``calls`` so tests can verify the
    proxy is forwarding correctly. The ``serde`` attribute mirrors what
    a real PostgresSaver exposes so the proxy can adopt it post-open.
    """

    calls: list[_CallLog] = field(default_factory=list)
    serde: Any = field(default_factory=lambda: object())

    # ---- sync surface ---------------------------------------------------

    def get_tuple(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append(_CallLog("get_tuple", args, kwargs))
        return "sentinel_tuple"

    def list(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append(_CallLog("list", args, kwargs))
        return iter(["sentinel_list_item"])

    def put(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append(_CallLog("put", args, kwargs))
        return "sentinel_put_cfg"

    def put_writes(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append(_CallLog("put_writes", args, kwargs))

    def delete_thread(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append(_CallLog("delete_thread", args, kwargs))

    def delete_for_runs(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append(_CallLog("delete_for_runs", args, kwargs))

    def copy_thread(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append(_CallLog("copy_thread", args, kwargs))

    def prune(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append(_CallLog("prune", args, kwargs))

    # ---- async surface --------------------------------------------------

    async def aget_tuple(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append(_CallLog("aget_tuple", args, kwargs))
        return "sentinel_atuple"

    async def alist(self, *args: Any, **kwargs: Any) -> Any:
        # alist must be an async generator on the real saver; emulate that.
        self.calls.append(_CallLog("alist", args, kwargs))

        async def _gen() -> Any:
            yield "sentinel_alist_item"

        return _gen()

    async def aput(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append(_CallLog("aput", args, kwargs))
        return "sentinel_aput_cfg"

    async def aput_writes(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append(_CallLog("aput_writes", args, kwargs))

    # ---- introspection helpers ------------------------------------------

    # NOTE: built-in `list` is shadowed by the `list` method on this
    # class, so the unqualified `list[Any]` annotation is interpreted as
    # the method. Use the fully-qualified `builtins.list` form instead.
    @property
    def config_specs(self) -> builtins.list[Any]:
        self.calls.append(_CallLog("config_specs", (), {}))
        return ["inner_spec"]


class _FakeBuilder:
    """Mimics ``PostgresSaver.from_conn_string`` contextmanager surface."""

    def __init__(self) -> None:
        self.opened_count = 0
        self.exited_count = 0
        self.last_inner: _FakeInnerSaver | None = None

    @contextlib.contextmanager
    def from_conn_string(  # noqa: ANN201 — cm
        self, dsn: str
    ) -> Any:
        self.opened_count += 1
        inner = _FakeInnerSaver()
        self.last_inner = inner
        try:
            yield inner
        finally:
            self.exited_count += 1


# ---------------------------------------------------------------------------
# Subclass identity
# ---------------------------------------------------------------------------


def test_lazy_postgres_saver_is_basecheckpointsaver_subclass() -> None:
    """LangGraph 0.6 ``ensure_valid_checkpointer`` requires this isinstance."""
    saver = LazyPostgresSaver(dsn="postgresql://x:5432/y", builder=_FakeBuilder())
    assert isinstance(saver, BaseCheckpointSaver), (
        "LazyPostgresSaver must subclass BaseCheckpointSaver so "
        "graph.compile() accepts it (see langgraph/types.py:111)."
    )


def test_lazy_postgres_saver_class_is_subclass_at_type_level() -> None:
    assert issubclass(LazyPostgresSaver, BaseCheckpointSaver)


# ---------------------------------------------------------------------------
# Public proxy API preserved
# ---------------------------------------------------------------------------


def test_public_proxy_api_preserved() -> None:
    """`opened`, `open()`, `close()` + private slots must still exist."""
    saver = LazyPostgresSaver(dsn="postgresql://x:5432/y", builder=_FakeBuilder())
    assert hasattr(saver, "open")
    assert hasattr(saver, "close")
    assert hasattr(saver, "opened")
    assert hasattr(saver, "_inner")
    assert hasattr(saver, "_builder")
    assert hasattr(saver, "_dsn")
    assert hasattr(saver, "_cm")
    assert saver.opened is False
    assert saver._inner is None


# ---------------------------------------------------------------------------
# Sync delegation — each method forwards args/kwargs to inner exactly once.
# ---------------------------------------------------------------------------


def test_sync_get_tuple_delegates_and_opens_lazily() -> None:
    builder = _FakeBuilder()
    saver = LazyPostgresSaver(dsn="postgresql://x:5432/y", builder=builder)

    result = saver.get_tuple({"configurable": {"thread_id": "t:w:r"}})

    assert builder.opened_count == 1
    assert result == "sentinel_tuple"
    inner = builder.last_inner
    assert inner is not None
    assert [c.name for c in inner.calls] == ["get_tuple"]
    assert inner.calls[0].args == ({"configurable": {"thread_id": "t:w:r"}},)


def test_sync_list_delegates() -> None:
    builder = _FakeBuilder()
    saver = LazyPostgresSaver(dsn="postgresql://x:5432/y", builder=builder)

    result = list(saver.list({"configurable": {"thread_id": "t:w:r"}}, limit=5))

    assert result == ["sentinel_list_item"]
    assert builder.last_inner is not None
    assert [c.name for c in builder.last_inner.calls] == ["list"]
    # The proxy preserves the BaseCheckpointSaver signature so all
    # filter/before/limit keys are forwarded (None defaults included).
    call_kwargs = builder.last_inner.calls[0].kwargs
    assert call_kwargs.get("limit") == 5
    assert "filter" in call_kwargs
    assert "before" in call_kwargs


def test_sync_put_delegates() -> None:
    builder = _FakeBuilder()
    saver = LazyPostgresSaver(dsn="postgresql://x:5432/y", builder=builder)

    cfg = {"configurable": {"thread_id": "t:w:r"}}
    metadata = {"source": "loop", "step": 0}
    versions = {"channel": 1}
    result = saver.put(cfg, {"v": 1, "id": "cp"}, metadata, versions)

    assert result == "sentinel_put_cfg"
    assert builder.last_inner is not None
    call = builder.last_inner.calls[0]
    assert call.name == "put"
    assert call.args == (cfg, {"v": 1, "id": "cp"}, metadata, versions)


def test_sync_put_writes_delegates() -> None:
    builder = _FakeBuilder()
    saver = LazyPostgresSaver(dsn="postgresql://x:5432/y", builder=builder)

    cfg = {"configurable": {"thread_id": "t:w:r"}}
    writes = [("channel_a", "value_a")]
    saver.put_writes(cfg, writes, "task-1", "/")

    assert builder.last_inner is not None
    call = builder.last_inner.calls[0]
    assert call.name == "put_writes"
    assert call.args == (cfg, writes, "task-1", "/")


def test_sync_delete_thread_delegates() -> None:
    builder = _FakeBuilder()
    saver = LazyPostgresSaver(dsn="postgresql://x:5432/y", builder=builder)
    saver.delete_thread("thread-xyz")
    assert builder.last_inner is not None
    call = builder.last_inner.calls[0]
    assert call.name == "delete_thread"
    assert call.args == ("thread-xyz",)


def test_sync_delete_for_runs_delegates() -> None:
    builder = _FakeBuilder()
    saver = LazyPostgresSaver(dsn="postgresql://x:5432/y", builder=builder)
    saver.delete_for_runs(["r1", "r2"])
    assert builder.last_inner is not None
    call = builder.last_inner.calls[0]
    assert call.name == "delete_for_runs"
    assert call.args == (["r1", "r2"],)


def test_sync_copy_thread_delegates() -> None:
    builder = _FakeBuilder()
    saver = LazyPostgresSaver(dsn="postgresql://x:5432/y", builder=builder)
    saver.copy_thread("src", "tgt")
    assert builder.last_inner is not None
    call = builder.last_inner.calls[0]
    assert call.name == "copy_thread"
    assert call.args == ("src", "tgt")


def test_sync_prune_delegates() -> None:
    builder = _FakeBuilder()
    saver = LazyPostgresSaver(dsn="postgresql://x:5432/y", builder=builder)
    saver.prune(["t1"], strategy="delete")
    assert builder.last_inner is not None
    call = builder.last_inner.calls[0]
    assert call.name == "prune"
    assert call.args == (["t1"],)
    assert call.kwargs == {"strategy": "delete"}


# ---------------------------------------------------------------------------
# Async delegation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_aget_tuple_delegates_to_sync() -> None:
    """Async aget_tuple proxies the SYNC inner.get_tuple — the upstream
    PostgresSaver doesn't implement aget_tuple natively (it inherits the
    BaseCheckpointSaver NotImplementedError stub). Tech debt: migrate
    to AsyncPostgresSaver for native async DB I/O.
    """
    builder = _FakeBuilder()
    saver = LazyPostgresSaver(dsn="postgresql://x:5432/y", builder=builder)

    cfg = {"configurable": {"thread_id": "t:w:r"}}
    result = await saver.aget_tuple(cfg)
    assert result == "sentinel_tuple"  # sync sentinel from _FakeInnerSaver.get_tuple
    assert builder.last_inner is not None
    call = builder.last_inner.calls[0]
    assert call.name == "get_tuple"
    assert call.args == (cfg,)


@pytest.mark.asyncio
async def test_async_aput_delegates_to_sync() -> None:
    builder = _FakeBuilder()
    saver = LazyPostgresSaver(dsn="postgresql://x:5432/y", builder=builder)

    cfg = {"configurable": {"thread_id": "t:w:r"}}
    result = await saver.aput(cfg, {"v": 1, "id": "cp"}, {"source": "loop"}, {"ch": 2})
    assert result == "sentinel_put_cfg"  # sync sentinel from _FakeInnerSaver.put
    assert builder.last_inner is not None
    call = builder.last_inner.calls[0]
    assert call.name == "put"
    assert call.args == (cfg, {"v": 1, "id": "cp"}, {"source": "loop"}, {"ch": 2})


@pytest.mark.asyncio
async def test_async_aput_writes_delegates_to_sync() -> None:
    builder = _FakeBuilder()
    saver = LazyPostgresSaver(dsn="postgresql://x:5432/y", builder=builder)

    cfg = {"configurable": {"thread_id": "t:w:r"}}
    writes = [("ch", "v")]
    await saver.aput_writes(cfg, writes, "task-1", "/")

    assert builder.last_inner is not None
    call = builder.last_inner.calls[0]
    assert call.name == "put_writes"
    assert call.args == (cfg, writes, "task-1", "/")


@pytest.mark.asyncio
async def test_async_alist_delegates_to_sync() -> None:
    """alist on the proxy iterates the SYNC inner.list and yields through
    into the async generator surface. Required because sync PostgresSaver
    has no async alist."""
    builder = _FakeBuilder()
    saver = LazyPostgresSaver(dsn="postgresql://x:5432/y", builder=builder)

    cfg = {"configurable": {"thread_id": "t:w:r"}}
    collected: list[Any] = []
    async for item in saver.alist(cfg, limit=10):
        collected.append(item)

    assert collected == ["sentinel_list_item"]  # sync sentinel from _FakeInnerSaver.list
    assert builder.last_inner is not None
    call_names = [c.name for c in builder.last_inner.calls]
    assert "list" in call_names


# ---------------------------------------------------------------------------
# Property delegation
# ---------------------------------------------------------------------------


def test_config_specs_delegates_when_opened() -> None:
    builder = _FakeBuilder()
    saver = LazyPostgresSaver(dsn="postgresql://x:5432/y", builder=builder)
    saver.open()
    assert saver.config_specs == ["inner_spec"]


def test_config_specs_empty_before_open() -> None:
    """Before open the proxy falls back to the BaseCheckpointSaver default (empty)."""
    builder = _FakeBuilder()
    saver = LazyPostgresSaver(dsn="postgresql://x:5432/y", builder=builder)
    # Must not trigger open() — config_specs is a probe-friendly accessor.
    assert saver.config_specs == []
    assert builder.opened_count == 0


# ---------------------------------------------------------------------------
# Lifecycle invariants preserved across the refactor
# ---------------------------------------------------------------------------


def test_open_close_idempotent() -> None:
    builder = _FakeBuilder()
    saver = LazyPostgresSaver(dsn="postgresql://x:5432/y", builder=builder)

    saver.open()
    saver.open()  # idempotent
    assert builder.opened_count == 1
    assert saver.opened is True

    saver.close()
    saver.close()  # idempotent
    assert builder.exited_count == 1
    assert saver.opened is False


def test_explicit_open_returns_inner() -> None:
    builder = _FakeBuilder()
    saver = LazyPostgresSaver(dsn="postgresql://x:5432/y", builder=builder)
    inner = saver.open()
    assert isinstance(inner, _FakeInnerSaver)


def test_close_clears_inner_state() -> None:
    builder = _FakeBuilder()
    saver = LazyPostgresSaver(dsn="postgresql://x:5432/y", builder=builder)
    saver.open()
    assert saver._inner is not None
    saver.close()
    assert saver._inner is None
    assert saver._cm is None


# ---------------------------------------------------------------------------
# Compile-with-LangGraph smoke (the actual failure mode that drove the fix).
# ---------------------------------------------------------------------------


def test_ensure_valid_checkpointer_accepts_lazy_saver() -> None:
    """The exact callsite that 503'd the orchestrator pod must now succeed."""
    from langgraph.types import ensure_valid_checkpointer

    saver = LazyPostgresSaver(dsn="postgresql://x:5432/y", builder=_FakeBuilder())
    # Pre-fix this raised TypeError. Post-fix it returns the saver unchanged.
    out = ensure_valid_checkpointer(saver)
    assert out is saver


# ---------------------------------------------------------------------------
# Async tail-surface coverage (adelete_*, acopy_thread, aprune)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_tail_surface_delegates() -> None:
    """`adelete_thread`/`adelete_for_runs`/`acopy_thread`/`aprune` all delegate.

    These are rarely-called methods on `BaseCheckpointSaver` but we
    explicitly override them on the proxy so the MRO does NOT resolve
    to the base class's `raise NotImplementedError`.
    """

    @dataclass
    class _TailInner:
        called: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = field(default_factory=list)

        # Sync surface — async proxy now delegates here per
        # LazyPostgresSaver tech debt (upstream sync PostgresSaver does
        # not implement the async tail methods).
        def delete_thread(self, thread_id: str) -> None:
            self.called.append(("delete_thread", (thread_id,), {}))

        def delete_for_runs(self, run_ids: Sequence[str]) -> None:
            self.called.append(("delete_for_runs", (run_ids,), {}))

        def copy_thread(self, src: str, tgt: str) -> None:
            self.called.append(("copy_thread", (src, tgt), {}))

        def prune(self, thread_ids: Sequence[str], *, strategy: str) -> None:
            self.called.append(("prune", (thread_ids,), {"strategy": strategy}))

    class _TailBuilder:
        def __init__(self) -> None:
            self.inner = _TailInner()

        @contextlib.contextmanager
        def from_conn_string(self, dsn: str) -> Any:
            try:
                yield self.inner
            finally:
                pass

    builder = _TailBuilder()
    saver = LazyPostgresSaver(dsn="postgresql://x:5432/y", builder=builder)

    await saver.adelete_thread("th-1")
    await saver.adelete_for_runs(["r1"])
    await saver.acopy_thread("src-th", "tgt-th")
    await saver.aprune(["th-2"], strategy="delete")

    assert [c[0] for c in builder.inner.called] == [
        "delete_thread",
        "delete_for_runs",
        "copy_thread",
        "prune",
    ]


# ---------------------------------------------------------------------------
# get_next_version — opened path delegates; unopened falls back to base
# ---------------------------------------------------------------------------


def test_get_next_version_unopened_uses_base_default() -> None:
    builder = _FakeBuilder()
    saver = LazyPostgresSaver(dsn="postgresql://x:5432/y", builder=builder)

    # Unopened: BaseCheckpointSaver default scheme. Current None -> 1.
    assert saver.get_next_version(None, None) == 1
    assert saver.get_next_version(7, None) == 8
    # Probing should not have triggered an open.
    assert builder.opened_count == 0


def test_get_next_version_opened_delegates() -> None:
    @dataclass
    class _VersionInner:
        called_with: list[Any] = field(default_factory=list)

        def get_next_version(self, current: int | None, channel: None) -> int:
            self.called_with.append((current, channel))
            return 999

    class _VersionBuilder:
        def __init__(self) -> None:
            self.inner = _VersionInner()

        @contextlib.contextmanager
        def from_conn_string(self, dsn: str) -> Any:
            try:
                yield self.inner
            finally:
                pass

    builder = _VersionBuilder()
    saver = LazyPostgresSaver(dsn="postgresql://x:5432/y", builder=builder)
    saver.open()
    assert saver.get_next_version(42, None) == 999
    assert builder.inner.called_with == [(42, None)]


# ---------------------------------------------------------------------------
# Stale-connection self-heal (CHO-2341 defect 1). The checkpointer must survive
# a Cloud SQL cost-pause/resume the same way ReconnectingAsyncConnection does
# for the outbox path: treat `.broken` (not just `.closed`) as dead, AND after
# an idle gap actively probe (SELECT 1) so the FIRST borrow following a silent
# server-side close reopens instead of throwing "server closed the connection
# unexpectedly" and forcing a rollout restart.
# ---------------------------------------------------------------------------


class _OperationalError(Exception):
    """Stand-in for psycopg.OperationalError (the probe catches broad Exception)."""


@dataclass
class _FakeClock:
    """Injectable monotonic clock for deterministic idle-gap tests."""

    t: float = 1000.0

    def now(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


class _FakeCursorCM:
    """A cursor context manager whose execute() can simulate a dead socket."""

    def __init__(self, conn: _FakeRawConn) -> None:
        self._conn = conn

    def __enter__(self) -> _FakeCursorCM:
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False

    def execute(self, sql: str, *args: Any, **kwargs: Any) -> None:
        self._conn.executed.append(sql)
        if self._conn.select_fails:
            raise _OperationalError("server closed the connection unexpectedly")


@dataclass
class _FakeRawConn:
    """Mimics the raw psycopg connection PostgresSaver exposes as `.conn`.

    `closed`/`broken` mirror psycopg3 flags; `select_fails` simulates the silent
    server-side close where BOTH flags stay clean until the first op hits the
    dead socket.
    """

    closed: int = 0
    broken: bool = False
    select_fails: bool = False
    executed: list[str] = field(default_factory=list)

    def cursor(self) -> _FakeCursorCM:
        return _FakeCursorCM(self)


@dataclass
class _ConnInnerSaver(_FakeInnerSaver):
    """Fake inner saver exposing a raw `.conn` that, like the real PostgresSaver,
    raises on get_tuple once its connection is dead."""

    conn: _FakeRawConn = field(default_factory=_FakeRawConn)

    def _dead(self) -> bool:
        return bool(self.conn.closed) or bool(self.conn.broken) or bool(self.conn.select_fails)

    def get_tuple(self, *args: Any, **kwargs: Any) -> Any:
        if self._dead():
            raise _OperationalError("server closed the connection unexpectedly")
        return super().get_tuple(*args, **kwargs)


class _SeqBuilder:
    """Builder that yields a pre-seeded SEQUENCE of inners across reopens, so a
    test can hand the proxy a dead inner first and a fresh one after reopen."""

    def __init__(self, inners: Sequence[Any]) -> None:
        self._inners = list(inners)
        self.opened_count = 0
        self.exited_count = 0
        self.last_inner: Any = None

    @contextlib.contextmanager
    def from_conn_string(self, dsn: str) -> Any:
        idx = min(self.opened_count, len(self._inners) - 1)
        inner = self._inners[idx]
        self.opened_count += 1
        self.last_inner = inner
        try:
            yield inner
        finally:
            self.exited_count += 1


_CFG = {"configurable": {"thread_id": "t:u"}}


def test_ensure_open_reopens_when_raw_conn_broken() -> None:
    """`.broken` (a lost socket after a failing op, `.closed` still 0) must be
    treated as dead and trigger a reopen. Pre-fix `_is_inner_live` checked
    `.closed` only, so the dead inner was handed back and get_tuple threw."""
    clock = _FakeClock()
    raw = _FakeRawConn(closed=0, broken=False)
    dead = _ConnInnerSaver(conn=raw)
    fresh = _ConnInnerSaver(conn=_FakeRawConn())
    builder = _SeqBuilder([dead, fresh])
    # Huge idle window isolates the `.broken` check from the active probe.
    saver = LazyPostgresSaver(
        dsn="postgresql://x:5432/y",
        builder=builder,
        liveness_idle_seconds=1e9,
        monotonic=clock.now,
    )

    assert saver.get_tuple(_CFG) == "sentinel_tuple"  # opens `dead`, healthy
    assert builder.opened_count == 1

    raw.broken = True  # a prior op left the socket unusable

    result = saver.get_tuple(_CFG)  # must detect broken -> reopen to `fresh`
    assert builder.opened_count == 2
    assert result == "sentinel_tuple"
    assert [c.name for c in fresh.calls] == ["get_tuple"]


def test_ensure_open_reopens_when_raw_conn_closed() -> None:
    """Regression guard: the existing `.closed` detection still reopens."""
    clock = _FakeClock()
    raw = _FakeRawConn(closed=0)
    dead = _ConnInnerSaver(conn=raw)
    fresh = _ConnInnerSaver(conn=_FakeRawConn())
    builder = _SeqBuilder([dead, fresh])
    saver = LazyPostgresSaver(
        dsn="postgresql://x:5432/y",
        builder=builder,
        liveness_idle_seconds=1e9,
        monotonic=clock.now,
    )

    saver.get_tuple(_CFG)
    assert builder.opened_count == 1
    raw.closed = 1
    saver.get_tuple(_CFG)
    assert builder.opened_count == 2


def test_idle_probe_reopens_silently_dead_conn() -> None:
    """A silent server-side close leaves `.closed`/`.broken` clean until first
    use. After more idle than the threshold, the proxy must actively probe
    (SELECT 1) and reopen on failure so the FIRST borrow succeeds."""
    clock = _FakeClock(t=1000.0)
    raw = _FakeRawConn(closed=0, broken=False, select_fails=False)
    silently_dead = _ConnInnerSaver(conn=raw)
    fresh = _ConnInnerSaver(conn=_FakeRawConn())
    builder = _SeqBuilder([silently_dead, fresh])
    saver = LazyPostgresSaver(
        dsn="postgresql://x:5432/y",
        builder=builder,
        liveness_idle_seconds=30.0,
        monotonic=clock.now,
    )

    assert saver.get_tuple(_CFG) == "sentinel_tuple"  # healthy borrow @ t=1000
    assert builder.opened_count == 1

    raw.select_fails = True  # socket silently dies during the idle gap
    clock.advance(31.0)

    result = saver.get_tuple(_CFG)  # probe fails -> reopen to `fresh`
    assert "SELECT 1" in raw.executed  # the probe actually ran
    assert builder.opened_count == 2
    assert result == "sentinel_tuple"
    assert [c.name for c in fresh.calls] == ["get_tuple"]


@pytest.mark.asyncio
async def test_idle_probe_reopens_silently_dead_conn_async() -> None:
    """The production call site is async aget_tuple (weakness HITL resume). It
    routes through the same `_ensure_open`, so the probe+reopen must apply."""
    clock = _FakeClock(t=5000.0)
    raw = _FakeRawConn()
    silently_dead = _ConnInnerSaver(conn=raw)
    fresh = _ConnInnerSaver(conn=_FakeRawConn())
    builder = _SeqBuilder([silently_dead, fresh])
    saver = LazyPostgresSaver(
        dsn="postgresql://x:5432/y",
        builder=builder,
        liveness_idle_seconds=30.0,
        monotonic=clock.now,
    )

    assert await saver.aget_tuple(_CFG) == "sentinel_tuple"
    assert builder.opened_count == 1

    raw.select_fails = True
    clock.advance(45.0)

    result = await saver.aget_tuple(_CFG)
    assert "SELECT 1" in raw.executed
    assert builder.opened_count == 2
    assert result == "sentinel_tuple"
    assert [c.name for c in fresh.calls] == ["get_tuple"]


def test_no_probe_within_idle_window() -> None:
    """Hot path: borrows within the idle window never probe (no SELECT 1) and
    never reopen. Guards against a per-op probe roundtrip regression."""
    clock = _FakeClock(t=1000.0)
    raw = _FakeRawConn()
    inner = _ConnInnerSaver(conn=raw)
    builder = _SeqBuilder([inner])
    saver = LazyPostgresSaver(
        dsn="postgresql://x:5432/y",
        builder=builder,
        liveness_idle_seconds=30.0,
        monotonic=clock.now,
    )

    saver.get_tuple(_CFG)  # @ t=1000
    clock.advance(5.0)  # still within the window
    saver.get_tuple(_CFG)  # @ t=1005

    assert raw.executed == []  # probe never ran
    assert builder.opened_count == 1


def test_probe_skipped_when_inner_has_no_raw_conn() -> None:
    """Introspection-less inners (InMemorySaver, plain fakes) have no `.conn`;
    the proxy must assume-live and never crash trying to probe."""
    clock = _FakeClock(t=1000.0)
    builder = _FakeBuilder()  # yields plain _FakeInnerSaver (no .conn)
    saver = LazyPostgresSaver(
        dsn="postgresql://x:5432/y",
        builder=builder,
        liveness_idle_seconds=30.0,
        monotonic=clock.now,
    )

    saver.get_tuple(_CFG)  # @ t=1000
    clock.advance(120.0)  # well past the window
    saver.get_tuple(_CFG)  # no raw conn to probe -> assume live, no reopen

    assert builder.opened_count == 1
