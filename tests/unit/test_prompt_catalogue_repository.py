"""CHO-2368 catalogue reads + the 0009 per-agent carve-out in the pg adapter.

Same fake-connection idiom as test_prompt_override_repository.py: assert on the
SQL text + bind params the adapter issues (no live Postgres in CI).
"""

from __future__ import annotations

from typing import Any

import pytest

from chora_ai_kernel_orchestrator.adapter.pg.prompt_override_repository import (
    PostgresPromptOverrideRepository,
)


class _FakeCursor:
    def __init__(self, owner: _FakeConn) -> None:
        self._owner = owner
        self.executed: list[tuple[str, Any]] = []
        self.rowcount = 1

    async def __aenter__(self) -> _FakeCursor:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def execute(self, sql: str, params: Any = None) -> None:
        self.executed.append((sql, params))

    async def fetchall(self) -> list[tuple[Any, ...]]:
        return self._owner.rows_queue.pop(0) if self._owner.rows_queue else []

    async def fetchone(self) -> tuple[Any, ...] | None:
        rows = await self.fetchall()
        return rows[0] if rows else None


class _FakeTxn:
    def __init__(self, owner: _FakeConn) -> None:
        self._owner = owner

    async def __aenter__(self) -> _FakeTxn:
        self._owner.txn_entered += 1
        return self

    async def __aexit__(self, *exc: object) -> None:
        self._owner.txn_exited += 1
        return None


class _FakeConn:
    def __init__(self, rows_queue: list[list[tuple[Any, ...]]] | None = None) -> None:
        self.cur = _FakeCursor(self)
        self.rows_queue = rows_queue or []
        self.txn_entered = 0
        self.txn_exited = 0

    def transaction(self) -> _FakeTxn:
        return _FakeTxn(self)

    def cursor(self) -> _FakeCursor:
        return self.cur


def _sql_of(conn: _FakeConn, index: int) -> str:
    return conn.cur.executed[index][0]


def _params_of(conn: _FakeConn, index: int) -> Any:
    return conn.cur.executed[index][1]


# ---------------------------------------------------------------------------
# 0009 carve-out on the existing paths
# ---------------------------------------------------------------------------


async def test_fetch_active_overrides_excludes_baselines_and_returns_version_label() -> None:
    conn = _FakeConn(rows_queue=[[("platform", "1.1.0", "role", "new role text")]])
    repo = PostgresPromptOverrideRepository(conn=conn)
    got = await repo.fetch_active_overrides(tenant_id="", agent_id="qgen_question")

    query_sql = _sql_of(conn, 1)
    assert "p.kind = 'override'" in query_sql, "baselines must never resolve as overrides"
    assert "version_label" in query_sql
    assert len(got) == 1
    assert got[0].version == "1.1.0"
    assert got[0].segments == {"role": "new role text"}


async def test_activate_plan_archives_only_the_same_agents_override_plans() -> None:
    conn = _FakeConn()
    repo = PostgresPromptOverrideRepository(conn=conn)
    await repo.activate_plan(
        plan_id="p-1", scope="platform", tenant_id=None, approved_by="gcid-a", agent_id="oe_evaluator"
    )
    archive_sql = _sql_of(conn, 1)
    import re

    assert re.search(r"kind\s+=\s+'override'", archive_sql), "archiving must never touch baselines"
    assert "agent_id IS NOT DISTINCT FROM %(agent_id)s" in archive_sql
    assert _params_of(conn, 1)["agent_id"] == "oe_evaluator"


async def test_create_draft_plan_binds_agent_id_and_version_label() -> None:
    conn = _FakeConn()
    repo = PostgresPromptOverrideRepository(conn=conn)
    await repo.create_draft_plan(
        plan_code="qgen_question-1.1.0",
        scope="platform",
        tenant_id=None,
        created_by="gcid-a",
        agent_id="qgen_question",
        version_label="1.1.0",
    )
    insert_params = _params_of(conn, 1)
    assert insert_params["agent_id"] == "qgen_question"
    assert insert_params["version_label"] == "1.1.0"
    assert "agent_id" in _sql_of(conn, 1) and "version_label" in _sql_of(conn, 1)


async def test_get_plan_returns_agent_id_and_version_label() -> None:
    conn = _FakeConn(rows_queue=[[("p-1", "platform", None, "pending_hitl", "qgen_critic", "1.1.0")]])
    repo = PostgresPromptOverrideRepository(conn=conn)
    plan = await repo.get_plan("p-1")
    assert plan is not None
    assert plan.agent_id == "qgen_critic"
    assert plan.version_label == "1.1.0"


# ---------------------------------------------------------------------------
# Catalogue reads (the new read API's data source)
# ---------------------------------------------------------------------------


async def test_list_agent_versions_shapes_rows() -> None:
    _t0 = "2026-07-27T00:00:00+00:00"
    _t1 = "2026-07-27T01:00:00+00:00"
    conn = _FakeConn(
        rows_queue=[
            [
                ("v1", "baseline", "active", _t0, None, None, _t0, "baseline-qgen_question-v1", "p-base"),
                ("1.1.0", "override", "draft", None, None, None, _t1, "qgen_question-1.1.0", "p-110"),
            ]
        ]
    )
    repo = PostgresPromptOverrideRepository(conn=conn)
    versions = await repo.list_agent_versions(agent_id="qgen_question")

    sql = _sql_of(conn, 1)
    assert "scope = 'platform'" in sql, "the catalogue lists platform plans only"
    assert _params_of(conn, 1)["agent_id"] == "qgen_question"
    assert [v.version_label for v in versions] == ["v1", "1.1.0"]
    assert versions[0].kind == "baseline"
    assert versions[1].status == "draft"
    assert versions[1].plan_id == "p-110"


async def test_list_agent_versions_requires_agent_id() -> None:
    repo = PostgresPromptOverrideRepository(conn=_FakeConn())
    with pytest.raises(ValueError):
        await repo.list_agent_versions(agent_id=" ")


async def test_get_agent_version_returns_plan_and_ordered_segments() -> None:
    _t0 = "2026-07-27T00:00:00+00:00"
    conn = _FakeConn(
        rows_queue=[
            [("v1", "baseline", "active", _t0, None, None, _t0, "baseline-familiar-v1", "p-base")],
            [
                ("context_frame", "body-a", True, 10, "hash-a", "note-a", 1),
                ("role_frame", "body-b", False, 20, "hash-b", "", 1),
            ],
        ]
    )
    repo = PostgresPromptOverrideRepository(conn=conn)
    got = await repo.get_agent_version(agent_id="familiar", version_label="v1")
    assert got is not None
    plan, segments = got
    assert plan.version_label == "v1"
    seg_sql = _sql_of(conn, 2)
    assert "position" in seg_sql and "locked" in seg_sql
    assert [s.segment_id for s in segments] == ["context_frame", "role_frame"]
    assert segments[0].locked is True
    assert segments[1].locked is False


async def test_get_agent_version_returns_none_for_unknown() -> None:
    conn = _FakeConn(rows_queue=[[]])
    repo = PostgresPromptOverrideRepository(conn=conn)
    got = await repo.get_agent_version(agent_id="familiar", version_label="9.9.9")
    assert got is None
