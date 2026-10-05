"""Unit tests for the write methods of ``PostgresPromptOverrideRepository``
(ADR-197 M-C.1 promotion write paths).

Exercised against a fake psycopg ``AsyncConnection`` (no live DB) that records
the executed SQL + params and exposes a configurable ``rowcount`` so we can
assert the guarded-UPDATE fail-loud behaviour.

Covered:
  * create_draft_plan — mints a UUIDv7 plan_id, INSERTs status='draft'.
  * add_segment — INSERTs a segment row, derives content_hash = sha256(body)
    when none is supplied.
  * update_plan_status — guarded UPDATE (WHERE status=expected_from); raises
    fail-loud when 0 rows match (illegal/stale transition).
  * activate_plan — ATOMIC archive-prior-then-activate inside ONE transaction;
    fail-loud when the activate UPDATE matches 0 rows.
"""

from __future__ import annotations

import hashlib
from typing import Any
from uuid import UUID

import pytest

from chora_ai_kernel_orchestrator.adapter.pg.prompt_override_repository import (
    PostgresPromptOverrideRepository,
)
from chora_ai_kernel_orchestrator.domain.prompt_registry.state_machine import (
    PromptPlanState,
)

# -----------------------------------------------------------------------------
# Fake async connection / cursor
# -----------------------------------------------------------------------------


class _FakeCursor:
    def __init__(self, owner: _FakeConn) -> None:
        self._owner = owner
        self.rowcount = -1

    async def __aenter__(self) -> _FakeCursor:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def execute(self, sql: str, params: Any = None) -> None:
        self._owner.executed.append((sql, params))
        # set_config GUC calls always "succeed"; data statements report the
        # configured rowcount so guarded UPDATEs can be exercised.
        if "set_config" in sql:
            self.rowcount = 1
        else:
            self.rowcount = self._owner.default_rowcount

    async def fetchall(self) -> list[tuple[Any, ...]]:
        return []


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
    def __init__(self, default_rowcount: int = 1) -> None:
        self.executed: list[tuple[str, Any]] = []
        self.default_rowcount = default_rowcount
        self.txn_entered = 0
        self.txn_exited = 0
        self._cur = _FakeCursor(self)

    def transaction(self) -> _FakeTxn:
        return _FakeTxn(self)

    def cursor(self) -> _FakeCursor:
        return self._cur


def _data_statements(conn: _FakeConn) -> list[tuple[str, Any]]:
    """Executed statements excluding set_config GUC binds."""
    return [(s, p) for (s, p) in conn.executed if "set_config" not in s]


# -----------------------------------------------------------------------------
# create_draft_plan
# -----------------------------------------------------------------------------


class TestCreateDraftPlan:
    @pytest.mark.asyncio
    async def test_returns_uuidv7_plan_id_and_inserts_draft(self) -> None:
        conn = _FakeConn()
        repo = PostgresPromptOverrideRepository(conn=conn)

        plan_id = await repo.create_draft_plan(
            plan_code="qgen-tone-tweak",
            scope="tenant",
            tenant_id="01970000-0000-7000-8000-000000000001",
            created_by="01970000-0000-7000-9000-000000000001",
        )

        # plan_id is a real UUID (v7 minted in app code).
        assert UUID(plan_id).version == 7
        # one transaction wrapped the work + tenant GUC bound for RLS.
        assert conn.txn_entered == 1
        assert any("set_config('chora.tenant_id'" in s for s, _ in conn.executed)

        data = _data_statements(conn)
        insert_sql, params = data[-1]
        assert "INSERT INTO prompt_override_plan" in insert_sql
        assert params["plan_id"] == plan_id
        assert params["plan_code"] == "qgen-tone-tweak"
        assert params["scope"] == "tenant"
        assert params["tenant_id"] == "01970000-0000-7000-8000-000000000001"
        assert params["created_by"] == "01970000-0000-7000-9000-000000000001"
        # status is forced to 'draft' (literal in SQL or bound param).
        assert "draft" in insert_sql or params.get("status") == "draft"

    @pytest.mark.asyncio
    async def test_platform_scope_stores_null_tenant(self) -> None:
        conn = _FakeConn()
        repo = PostgresPromptOverrideRepository(conn=conn)

        await repo.create_draft_plan(
            plan_code="platform-default-v2",
            scope="platform",
            tenant_id=None,
            created_by="01970000-0000-7000-9000-000000000001",
        )
        _, params = _data_statements(conn)[-1]
        assert params["tenant_id"] is None

    @pytest.mark.asyncio
    async def test_rejects_unknown_scope(self) -> None:
        repo = PostgresPromptOverrideRepository(conn=_FakeConn())
        with pytest.raises(ValueError, match="scope"):
            await repo.create_draft_plan(
                plan_code="x",
                scope="global",  # not platform|tenant
                tenant_id=None,
                created_by="g",
            )

    @pytest.mark.asyncio
    async def test_rejects_blank_plan_code(self) -> None:
        repo = PostgresPromptOverrideRepository(conn=_FakeConn())
        with pytest.raises(ValueError, match="plan_code"):
            await repo.create_draft_plan(
                plan_code="  ",
                scope="platform",
                tenant_id=None,
                created_by="g",
            )


# -----------------------------------------------------------------------------
# add_segment
# -----------------------------------------------------------------------------


class TestAddSegment:
    @pytest.mark.asyncio
    async def test_inserts_segment_and_returns_row_id(self) -> None:
        conn = _FakeConn()
        repo = PostgresPromptOverrideRepository(conn=conn)

        row_id = await repo.add_segment(
            plan_id="01970000-0000-7000-8000-0000000000aa",
            agent_id="qgen_question",
            segment_id="role",
            body="You are a meticulous question author.",
            content_hash="deadbeef",
            note="tighten tone",
        )
        assert UUID(row_id).version == 7
        _, params = _data_statements(conn)[-1]
        assert params["plan_id"] == "01970000-0000-7000-8000-0000000000aa"
        assert params["agent_id"] == "qgen_question"
        assert params["segment_id"] == "role"
        assert params["body"] == "You are a meticulous question author."
        assert params["content_hash"] == "deadbeef"
        assert params["note"] == "tighten tone"

    @pytest.mark.asyncio
    async def test_derives_content_hash_from_body_when_absent(self) -> None:
        conn = _FakeConn()
        repo = PostgresPromptOverrideRepository(conn=conn)

        body = "Generate exactly one MCQ."
        await repo.add_segment(
            plan_id="01970000-0000-7000-8000-0000000000aa",
            agent_id="qgen_question",
            segment_id="task",
            body=body,
        )
        _, params = _data_statements(conn)[-1]
        expected = hashlib.sha256(body.encode("utf-8")).hexdigest()
        assert params["content_hash"] == expected

    @pytest.mark.asyncio
    async def test_rejects_blank_body(self) -> None:
        repo = PostgresPromptOverrideRepository(conn=_FakeConn())
        with pytest.raises(ValueError, match="body"):
            await repo.add_segment(
                plan_id="p",
                agent_id="qgen_question",
                segment_id="role",
                body="   ",
            )


# -----------------------------------------------------------------------------
# update_plan_status
# -----------------------------------------------------------------------------


class TestUpdatePlanStatus:
    @pytest.mark.asyncio
    async def test_guarded_update_uses_expected_from(self) -> None:
        conn = _FakeConn(default_rowcount=1)
        repo = PostgresPromptOverrideRepository(conn=conn)

        await repo.update_plan_status(
            plan_id="01970000-0000-7000-8000-0000000000aa",
            expected_from=PromptPlanState.DRAFT,
            to=PromptPlanState.PENDING_EVAL,
        )
        sql, params = _data_statements(conn)[-1]
        assert "UPDATE prompt_override_plan" in sql
        # the WHERE clause guards on the expected current status.
        assert "status" in sql
        assert params["plan_id"] == "01970000-0000-7000-8000-0000000000aa"
        assert params["expected_from"] == "draft"
        assert params["to"] == "pending_eval"

    @pytest.mark.asyncio
    async def test_fail_loud_when_zero_rows_match(self) -> None:
        # 0 rows = the plan is not in expected_from (illegal/stale transition).
        conn = _FakeConn(default_rowcount=0)
        repo = PostgresPromptOverrideRepository(conn=conn)
        with pytest.raises(RuntimeError, match="0 rows|stale|expected"):
            await repo.update_plan_status(
                plan_id="missing",
                expected_from=PromptPlanState.PENDING_EVAL,
                to=PromptPlanState.PENDING_HITL,
            )

    @pytest.mark.asyncio
    async def test_eval_run_id_bound_when_supplied(self) -> None:
        conn = _FakeConn(default_rowcount=1)
        repo = PostgresPromptOverrideRepository(conn=conn)
        await repo.update_plan_status(
            plan_id="p",
            expected_from=PromptPlanState.PENDING_EVAL,
            to=PromptPlanState.PENDING_HITL,
            eval_run_id="eval-run-42",
        )
        _, params = _data_statements(conn)[-1]
        assert params["eval_run_id"] == "eval-run-42"


# -----------------------------------------------------------------------------
# activate_plan
# -----------------------------------------------------------------------------


class TestActivatePlan:
    @pytest.mark.asyncio
    async def test_archives_prior_then_activates_in_one_transaction(self) -> None:
        conn = _FakeConn(default_rowcount=1)
        repo = PostgresPromptOverrideRepository(conn=conn)

        await repo.activate_plan(
            plan_id="01970000-0000-7000-8000-0000000000aa",
            scope="tenant",
            tenant_id="01970000-0000-7000-8000-000000000001",
            approved_by="01970000-0000-7000-9000-000000000001",
        )
        # ONE atomic transaction around both UPDATEs.
        assert conn.txn_entered == 1
        assert conn.txn_exited == 1

        data = _data_statements(conn)
        # First data statement archives the prior active plan; second activates.
        archive_sql, archive_params = data[0]
        activate_sql, activate_params = data[1]
        assert "UPDATE prompt_override_plan" in archive_sql
        assert "archived" in archive_sql
        assert "active" in archive_sql  # WHERE status = 'active'
        assert "UPDATE prompt_override_plan" in activate_sql
        assert "active" in activate_sql
        assert activate_params["plan_id"] == "01970000-0000-7000-8000-0000000000aa"
        assert activate_params["approved_by"] == "01970000-0000-7000-9000-000000000001"

    @pytest.mark.asyncio
    async def test_sets_tenant_guc(self) -> None:
        conn = _FakeConn(default_rowcount=1)
        repo = PostgresPromptOverrideRepository(conn=conn)
        await repo.activate_plan(
            plan_id="p",
            scope="tenant",
            tenant_id="01970000-0000-7000-8000-000000000001",
            approved_by="g",
        )
        assert any("set_config('chora.tenant_id'" in s for s, _ in conn.executed)

    @pytest.mark.asyncio
    async def test_fail_loud_when_activate_matches_no_pending_hitl_row(self) -> None:
        # 0 rows on the activate UPDATE = plan was not pending_hitl (stale).
        conn = _FakeConn(default_rowcount=0)
        repo = PostgresPromptOverrideRepository(conn=conn)
        with pytest.raises(RuntimeError, match="0 rows|pending_hitl|activate"):
            await repo.activate_plan(
                plan_id="p",
                scope="platform",
                tenant_id=None,
                approved_by="g",
            )
