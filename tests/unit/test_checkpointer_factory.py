"""Tests for the PostgresSaver factory + thread_id helper (S3.4 P0 RED).

Per `langgraph-orchestrator-python` skill the checkpointer thread_id is
formed as `{tenant_id}:{workflow_id}:{run_id}` so checkpoints isolate by
tenant + workflow.
"""

from __future__ import annotations

import os

import pytest


def test_thread_id_format() -> None:
    from chora_ai_kernel_orchestrator.adapter.checkpointer.factory import (
        build_thread_id,
    )

    tid = build_thread_id(
        tenant_id="00000000-0000-7000-8000-000000000001",
        workflow_id="ai-assist",
        run_id="01975c83-0000-7000-8000-000000000000",
    )
    assert tid == "00000000-0000-7000-8000-000000000001:ai-assist:01975c83-0000-7000-8000-000000000000"


def test_weakness_thread_id_is_two_segment_no_run_id() -> None:
    """CHO-1973 Wave A: the Growth-Edge crew thread is DETERMINISTIC on
    {tenant}:{upload} ONLY — one analysis per upload, reiterate stays in-thread,
    so the FE resume needs no run_id. (build_thread_id stays 3-segment for
    qgen/OE/ai-assist — they are unaffected.)"""
    from chora_ai_kernel_orchestrator.adapter.checkpointer.factory import (
        build_weakness_thread_id,
    )

    tid = build_weakness_thread_id(
        tenant_id="00000000-0000-7000-8000-000000000001",
        upload_id="0190aaaa-bbbb-7ccc-8ddd-eeeeeeeeeeee",
    )
    assert tid == "00000000-0000-7000-8000-000000000001:0190aaaa-bbbb-7ccc-8ddd-eeeeeeeeeeee"
    # exactly two colon-separated segments (no run_id tail)
    assert tid.count(":") == 1


def test_weakness_thread_id_rejects_blank_segments() -> None:
    from chora_ai_kernel_orchestrator.adapter.checkpointer.factory import (
        build_weakness_thread_id,
    )

    with pytest.raises(ValueError):
        build_weakness_thread_id(tenant_id="", upload_id="u")
    with pytest.raises(ValueError):
        build_weakness_thread_id(tenant_id="t", upload_id="   ")


def test_weakness_thread_id_exported_from_package() -> None:
    from chora_ai_kernel_orchestrator.adapter.checkpointer import (
        build_weakness_thread_id,
    )

    assert build_weakness_thread_id(tenant_id="t", upload_id="u") == "t:u"


def test_factory_returns_inmemory_when_dsn_missing() -> None:
    """No env var -> in-memory checkpointer (dev fallback)."""
    from langgraph.checkpoint.memory import InMemorySaver

    from chora_ai_kernel_orchestrator.adapter.checkpointer.factory import (
        build_checkpointer_from_env,
    )

    os.environ.pop("CHORA_AI_KERNEL_PG_DSN", None)
    saver = build_checkpointer_from_env()
    assert isinstance(saver, InMemorySaver)


def test_factory_returns_postgres_saver_when_dsn_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """When DSN env is set, the factory builds a PostgresSaver (no connect yet)."""
    monkeypatch.setenv("CHORA_AI_KERNEL_PG_DSN", "postgresql://localhost:5432/chora_ai_kernel")

    from chora_ai_kernel_orchestrator.adapter.checkpointer.factory import (
        build_checkpointer_from_env,
    )

    saver = build_checkpointer_from_env()
    # We only verify class identity — actual setup happens lazily so the
    # factory does not establish a TCP connection at import time.
    cls_name = saver.__class__.__name__
    assert "PostgresSaver" in cls_name


def test_factory_validates_dsn_blank_string() -> None:
    """Empty / whitespace DSN -> in-memory fallback (defensive)."""
    from langgraph.checkpoint.memory import InMemorySaver

    from chora_ai_kernel_orchestrator.adapter.checkpointer.factory import (
        build_checkpointer_from_env,
    )

    os.environ["CHORA_AI_KERNEL_PG_DSN"] = "   "
    try:
        saver = build_checkpointer_from_env()
        assert isinstance(saver, InMemorySaver)
    finally:
        os.environ.pop("CHORA_AI_KERNEL_PG_DSN", None)
