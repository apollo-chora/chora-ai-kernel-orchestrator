"""Tests for the shared run-id helper."""

from __future__ import annotations

from chora_ai_kernel_orchestrator.domain.state import new_request_id


def test_new_request_id_returns_uuidv7_string() -> None:
    """new_request_id returns a UUIDv7 (version nibble = 7)."""
    rid = new_request_id()
    assert isinstance(rid, str)
    assert len(rid) == 36
    # Version-7 UUIDs have the digit 7 in the version position (chars[14])
    assert rid[14] == "7"


def test_new_request_id_uniqueness() -> None:
    """Two consecutive new_request_id calls must differ."""
    a = new_request_id()
    b = new_request_id()
    assert a != b
