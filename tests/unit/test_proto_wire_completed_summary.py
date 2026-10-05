"""proto_wire_encoder tests — field 16 ``generation_summary`` + populated
``generated_count`` (mixed-type batch / honest strict shortfall, CHO-1819).

Encodes ``AiAssistCompleted`` with the hand-rolled encoder, parses the bytes
with the COMMITTED generated pb (the oracle), and asserts the canonical proto3
parser reads ``generation_summary`` + ``generated_count`` back correctly. This
is the strongest guarantee the Pub/Sub Schema Registry will accept the wire.
"""

from __future__ import annotations

import datetime as _dt
import sys
from pathlib import Path

import pytest

from chora_ai_kernel_orchestrator.adapter.pubsub.proto_wire_encoder import (
    encode_ai_assist_completed,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GEN_PYTHON = _REPO_ROOT / "testdata" / "chora_contracts_gen"


def _load_ai_assist_pb2():  # type: ignore[no-untyped-def]
    if not _GEN_PYTHON.is_dir():  # pragma: no cover — narrow build contexts only
        pytest.skip(f"chora_contracts_gen bindings not present at {_GEN_PYTHON}")
    if "chora_contracts_gen" not in sys.modules:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "chora_contracts_gen",
            _GEN_PYTHON / "__init__.py",
            submodule_search_locations=[str(_GEN_PYTHON)],
        )
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules["chora_contracts_gen"] = mod
        spec.loader.exec_module(mod)
    from chora_contracts_gen.events.creation import (  # type: ignore[import-not-found]
        ai_assist_pb2,
    )

    return ai_assist_pb2


def _envelope() -> dict[str, object]:
    now_iso = _dt.datetime(2026, 6, 21, 12, 0, 0, tzinfo=_dt.UTC).isoformat()
    return {
        "event_id": "01970000-aaaa-7000-8000-000000000099",
        "idempotency_key": "ai_assist.completed.assist-mix",
        "tenant_id": "01970000-0000-7000-8000-000000000001",
        "gcid": "01970000-0000-7000-9000-000000000001",
        "occurred_at": now_iso,
        "published_at": now_iso,
        "schema_version": "1",
        "chora_imda_dimension": "accountability",
    }


def test_completed_encodes_generation_summary_round_trip() -> None:
    pb = _load_ai_assist_pb2()
    body = {
        "assist_id": "assist-mix",
        "generated_count": 7,
        "candidate_payload_json": '{"candidates":[]}',
        "generation_summary": {
            "requested_total": 10,
            "generated_total": 7,
            "generated_per_type": {"mcq": 6, "oe": 1},
            "shortfall_reason": ("Source supported 6 distinct MCQ concepts and 1 OE prompt."),
        },
    }

    raw = encode_ai_assist_completed(_envelope(), body)

    msg = pb.AiAssistCompleted()
    msg.ParseFromString(raw)

    assert msg.assist_id == "assist-mix"
    assert msg.generated_count == 7
    assert msg.generation_summary.requested_total == 10
    assert msg.generation_summary.generated_total == 7
    assert dict(msg.generation_summary.generated_per_type) == {"mcq": 6, "oe": 1}
    assert msg.generation_summary.shortfall_reason.startswith("Source supported")


def test_completed_full_count_has_empty_shortfall_reason() -> None:
    # Full count produced ⇒ shortfall_reason MUST be empty (not a shortfall).
    pb = _load_ai_assist_pb2()
    body = {
        "assist_id": "assist-full",
        "generated_count": 10,
        "candidate_payload_json": '{"candidates":[]}',
        "generation_summary": {
            "requested_total": 10,
            "generated_total": 10,
            "generated_per_type": {"mcq": 8, "oe": 2},
            "shortfall_reason": "",
        },
    }

    raw = encode_ai_assist_completed(_envelope(), body)
    msg = pb.AiAssistCompleted()
    msg.ParseFromString(raw)

    assert msg.generated_count == 10
    assert msg.generation_summary.generated_total == 10
    assert msg.generation_summary.shortfall_reason == ""


def test_completed_omits_summary_when_absent_legacy_path() -> None:
    # Back-compat: a body without generation_summary (legacy single-candidate
    # path) must not crash and parses to the proto3-default empty summary.
    pb = _load_ai_assist_pb2()
    body = {
        "assist_id": "assist-legacy",
        "generated_count": 1,
        "candidate_payload_json": "{}",
    }

    raw = encode_ai_assist_completed(_envelope(), body)
    msg = pb.AiAssistCompleted()
    msg.ParseFromString(raw)

    assert msg.generated_count == 1
    assert msg.generation_summary.requested_total == 0
    assert msg.generation_summary.generated_total == 0
    assert msg.generation_summary.shortfall_reason == ""
