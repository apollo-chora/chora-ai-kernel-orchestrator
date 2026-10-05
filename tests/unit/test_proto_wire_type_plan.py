"""proto_wire decoder tests — field 21 ``type_plan`` (mixed-type batch, CHO-1819).

``AiAssistStarted.type_plan`` (field 21, repeated ``GenerationTypeQuota``) is the
per-question-type quota plan for mixed MCQ+OE batches:

    message GenerationTypeQuota {
        string question_type = 1;
        int32  count         = 2;
        int32  max_images    = 3;
    }

EMPTY ⇒ legacy single-type path (no ``type_plan`` key — byte/behaviour-compatible
with the single/1a/1c shapes). NON-EMPTY ⇒ mixed batch. Round-trips against the
COMMITTED generated pb (the oracle), mirroring ``test_proto_wire_source_files``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from chora_ai_kernel_orchestrator.adapter.pubsub.proto_wire import (
    decode_ai_assist_started,
)

# --- Committed generated bindings (chora-contracts/gen/python) --------------
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


def test_round_trip_type_plan_against_committed_pb() -> None:
    pb = _load_ai_assist_pb2()

    msg = pb.AiAssistStartedV2()
    msg.assist_id = "job-mix"
    msg.tenant_id = "TEN"
    msg.author_gcid = "GCID"
    msg.content_type = "mixed"
    msg.prompt = "Generate from the attached source."
    msg.requested_count = 10
    msg.intent = "new_question"
    msg.grounding_mode = "strict"
    for qt, count, max_images in (("mcq", 8, 3), ("oe", 2, 1)):
        q = msg.type_plan.add()
        q.question_type = qt
        q.count = count
        q.max_images = max_images

    out = decode_ai_assist_started(msg.SerializeToString())

    # Order preserved (declaration order); each quota faithfully surfaced.
    assert out["type_plan"] == [
        {"question_type": "mcq", "count": 8, "max_images": 3},
        {"question_type": "oe", "count": 2, "max_images": 1},
    ]
    assert out["requested_count"] == 10
    assert out["content_type"] == "mixed"
    assert out["grounding_mode"] == "strict"


def test_type_plan_key_absent_on_legacy_single_type_message() -> None:
    # proto3 omits empty repeated fields — a legacy single-type job carries no
    # f21 bytes, so the key MUST stay absent (the single/batch single-type path
    # stays byte/behaviour-compatible).
    pb = _load_ai_assist_pb2()
    msg = pb.AiAssistStartedV2()
    msg.assist_id = "job-legacy"
    msg.content_type = "mcq"
    msg.requested_count = 5
    msg.intent = "new_question"

    out = decode_ai_assist_started(msg.SerializeToString())
    assert "type_plan" not in out


def test_type_plan_quota_defaults_zero_when_scalars_omitted() -> None:
    # A quota with only question_type set decodes count/max_images to 0 (proto3
    # scalar default). The decoder surfaces faithfully; fail-loud validation of
    # the invariants happens in the chora-creation domain + the graph, not here.
    pb = _load_ai_assist_pb2()
    msg = pb.AiAssistStartedV2()
    msg.assist_id = "j"
    msg.content_type = "mixed"
    q = msg.type_plan.add()
    q.question_type = "mcq"  # count + max_images left at 0

    out = decode_ai_assist_started(msg.SerializeToString())
    assert out["type_plan"] == [{"question_type": "mcq", "count": 0, "max_images": 0}]


def test_type_plan_skips_unknown_quota_subfield() -> None:
    # Forward-compat: a future GenerationTypeQuota field must be skipped without
    # corrupting the known fields. Hand-build a quota with an unknown field 9
    # (fields 1-5 are now all meaningful: question_type/count/max_images +
    # image_for_stem/image_for_answer).
    base = (
        b"\x12\x02AS"  # field 2 assist_id="AS"
        b"\x32\x05mixed"  # field 6 content_type="mixed"
    )
    # GenerationTypeQuota{ question_type="oe"(f1), count=3(f2), <unknown f9 str> }
    # field 9 length-delimited tag = (9<<3)|2 = 74 → 0x4A
    quota = b"\x0a\x02oe\x10\x03\x4a\x04NAME"
    # field 21 tag = (21<<3)|2 = 170 → varint 0xAA 0x01
    f21 = b"\xaa\x01" + bytes([len(quota)]) + quota

    out = decode_ai_assist_started(base + f21)
    assert out["type_plan"] == [{"question_type": "oe", "count": 3, "max_images": 0}]


def test_round_trip_type_plan_image_flags_against_committed_pb() -> None:
    # CHO-1825 — the per-type author image opt-ins image_for_stem (f4) /
    # image_for_answer (f5) on GenerationTypeQuota are deterministic toggles:
    # when set, EVERY question of that type must carry that image. They MUST
    # decode off the wire (varint bools) or the toggle is silently dropped and
    # the agent never forces the image.
    pb = _load_ai_assist_pb2()
    msg = pb.AiAssistStartedV2()
    msg.assist_id = "job-img"
    msg.content_type = "mixed"
    msg.requested_count = 3
    # mcq: stem image forced; oe: answer image forced; a third quota leaves both
    # off to prove proto3-omit-false keeps the legacy 3-key shape.
    q1 = msg.type_plan.add()
    q1.question_type, q1.count, q1.image_for_stem = "mcq", 2, True
    q2 = msg.type_plan.add()
    q2.question_type, q2.count, q2.image_for_answer = "oe", 1, True

    out = decode_ai_assist_started(msg.SerializeToString())
    assert out["type_plan"] == [
        {"question_type": "mcq", "count": 2, "max_images": 0, "image_for_stem": True},
        {"question_type": "oe", "count": 1, "max_images": 0, "image_for_answer": True},
    ]


def test_type_plan_image_flags_absent_when_off() -> None:
    # proto3 omits false scalars — an opt-in-off quota carries no bytes for
    # f4/f5, so the keys MUST stay absent (legacy mixed-batch shape unchanged).
    pb = _load_ai_assist_pb2()
    msg = pb.AiAssistStartedV2()
    msg.assist_id = "j"
    msg.content_type = "mixed"
    q = msg.type_plan.add()
    q.question_type, q.count, q.max_images = "mcq", 4, 2

    out = decode_ai_assist_started(msg.SerializeToString())
    assert out["type_plan"] == [{"question_type": "mcq", "count": 4, "max_images": 2}]
    assert "image_for_stem" not in out["type_plan"][0]
    assert "image_for_answer" not in out["type_plan"][0]
