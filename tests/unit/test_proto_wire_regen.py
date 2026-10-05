"""proto_wire decoder tests — field 22 ``regen`` (review image regenerate, CHO-1819 P3).

``AiAssistStarted.regen`` (field 22, singular ``ImageRegenSpec``) carries the
single-candidate image regenerate request:

    message ImageRegenSpec {
        string draft_id             = 1;
        string placement            = 2;   // "stem" | "answer"
        string prompt               = 3;
        string mode                 = 4;
        string current_stem         = 5;   // I2: author's edited context
        string current_model_answer = 6;
        string original_source      = 7;
    }

ABSENT ⇒ non-regen job (no ``regen`` key — byte/behaviour-compatible with the
single/batch shapes). PRESENT ⇒ the orchestrator routes job_kind="image_regen"
to the ImageRegenRunner. Round-trips against the COMMITTED generated pb (oracle),
mirroring ``test_proto_wire_type_plan``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from chora_ai_kernel_orchestrator.adapter.pubsub.proto_wire import (
    decode_ai_assist_started,
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


def test_round_trip_regen_against_committed_pb() -> None:
    pb = _load_ai_assist_pb2()

    msg = pb.AiAssistStartedV2()
    msg.assist_id = "job-regen"
    msg.tenant_id = "TEN"
    msg.author_gcid = "GCID"
    msg.content_type = "mcq"
    msg.prompt = "Regenerate the stem illustration."
    msg.intent = "image_regen"
    msg.regen.draft_id = "draft-7"
    msg.regen.placement = "stem"
    msg.regen.prompt = "A clearer diagram of the light-dependent reactions"
    msg.regen.mode = "replace"

    out = decode_ai_assist_started(msg.SerializeToString())

    assert out["regen"] == {
        "draft_id": "draft-7",
        "placement": "stem",
        "prompt": "A clearer diagram of the light-dependent reactions",
        "mode": "replace",
        # I2 edited-context fields default to "" when the producer omits them.
        "current_stem": "",
        "current_model_answer": "",
        "original_source": "",
        # ADR-210 f8 — image-to-image original; "" when the producer omits it.
        "original_image_gcs_uri": "",
    }
    assert out["intent"] == "image_regen"
    assert out["content_type"] == "mcq"


def test_round_trip_regen_edited_context_against_committed_pb() -> None:
    # I2 contract add — the regen spec carries the author's CURRENT edited
    # question context (current_stem f5 / current_model_answer f6 /
    # original_source f7) so the image regenerate prompt reflects unsaved edits.
    pb = _load_ai_assist_pb2()

    msg = pb.AiAssistStartedV2()
    msg.assist_id = "job-regen-ctx"
    msg.content_type = "mcq"
    msg.intent = "image_regen"
    msg.regen.draft_id = "draft-9"
    msg.regen.placement = "stem"
    msg.regen.prompt = "Match the edited stem."
    msg.regen.mode = "replace"
    msg.regen.current_stem = "What gas do plants release during photosynthesis?"
    msg.regen.current_model_answer = "Oxygen (O2)."
    msg.regen.original_source = "Biology textbook, ch.4."

    out = decode_ai_assist_started(msg.SerializeToString())

    assert out["regen"] == {
        "draft_id": "draft-9",
        "placement": "stem",
        "prompt": "Match the edited stem.",
        "mode": "replace",
        "current_stem": "What gas do plants release during photosynthesis?",
        "current_model_answer": "Oxygen (O2).",
        "original_source": "Biology textbook, ch.4.",
        "original_image_gcs_uri": "",
    }


def test_regen_key_absent_on_non_regen_message() -> None:
    # proto3 omits an unset singular message — a non-regen job carries no f22
    # bytes, so the key MUST stay absent (byte/behaviour-compatible).
    pb = _load_ai_assist_pb2()
    msg = pb.AiAssistStartedV2()
    msg.assist_id = "job-batch"
    msg.content_type = "mcq"
    msg.intent = "new_question"

    out = decode_ai_assist_started(msg.SerializeToString())
    assert "regen" not in out


def test_regen_defaults_empty_strings_when_subfields_omitted() -> None:
    # A spec with only draft_id set decodes placement/prompt/mode to "" (proto3
    # scalar default). Surfaced faithfully; invariant validation lives upstream.
    pb = _load_ai_assist_pb2()
    msg = pb.AiAssistStartedV2()
    msg.assist_id = "j"
    msg.regen.draft_id = "d1"  # placement / prompt / mode left at ""

    out = decode_ai_assist_started(msg.SerializeToString())
    assert out["regen"] == {
        "draft_id": "d1",
        "placement": "",
        "prompt": "",
        "mode": "",
        "current_stem": "",
        "current_model_answer": "",
        "original_source": "",
        "original_image_gcs_uri": "",
    }


def test_regen_skips_unknown_subfield() -> None:
    # Forward-compat: a future ImageRegenSpec field must be skipped without
    # corrupting the known fields. Hand-build a spec with an unknown field 9
    # (fields 1-8 are now defined: draft_id/placement/prompt/mode +
    # current_stem/current_model_answer/original_source/original_image_gcs_uri).
    base = (
        b"\x12\x02AS"  # field 2 assist_id="AS"
        b"\x32\x03mcq"  # field 6 content_type="mcq"
    )
    # ImageRegenSpec{ draft_id="d"(f1), placement="stem"(f2), <unknown f9 str> }
    # field 9 tag = (9<<3)|2 = 74 → 0x4a.
    spec = b"\x0a\x01d\x12\x04stem\x4a\x02NO"
    # field 22 tag = (22<<3)|2 = 178 → varint 0xB2 0x01
    f22 = b"\xb2\x01" + bytes([len(spec)]) + spec

    out = decode_ai_assist_started(base + f22)
    assert out["regen"] == {
        "draft_id": "d",
        "placement": "stem",
        "prompt": "",
        "mode": "",
        "current_stem": "",
        "current_model_answer": "",
        "original_source": "",
        "original_image_gcs_uri": "",
    }
