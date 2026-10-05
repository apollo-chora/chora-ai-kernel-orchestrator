"""proto_wire decoder tests — Lane 1c field 20 ``source_files`` (CHO-1703).

``AiAssistStarted.source_files`` (field 20, repeated ``SourceFileRef``) is the
canonical role-tagged grounding-file list for multi-file batch jobs:

    message SourceFileRef {
        string blob_uri  = 1;
        string mime_type = 2;
        string role      = 3;   // "source" | "rubric"
    }

Wire-format note: field 20 tag = (20 << 3) | 2 = 162 → varint ``0xA2 0x01``;
each occurrence is one length-delimited SourceFileRef submessage (strings at
1/2/3). proto3 omits empty repeated fields entirely, so a pre-1c message
carries no bytes for it and the decoded dict has NO ``source_files`` key —
byte/behaviour-compatible with the EPIC-1a single-file shape (f17/18).

Includes the COMMITTED-generated-pb round-trip mandated by the W2 brief:
serialize with ``chora-contracts/gen/python/events/creation/ai_assist_pb2``,
decode with ``proto_wire.decode_ai_assist_started``, assert equality.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from chora_ai_kernel_orchestrator.adapter.pubsub.proto_wire import (
    decode_ai_assist_started,
)

# --- Committed generated bindings (chora-contracts/gen/python) --------------
# The chora-contracts wheel maps gen/python → the `chora_contracts_gen`
# package (hatchling [tool.hatch.build.targets.wheel.sources]), and the
# generated modules import each other under that prefix. When the wheel is
# not installed (this repo's unit-test venv), alias the committed source tree
# as `chora_contracts_gen` so the SAME import path works. Resolve relative to
# the repo root so the test is worktree-portable; skip (loudly) when the
# monorepo contracts tree isn't in the build context — the hand-built
# fixtures below still cover the wire shape.
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


# -----------------------------------------------------------------------------
# Hand-built wire fixtures
# -----------------------------------------------------------------------------


def _source_file_ref(blob_uri: str, mime_type: str, role: str) -> bytes:
    """Encode one SourceFileRef submessage (strings at fields 1/2/3)."""
    out = b""
    for tag, value in ((b"\x0a", blob_uri), (b"\x12", mime_type), (b"\x1a", role)):
        if value:
            vb = value.encode("utf-8")
            out += tag + bytes([len(vb)]) + vb
    return out


def _f20(ref: bytes) -> bytes:
    """Wrap one encoded SourceFileRef as field 20 (tag varint 0xA2 0x01)."""
    return b"\xa2\x01" + bytes([len(ref)]) + ref


def _base_event_bytes() -> bytes:
    return (
        b"\x12\x02AS"  # field 2 assist_id="AS"
        b"\x1a\x03TEN"  # field 3 tenant_id="TEN"
        b"\x22\x04GCID"  # field 4 author_gcid="GCID"
        b"\x32\x03mcq"  # field 6 content_type="mcq"
        b"\x3a\x02hi"  # field 7 prompt="hi"
    )


def test_decode_surfaces_source_files_in_declaration_order() -> None:
    bz = (
        _base_event_bytes()
        + _f20(_source_file_ref("gs://b/p1.pdf", "application/pdf", "source"))
        + _f20(_source_file_ref("gs://b/p2.png", "image/png", "source"))
        + _f20(_source_file_ref("gs://b/marks.pdf", "application/pdf", "rubric"))
    )
    out = decode_ai_assist_started(bz)
    assert out["source_files"] == [
        {"blob_uri": "gs://b/p1.pdf", "mime_type": "application/pdf", "role": "source"},
        {"blob_uri": "gs://b/p2.png", "mime_type": "image/png", "role": "source"},
        {"blob_uri": "gs://b/marks.pdf", "mime_type": "application/pdf", "role": "rubric"},
    ]


def test_decode_source_files_key_absent_on_pre_1c_message() -> None:
    # proto3 omits empty repeated fields — a pre-1c (EPIC-1a) message carries
    # no f20 bytes, so the key MUST stay absent (from_event defaults apply and
    # the single/1a paths are byte/behaviour-compatible).
    out = decode_ai_assist_started(_base_event_bytes())
    assert "source_files" not in out


def test_decode_source_file_ref_defaults_missing_strings_empty() -> None:
    # A ref missing mime_type/role decodes those to "" (proto3 scalar default)
    # rather than crashing or dropping the entry.
    bz = _base_event_bytes() + _f20(_source_file_ref("gs://b/only-uri.txt", "", ""))
    out = decode_ai_assist_started(bz)
    assert out["source_files"] == [{"blob_uri": "gs://b/only-uri.txt", "mime_type": "", "role": ""}]


def test_decode_source_file_ref_skips_unknown_subfields() -> None:
    # Forward-compat: a future SourceFileRef field 4 (e.g. display_name) must
    # be skipped without corrupting the known strings.
    ref = _source_file_ref("gs://b/x.pdf", "application/pdf", "source")
    ref += b"\x22\x04NAME"  # unknown subfield 4 (length-delimited)
    out = decode_ai_assist_started(_base_event_bytes() + _f20(ref))
    assert out["source_files"] == [{"blob_uri": "gs://b/x.pdf", "mime_type": "application/pdf", "role": "source"}]


# -----------------------------------------------------------------------------
# Round-trip against the COMMITTED generated pb (the W2 mandate)
# -----------------------------------------------------------------------------


def test_round_trip_against_committed_generated_pb() -> None:
    pb = _load_ai_assist_pb2()

    msg = pb.AiAssistStartedV2()
    msg.envelope.event_id = "ev-1"
    msg.envelope.idempotency_key = "idem-1"
    msg.envelope.tenant_id = "TEN"
    msg.envelope.gcid = "GCID"
    msg.envelope.traceparent = "00-aa-bb-01"
    msg.assist_id = "job-1c"
    msg.tenant_id = "TEN"
    msg.author_gcid = "GCID"
    msg.atom_id = "atom-1"
    msg.content_type = "mcq"
    msg.prompt = "Generate from the attached paper."
    msg.requested_count = 5
    msg.difficulty = 3
    msg.max_retries = 2
    msg.metadata["subject"] = "Biology"
    msg.image_for_stem = True
    msg.intent = "new_question"
    msg.grounding_mode = "strict"
    msg.source_blob_uri = "gs://b/exam.pdf"
    msg.source_mime_type = "application/pdf"
    msg.target_growth_edges.extend(["fractions", "ratios"])
    for blob_uri, mime_type, role in (
        ("gs://b/exam.pdf", "application/pdf", "source"),
        ("gs://b/figure.png", "image/png", "source"),
        ("gs://b/mark-scheme.pdf", "application/pdf", "rubric"),
    ):
        ref = msg.source_files.add()
        ref.blob_uri = blob_uri
        ref.mime_type = mime_type
        ref.role = role

    out = decode_ai_assist_started(msg.SerializeToString())

    assert out["assist_id"] == "job-1c"
    assert out["tenant_id"] == "TEN"
    assert out["author_gcid"] == "GCID"
    assert out["atom_id"] == "atom-1"
    assert out["content_type"] == "mcq"
    assert out["question_type"] == "mcq"  # mirrored
    assert out["prompt"] == "Generate from the attached paper."
    assert out["requested_count"] == 5
    assert out["difficulty"] == 3
    assert out["max_retries"] == 2
    assert out["metadata"] == {"subject": "Biology"}
    assert out["image_for_stem"] is True
    assert out["image_for_answer"] is False
    assert out["intent"] == "new_question"
    assert out["grounding_mode"] == "strict"
    assert out["source_blob_uri"] == "gs://b/exam.pdf"
    assert out["source_mime_type"] == "application/pdf"
    assert out["target_growth_edges"] == ["fractions", "ratios"]
    assert out["source_files"] == [
        {"blob_uri": "gs://b/exam.pdf", "mime_type": "application/pdf", "role": "source"},
        {"blob_uri": "gs://b/figure.png", "mime_type": "image/png", "role": "source"},
        {"blob_uri": "gs://b/mark-scheme.pdf", "mime_type": "application/pdf", "role": "rubric"},
    ]
    assert out["envelope"]["event_id"] == "ev-1"
    assert out["envelope"]["idempotency_key"] == "idem-1"
    assert out["traceparent"] == "00-aa-bb-01"


def test_round_trip_pb_without_source_files_keeps_key_absent() -> None:
    pb = _load_ai_assist_pb2()
    msg = pb.AiAssistStartedV2()
    msg.assist_id = "job-1a"
    msg.tenant_id = "TEN"
    msg.author_gcid = "GCID"
    msg.content_type = "oe"
    msg.prompt = "p"
    msg.intent = "new_question"
    msg.source_blob_uri = "gs://b/single.pdf"
    msg.source_mime_type = "application/pdf"

    out = decode_ai_assist_started(msg.SerializeToString())
    assert "source_files" not in out
    assert out["source_blob_uri"] == "gs://b/single.pdf"
