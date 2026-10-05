"""Minimal proto3 wire-format decoder for AiAssistStarted events.

Why hand-rolled
---------------
The chora-creation publisher emits binary-protobuf-encoded
``chora.creation.ai_assist.started.v2`` payloads (per Pub/Sub Schema
Registry constraint encoding=BINARY on the topic + chora-creation
``protomarshal.encodeAiAssistStarted``). The orchestrator's
QGenCrewRunner consumes a flat dict shape historically populated by
``json.loads`` of legacy JSON bodies.

Rather than add a full ``chora-contracts/gen/python`` dependency +
Dockerfile path-COPY (which would force a multi-file packaging
refactor + reset the build cache), this module hand-rolls the small
subset of the proto3 wire-format the orchestrator needs to convert a
binary AiAssistStarted message into the same flat dict shape
``qgen_crew_runner.AiAssistStartedPayload.from_event`` already
understands.

Wire-format spec — see https://protobuf.dev/programming-guides/encoding/.

Schema reference — ``chora-contracts/proto/events-flat/creation/ai_assist/started.proto``:

    message AiAssistStarted {
        Envelope envelope = 1;              // length-delimited submessage
        string assist_id = 2;
        string tenant_id = 3;
        string author_gcid = 4;
        string atom_id = 5;
        string content_type = 6;
        string prompt = 7;
        int32 requested_count = 8;
        int32 difficulty = 9;
        Timestamp started_at = 10;          // length-delimited submessage
        int32 max_retries = 11;
        map<string,string> metadata = 12;   // repeated MapEntry submessages
    }

    message Envelope {
        string event_id = 1;
        string idempotency_key = 2;
        string tenant_id = 3;
        string gcid = 4;
        Timestamp occurred_at = 5;
        Timestamp published_at = 6;
        string traceparent = 7;
        string tracestate = 8;
        ...
    }

W8 author-opt-in image flags (added to AiAssistStarted 2026-06-01):

    bool image_for_stem   = 13;  // varint (wire type 0)
    bool image_for_answer = 14;  // varint (wire type 0)

These two bools are surfaced top-level on the decoded dict so
``qgen_crew_runner.AiAssistStartedPayload.from_event`` (which reads
``image_for_stem`` / ``image_for_answer`` with a default-false fallback)
populates the W8 render-image opt-in for real. proto3 omits false
scalars, so a pre-W8 / opt-in-off message carries no bytes for them and
the decoder defaults both to ``False`` — byte/behaviour-compatible with
the legacy shape.

EPIC-1a batch + grounding fields (added to AiAssistStarted 2026-06-09,
contract c6a9e38b) — all length-delimited (wire type 2):

    string job_kind            = 15;  // "" | "single" | "batch"
    string grounding_mode      = 16;  // "" | "starting_point" | "strict"
    string source_blob_uri     = 17;  // gs:// upload for grounding
    string source_mime_type    = 18;  // e.g. "application/pdf"
    repeated string target_growth_edges = 19;

These surface top-level so ``AiAssistStartedPayload.from_event`` can route
batch jobs to ``QGenBatchRunner`` (``job_kind``) and stamp the qgen
grounding plugin's session state (``grounding_mode`` / ``source_blob_uri``
/ ``source_mime_type``). proto3 omits empty scalars, so a single /
non-grounded message carries no bytes for them and the keys stay absent —
``from_event`` defaults them, keeping the live single-candidate path
byte/behaviour-compatible.

Lane 1c multi-file + rubric grounding (added to AiAssistStarted
2026-06-10, contract afea5870 / CHO-1703):

    repeated SourceFileRef source_files = 20;   // length-delimited submessages

    message SourceFileRef {
        string blob_uri  = 1;
        string mime_type = 2;
        string role      = 3;   // "source" | "rubric"
    }

Each field-20 occurrence decodes to one ``{"blob_uri", "mime_type",
"role"}`` dict, accumulated in declaration order under the top-level
``source_files`` key. When non-empty this is the CANONICAL grounding-file
list (f17/18 merely mirror source_files[0] for rollout back-compat —
consumers SHOULD prefer source_files when set). proto3 omits empty
repeated fields, so a pre-1c message carries no bytes for it and the key
stays absent — ``from_event`` defaults it, keeping the 1a single-file
batch path byte/behaviour-compatible.

ADR-195 WS7 (D7) compose model — AiAssistStartedV2 (2026-06-26):

    reserved 15;                 // job_kind DROPPED
    string operation   = 23;     // always "compose"
    string intent      = 24;     // new_question | model_answer_fill | image_regen
    string input_kind  = 25;     // prompt | source_files | by_hand

The .v2 message replaces the legacy ``job_kind`` discriminant with the explicit
compose model so the orchestrator's router dispatches on intent/input_kind. ONE
decoder handles BOTH versions: every non-discriminant field keeps its v1 tag, v1
carries ``job_kind`` (15) and none of 23-25, v2 carries 23-25 and no 15 — no
field-number collision. proto3 omits empty scalars, so the keys for the version's
absent discriminant simply do not appear (the router falls back accordingly).

This decoder is intentionally permissive — unknown field numbers are
skipped (forward-compatibility), and any decode error bubbles up so
the subscriber can NACK + log the malformed envelope.

When chora-contracts python bindings land as a first-class dep, this
module can be replaced with a thin wrapper around the generated
``AiAssistStarted.ParseFromString`` + ``MessageToDict``.
"""

from __future__ import annotations

import json
from typing import Any


class ProtoDecodeError(ValueError):
    """Raised when binary input fails to parse as AiAssistStarted."""


def _read_varint(data: bytes, offset: int) -> tuple[int, int]:
    """Read one varint starting at ``offset``. Returns (value, new_offset)."""
    result = 0
    shift = 0
    while offset < len(data):
        b = data[offset]
        offset += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, offset
        shift += 7
        if shift > 63:
            raise ProtoDecodeError("varint exceeds 64 bits")
    raise ProtoDecodeError("varint truncated")


def _read_length_delimited(data: bytes, offset: int) -> tuple[bytes, int]:
    length, offset = _read_varint(data, offset)
    if offset + length > len(data):
        raise ProtoDecodeError("length-delimited field overruns buffer")
    return data[offset : offset + length], offset + length


def _skip_field(data: bytes, offset: int, wire_type: int) -> int:
    """Advance offset past a field whose contents we don't care about."""
    if wire_type == 0:  # varint
        _, offset = _read_varint(data, offset)
    elif wire_type == 1:  # 64-bit fixed
        offset += 8
    elif wire_type == 2:  # length-delimited
        _, offset = _read_length_delimited(data, offset)
    elif wire_type == 5:  # 32-bit fixed
        offset += 4
    else:
        raise ProtoDecodeError(f"unsupported wire type {wire_type}")
    return offset


def _decode_envelope(data: bytes) -> dict[str, Any]:
    """Decode an Envelope submessage to a dict of just the fields the
    orchestrator's idempotency + trace plumbing needs.
    """
    env: dict[str, Any] = {}
    offset = 0
    while offset < len(data):
        tag, offset = _read_varint(data, offset)
        field = tag >> 3
        wire_type = tag & 0x07

        if wire_type == 2 and field in (1, 2, 3, 4, 7, 8):
            buf, offset = _read_length_delimited(data, offset)
            value = buf.decode("utf-8")
            if field == 1:
                env["event_id"] = value
            elif field == 2:
                env["idempotency_key"] = value
            elif field == 3:
                env["tenant_id"] = value
            elif field == 4:
                env["gcid"] = value
            elif field == 7:
                env["traceparent"] = value
            elif field == 8:
                env["tracestate"] = value
        else:
            offset = _skip_field(data, offset, wire_type)
    return env


def _decode_map_entry(data: bytes) -> tuple[str, str] | None:
    """Decode a single map<string,string> MapEntry submessage."""
    key = ""
    value = ""
    offset = 0
    while offset < len(data):
        tag, offset = _read_varint(data, offset)
        field = tag >> 3
        wire_type = tag & 0x07
        if wire_type == 2 and field == 1:
            buf, offset = _read_length_delimited(data, offset)
            key = buf.decode("utf-8")
        elif wire_type == 2 and field == 2:
            buf, offset = _read_length_delimited(data, offset)
            value = buf.decode("utf-8")
        else:
            offset = _skip_field(data, offset, wire_type)
    return key, value


def _decode_source_file_ref(data: bytes) -> dict[str, str]:
    """Decode a single ``chora.creation.v1.SourceFileRef`` submessage (Lane 1c
    multi-file + rubric grounding — strings at fields 1/2/3). Missing strings
    default to ``""`` per proto3 scalar semantics; unknown subfields are
    skipped (forward-compatibility).
    """
    ref = {"blob_uri": "", "mime_type": "", "role": ""}
    offset = 0
    while offset < len(data):
        tag, offset = _read_varint(data, offset)
        field = tag >> 3
        wire_type = tag & 0x07
        if wire_type == 2 and field in (1, 2, 3):
            buf, offset = _read_length_delimited(data, offset)
            value = buf.decode("utf-8")
            if field == 1:
                ref["blob_uri"] = value
            elif field == 2:
                ref["mime_type"] = value
            else:
                ref["role"] = value
        else:
            offset = _skip_field(data, offset, wire_type)
    return ref


def _decode_generation_type_quota(data: bytes) -> dict[str, Any]:
    """Decode a single ``GenerationTypeQuota`` submessage (mixed-type batch,
    CHO-1819): ``question_type`` string at field 1; ``count`` / ``max_images``
    int32 varints at fields 2/3. Missing scalars default per proto3
    (``""`` / 0); unknown subfields are skipped (forward-compatibility). The
    decoder surfaces faithfully — fail-loud validation of the invariants
    (count >= 1, 0 <= max_images <= count) lives in the chora-creation domain
    and is re-checked in the graph, not here.

    CHO-1825 — the per-type author image opt-ins ``image_for_stem`` (field 4) /
    ``image_for_answer`` (field 5) are bools (varint, wire type 0). They are
    deterministic toggles: when true, EVERY question of this type MUST carry that
    image. proto3 omits false scalars, so an opt-in-off quota carries no bytes for
    f4/f5 and the key stays ABSENT — ``build_type_plan`` reads them with a
    default-false fallback, so the legacy mixed-batch shape stays
    byte/behaviour-compatible. Dropping them silently disables the toggle (the
    Go agent never forces the image).
    """
    quota: dict[str, Any] = {"question_type": "", "count": 0, "max_images": 0}
    offset = 0
    while offset < len(data):
        tag, offset = _read_varint(data, offset)
        field = tag >> 3
        wire_type = tag & 0x07
        if wire_type == 2 and field == 1:
            buf, offset = _read_length_delimited(data, offset)
            quota["question_type"] = buf.decode("utf-8")
        elif wire_type == 0 and field == 2:
            value, offset = _read_varint(data, offset)
            quota["count"] = value
        elif wire_type == 0 and field == 3:
            value, offset = _read_varint(data, offset)
            quota["max_images"] = value
        elif wire_type == 0 and field == 4:
            value, offset = _read_varint(data, offset)
            quota["image_for_stem"] = bool(value)
        elif wire_type == 0 and field == 5:
            value, offset = _read_varint(data, offset)
            quota["image_for_answer"] = bool(value)
        else:
            offset = _skip_field(data, offset, wire_type)
    return quota


def _decode_image_regen_spec(data: bytes) -> dict[str, str]:
    """Decode an ``ImageRegenSpec`` submessage (CHO-1819 P3, AiAssistStarted
    field 22): {1: draft_id, 2: placement, 3: prompt, 4: mode} — all strings.
    Unknown fields are skipped (forward-compat); omitted scalars default to ""
    (proto3). Faithful surface only — placement/prompt validation is upstream.

    I2 add — fields 5/6/7 carry the author's CURRENT edited question context so
    the image regenerate prompt reflects unsaved review-UI edits:
    ``current_stem`` (f5) / ``current_model_answer`` (f6) / ``original_source``
    (f7) — all strings. proto3 omits empty scalars, so a regen spec that omits
    them carries no bytes for f5-7 and they default to ``""`` here (the
    orchestrator falls back to the candidate's stored context) — byte/behaviour-
    compatible with a pre-I2 producer. Dropping them silently re-renders the
    image against the stale stored candidate instead of the edited question.
    """
    spec: dict[str, str] = {
        "draft_id": "",
        "placement": "",
        "prompt": "",
        "mode": "",
        "current_stem": "",
        "current_model_answer": "",
        "original_source": "",
        "original_image_gcs_uri": "",
    }
    offset = 0
    while offset < len(data):
        tag, offset = _read_varint(data, offset)
        field = tag >> 3
        wire_type = tag & 0x07
        if wire_type == 2 and field == 1:
            buf, offset = _read_length_delimited(data, offset)
            spec["draft_id"] = buf.decode("utf-8")
        elif wire_type == 2 and field == 2:
            buf, offset = _read_length_delimited(data, offset)
            spec["placement"] = buf.decode("utf-8")
        elif wire_type == 2 and field == 3:
            buf, offset = _read_length_delimited(data, offset)
            spec["prompt"] = buf.decode("utf-8")
        elif wire_type == 2 and field == 4:
            buf, offset = _read_length_delimited(data, offset)
            spec["mode"] = buf.decode("utf-8")
        elif wire_type == 2 and field == 5:
            buf, offset = _read_length_delimited(data, offset)
            spec["current_stem"] = buf.decode("utf-8")
        elif wire_type == 2 and field == 6:
            buf, offset = _read_length_delimited(data, offset)
            spec["current_model_answer"] = buf.decode("utf-8")
        elif wire_type == 2 and field == 7:
            buf, offset = _read_length_delimited(data, offset)
            spec["original_source"] = buf.decode("utf-8")
        elif wire_type == 2 and field == 8:
            buf, offset = _read_length_delimited(data, offset)
            spec["original_image_gcs_uri"] = buf.decode("utf-8")
        else:
            offset = _skip_field(data, offset, wire_type)
    return spec


def decode_ai_assist_started(data: bytes) -> dict[str, Any]:
    """Decode a binary-proto ``chora.creation.v1.AiAssistStarted`` message
    into a flat dict matching what ``QGenCrewRunner.handle_started`` +
    ``AiAssistStartedPayload.from_event`` already consume.

    Output keys (all optional except as required by ``from_event``):

    * ``assist_id``, ``tenant_id``, ``author_gcid``, ``atom_id``,
      ``content_type``, ``question_type`` (mirrored from content_type),
      ``prompt``, ``max_retries``, ``metadata`` (dict[str,str])
    * ``image_for_stem``, ``image_for_answer`` (W8 author-opt-in bools,
      fields 13/14 — default ``False`` when absent)
    * ``traceparent``, ``tracestate`` (from the proto envelope)
    * ``envelope`` (nested dict: event_id, idempotency_key, …)
    """
    # W8 image-render opt-in flags default False so the key is always
    # present for the runner's ``bool(event.get(...) or False)`` read and a
    # pre-W8 message (which carries no bytes for fields 13/14) is
    # byte/behaviour-compatible.
    out: dict[str, Any] = {"image_for_stem": False, "image_for_answer": False}
    metadata: dict[str, str] = {}
    offset = 0
    while offset < len(data):
        tag, offset = _read_varint(data, offset)
        field = tag >> 3
        wire_type = tag & 0x07

        if wire_type == 2:
            buf, offset = _read_length_delimited(data, offset)
            if field == 1:
                env = _decode_envelope(buf)
                out["envelope"] = env
                # Surface envelope dedup + trace fields at the top level
                # so the subscriber + runner do not need to peek inside.
                if "traceparent" in env:
                    out["traceparent"] = env["traceparent"]
                if "tracestate" in env:
                    out["tracestate"] = env["tracestate"]
            elif field == 2:
                out["assist_id"] = buf.decode("utf-8")
            elif field == 3:
                out["tenant_id"] = buf.decode("utf-8")
            elif field == 4:
                out["author_gcid"] = buf.decode("utf-8")
            elif field == 5:
                out["atom_id"] = buf.decode("utf-8")
            elif field == 6:
                content_type = buf.decode("utf-8")
                out["content_type"] = content_type
                # Mirror to question_type because the proto schema didn't add
                # a separate field 13 — see chora-creation handler comment at
                # ai_assist_async_handler.go line 174-175 — legacy
                # content_type carries the question_type alias.
                out["question_type"] = content_type
            elif field == 7:
                out["prompt"] = buf.decode("utf-8")
            elif field == 10:
                # started_at — Timestamp submessage. The runner does not
                # currently consume it (uses NOW() server-side at terminal
                # publish), so leave it as the raw bytes — future-proof.
                pass
            elif field == 12:
                # map<string,string> metadata — one MapEntry per field 12
                # occurrence.
                entry = _decode_map_entry(buf)
                if entry is not None:
                    k, v = entry
                    metadata[k] = v
            # EPIC-1a batch + grounding (AiAssistStarted f15-19, added by the 1a
            # contract c6a9e38b). All length-delimited strings; field 19 is
            # repeated. These MUST surface top-level so
            # AiAssistStartedPayload.from_event can route batch jobs to the
            # QGenBatchRunner (job_kind) and stamp the grounding plugin's session
            # state (grounding_mode / source_blob_uri / source_mime_type).
            # Dropping them collapsed count=N→1 and silenced grounding.
            elif field == 15:
                out["job_kind"] = buf.decode("utf-8")
            elif field == 16:
                out["grounding_mode"] = buf.decode("utf-8")
            elif field == 17:
                out["source_blob_uri"] = buf.decode("utf-8")
            elif field == 18:
                out["source_mime_type"] = buf.decode("utf-8")
            elif field == 19:
                # repeated string — accumulate in declaration order.
                out.setdefault("target_growth_edges", []).append(buf.decode("utf-8"))
            # Lane 1c multi-file + rubric grounding (AiAssistStarted f20,
            # contract afea5870 / CHO-1703) — repeated SourceFileRef
            # submessages, one per field-20 occurrence, accumulated in
            # declaration order. Canonical when non-empty; f17/18 mirror
            # source_files[0] for rollout back-compat.
            elif field == 20:
                out.setdefault("source_files", []).append(_decode_source_file_ref(buf))
            # Mixed-type batch (AiAssistStarted f21, CHO-1819) — repeated
            # GenerationTypeQuota submessages, one per field-21 occurrence,
            # accumulated in declaration order. EMPTY ⇒ legacy single-type
            # path (key stays absent → byte/behaviour-compatible).
            elif field == 21:
                out.setdefault("type_plan", []).append(_decode_generation_type_quota(buf))
            # Review image regenerate (AiAssistStarted f22, CHO-1819 P3) —
            # SINGULAR ImageRegenSpec submessage. ABSENT ⇒ non-regen job (key
            # stays absent → byte/behaviour-compatible). PRESENT ⇒ the runner
            # routes job_kind="image_regen" to the ImageRegenRunner.
            elif field == 22:
                out["regen"] = _decode_image_regen_spec(buf)
            # ADR-195 WS7 (D7) — AiAssistStartedV2 compose model. The .v2 message
            # DROPS job_kind (tag 15, reserved) and carries the explicit compose
            # discriminant {operation, intent, input_kind} on tags 23-25 (all
            # length-delimited strings). proto3 omits empty scalars, so a v1
            # message carries no bytes for them and the keys stay ABSENT — the
            # router then falls back to the legacy job_kind. ONE decoder serves
            # both wire versions: there is no field-number collision (v1 carries
            # tag 15 and none of 23-25; v2 carries 23-25 and no tag 15).
            elif field == 23:
                out["operation"] = buf.decode("utf-8")
            elif field == 24:
                out["intent"] = buf.decode("utf-8")
            elif field == 25:
                out["input_kind"] = buf.decode("utf-8")
            # CHO-1658 — existing_question_json (tag 26): the author's CURRENT
            # question content for intent=model_answer_fill, JSON-encoded. Surface
            # it as the PARSED dict under "existing_question" — the shape
            # reasoning_engine_executor reads into the qgen agent's author_stem /
            # author_options / author_rubric / model_answer session keys. Defensive:
            # a malformed payload is DROPPED (the key stays absent so the executor's
            # `or {}` degrades to a free-generation pass) rather than raising — a
            # broken fill must never poison-loop the subscription. Empty/absent on
            # every non-fill event (proto3 elision) ⇒ key absent ⇒ byte-compatible.
            elif field == 26:
                try:
                    parsed = json.loads(buf.decode("utf-8"))
                except (ValueError, UnicodeDecodeError):
                    parsed = None
                if isinstance(parsed, dict):
                    out["existing_question"] = parsed
            # other length-delimited fields ignored
        elif wire_type == 0:
            value, offset = _read_varint(data, offset)
            if field == 8:
                out["requested_count"] = value
            elif field == 9:
                out["difficulty"] = value
            elif field == 11:
                out["max_retries"] = value
            elif field == 13:
                # W8 image_for_stem (bool → varint): any non-zero is true.
                out["image_for_stem"] = bool(value)
            elif field == 14:
                # W8 image_for_answer (bool → varint): any non-zero is true.
                out["image_for_answer"] = bool(value)
        else:
            offset = _skip_field(data, offset, wire_type)

    if metadata:
        out["metadata"] = metadata
    return out


def looks_like_binary_proto(data: bytes) -> bool:
    """Cheap heuristic — is ``data`` more likely binary proto than JSON?

    ``chora.creation.v1.AiAssistStarted`` always emits an Envelope at
    field 1 (length-delimited), so the first byte is ``0x0A`` (tag 1,
    wire type 2). JSON bodies start with ``{`` (0x7B) or ``[`` (0x5B)
    or whitespace.

    Used by the subscriber to decide which decode path to take before
    full parse.
    """
    if not data:
        return False
    return data[0] == 0x0A
