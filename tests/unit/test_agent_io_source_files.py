"""Unit test: ``build_session_state`` surfaces the Lane 1c multi-file
grounding list (``source_files`` → ``source_files_json``) for the
qgen_question role (CHO-1703 W2).

The DEPLOYED groundingplugin reads only the scalar ``source_blob_uri`` /
``source_mime_type`` keys (one FileData part). ``source_files_json`` is the
ADDITIVE forward seam: a future agent build appends one FileData part per
entry (sources + the role-distinct rubric). Today's agent ignores the key —
generation grounds on the first source file via the scalar mirror, which the
executor DERIVES when the orchestrator threads only ``source_files``.

No live LLM calls: ``build_session_state`` is a pure function.
"""

from __future__ import annotations

import json

from chora_ai_kernel_orchestrator.adapter.agent_io import (
    ROLE_QGEN_QUESTION,
    build_session_state,
)

_TENANT = "11111111-1111-7111-8111-111111111111"

_SOURCE_1 = {"blob_uri": "gs://b/exam.pdf", "mime_type": "application/pdf", "role": "source"}
_SOURCE_2 = {"blob_uri": "gs://b/fig.png", "mime_type": "image/png", "role": "source"}
_RUBRIC = {"blob_uri": "gs://b/marks.pdf", "mime_type": "application/pdf", "role": "rubric"}


def _state(input_extra: dict) -> dict:
    base = {
        "prompt": "Generate MCQs from the source material.",
        "question_type": "mcq",
        "intent": "new_question",
        "gcid": "00000000-0000-7000-8000-000000001999",
    }
    base.update(input_extra)
    return build_session_state(
        agent_role=ROLE_QGEN_QUESTION,
        execution_id="job-1",
        tenant_id=_TENANT,
        input_obj=base,
    )


def test_source_files_json_stamped_when_present() -> None:
    s = _state(
        {
            "source_blob_uri": _SOURCE_1["blob_uri"],
            "source_mime_type": _SOURCE_1["mime_type"],
            "grounding_mode": "strict",
            "source_files": [_SOURCE_1, _SOURCE_2, _RUBRIC],
        }
    )
    assert json.loads(s["source_files_json"]) == [_SOURCE_1, _SOURCE_2, _RUBRIC]
    # Scalar mirror untouched when explicitly supplied.
    assert s["source_blob_uri"] == _SOURCE_1["blob_uri"]
    assert s["source_mime_type"] == _SOURCE_1["mime_type"]


def test_scalar_mirror_derived_from_first_source_file() -> None:
    # Orchestrator threads only source_files (no scalars) — the executor
    # derives the f17/18-equivalent scalars from the FIRST role="source"
    # entry so the deployed single-FileData groundingplugin still fires.
    s = _state({"source_files": [_RUBRIC, _SOURCE_2, _SOURCE_1]})
    assert s["source_blob_uri"] == _SOURCE_2["blob_uri"]  # first role="source"
    assert s["source_mime_type"] == _SOURCE_2["mime_type"]
    assert json.loads(s["source_files_json"]) == [_RUBRIC, _SOURCE_2, _SOURCE_1]


def test_no_source_files_keys_when_absent() -> None:
    s = _state({})
    assert "source_files_json" not in s
    assert "source_blob_uri" not in s


def test_garbage_source_files_ignored() -> None:
    s = _state({"source_files": "gs://not-a-list"})
    assert "source_files_json" not in s
    assert "source_blob_uri" not in s
