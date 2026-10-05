"""Unit test: `build_session_state` stamps the EPIC-1a batch grounding keys
(source_blob_uri / source_mime_type / grounding_mode) for the qgen_question role
so the agent's groundingplugin injects a gs:// FileData part. Absent material ⇒
no keys (the live single-candidate + non-grounded paths are byte-for-byte
unchanged).

No live LLM calls: `build_session_state` is a pure function.
"""

from __future__ import annotations

from chora_ai_kernel_orchestrator.adapter.agent_io import (
    ROLE_QGEN_QUESTION,
    build_session_state,
)

_TENANT = "11111111-1111-7111-8111-111111111111"


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


def test_grounding_keys_stamped_when_material_present() -> None:
    s = _state(
        {
            "source_blob_uri": "gs://chora-batch-uploads/t/job/material.pdf",
            "source_mime_type": "application/pdf",
            "grounding_mode": "strict",
        }
    )
    assert s["source_blob_uri"] == "gs://chora-batch-uploads/t/job/material.pdf"
    assert s["source_mime_type"] == "application/pdf"
    assert s["grounding_mode"] == "strict"


def test_no_grounding_keys_when_no_material() -> None:
    s = _state({})
    assert "source_blob_uri" not in s
    assert "source_mime_type" not in s
    assert "grounding_mode" not in s


def test_grounding_mode_omitted_when_blank() -> None:
    # Material present but grounding_mode unset → blob keys stamped, mode omitted
    # (the orchestrator default-prompt handles framing; the agent treats absent
    # mode as starting_point).
    s = _state(
        {
            "source_blob_uri": "gs://b/m.png",
            "source_mime_type": "image/png",
        }
    )
    assert s["source_blob_uri"] == "gs://b/m.png"
    assert s["source_mime_type"] == "image/png"
    assert "grounding_mode" not in s
