"""Unit tests — Lane 1c per-candidate citation normalisation + deterministic
draft ids (CHO-1703 W2).

The orchestrator emits MODEL-REPORTED citations — it never verifies them
(D15: chora-creation's completed-event subscriber matches excerpts against
the persisted source_material_chunks and stamps ``verified``/``chunk_id``).
The runner-side normalisation only makes the model output SAFE + canonical:

* clamp excerpts to the contract's 1024-char cap
* coerce ``page`` to a 1-based int (drop the key otherwise — nullable)
* resolve bare-filename ``source_file`` mentions to the job's gs:// blob_uri
* drop garbage entries; strip a hallucinated ``citations`` key from
  UNGROUNDED jobs entirely

``deterministic_draft_id`` keys the composer proposal (order/points maps) to
the candidates surfaced to the FE — uuid5 over (assist_id, index) so a
checkpoint-replayed batch re-publish mints IDENTICAL ids (idempotency).
"""

from __future__ import annotations

import uuid

from chora_ai_kernel_orchestrator.orchestrators.qgen_grounding import (
    deterministic_draft_id,
    normalise_candidate_citations,
)

_FILES = [
    {"blob_uri": "gs://b/folder/exam.pdf", "mime_type": "application/pdf", "role": "source"},
    {"blob_uri": "gs://b/fig.png", "mime_type": "image/png", "role": "source"},
]


# -----------------------------------------------------------------------------
# deterministic_draft_id
# -----------------------------------------------------------------------------


def test_draft_id_is_deterministic_and_uuid_shaped() -> None:
    a = deterministic_draft_id("job-1c", 0)
    b = deterministic_draft_id("job-1c", 0)
    assert a == b
    uuid.UUID(a)  # parses as a UUID (the OpenAPI order[] items are format:uuid)


def test_draft_id_varies_by_index_and_assist_id() -> None:
    ids = {
        deterministic_draft_id("job-1c", 0),
        deterministic_draft_id("job-1c", 1),
        deterministic_draft_id("job-other", 0),
    }
    assert len(ids) == 3


# -----------------------------------------------------------------------------
# normalise_candidate_citations — grounded jobs
# -----------------------------------------------------------------------------


def test_citations_normalised_in_place_shape() -> None:
    cand = {
        "stem": "Q1",
        "citations": [
            {"source_file": "gs://b/folder/exam.pdf", "page": 3, "excerpt": "verbatim text"},
        ],
    }
    out = normalise_candidate_citations(cand, _FILES)
    assert out["citations"] == [{"source_file": "gs://b/folder/exam.pdf", "page": 3, "excerpt": "verbatim text"}]


def test_citations_excerpt_clamped_to_1024() -> None:
    cand = {"citations": [{"source_file": "gs://b/fig.png", "excerpt": "x" * 5000}]}
    out = normalise_candidate_citations(cand, _FILES)
    assert len(out["citations"][0]["excerpt"]) == 1024


def test_citations_bare_filename_resolved_to_blob_uri() -> None:
    cand = {"citations": [{"source_file": "exam.pdf", "excerpt": "e"}]}
    out = normalise_candidate_citations(cand, _FILES)
    assert out["citations"][0]["source_file"] == "gs://b/folder/exam.pdf"


def test_citations_unknown_source_file_left_as_is() -> None:
    # Creation's verification pass flags it unverified — never silently drop.
    cand = {"citations": [{"source_file": "mystery.docx", "excerpt": "e"}]}
    out = normalise_candidate_citations(cand, _FILES)
    assert out["citations"][0]["source_file"] == "mystery.docx"


def test_citations_page_coercion() -> None:
    cand = {
        "citations": [
            {"source_file": "gs://b/fig.png", "page": "2", "excerpt": "a"},
            {"source_file": "gs://b/fig.png", "page": 0, "excerpt": "b"},
            {"source_file": "gs://b/fig.png", "page": "n/a", "excerpt": "c"},
            {"source_file": "gs://b/fig.png", "page": None, "excerpt": "d"},
        ]
    }
    out = normalise_candidate_citations(cand, _FILES)
    pages = [c.get("page") for c in out["citations"]]
    assert pages == [2, None, None, None]


def test_citations_garbage_entries_dropped() -> None:
    cand = {
        "citations": [
            "not-a-dict",
            {"page": 1},  # no source_file + no excerpt
            {"source_file": "", "excerpt": "   "},  # both blank
            {"source_file": "gs://b/fig.png", "excerpt": "keep me"},
        ]
    }
    out = normalise_candidate_citations(cand, _FILES)
    assert out["citations"] == [{"source_file": "gs://b/fig.png", "excerpt": "keep me"}]


def test_citations_non_list_value_removed() -> None:
    cand = {"stem": "Q", "citations": "see page 3"}
    out = normalise_candidate_citations(cand, _FILES)
    assert "citations" not in out


def test_citations_absent_stays_absent_grounded() -> None:
    # Best-effort model-reported — never fabricate.
    out = normalise_candidate_citations({"stem": "Q"}, _FILES)
    assert "citations" not in out


def test_citations_normalised_inside_wrapped_candidate_shapes() -> None:
    # The qgen agent may emit {"candidate": {...}} or the evaluator wrapper
    # {"scored": {"candidate": {...}}} — normalise wherever the core candidate
    # carries the citations array.
    wrapped = {"candidate": {"stem": "Q", "citations": [{"source_file": "exam.pdf", "excerpt": "e"}]}}
    out = normalise_candidate_citations(wrapped, _FILES)
    assert out["candidate"]["citations"][0]["source_file"] == "gs://b/folder/exam.pdf"

    scored = {
        "scored": {
            "candidate": {"stem": "Q", "citations": [{"source_file": "fig.png", "excerpt": "e"}]},
            "composite": 0.9,
        }
    }
    out2 = normalise_candidate_citations(scored, _FILES)
    assert out2["scored"]["candidate"]["citations"][0]["source_file"] == "gs://b/fig.png"


# -----------------------------------------------------------------------------
# normalise_candidate_citations — ungrounded jobs
# -----------------------------------------------------------------------------


def test_ungrounded_job_strips_hallucinated_citations() -> None:
    cand = {"stem": "Q", "citations": [{"source_file": "made-up.pdf", "excerpt": "x"}]}
    out = normalise_candidate_citations(cand, [])
    assert "citations" not in out


def test_ungrounded_job_without_citations_unchanged() -> None:
    cand = {"stem": "Q", "mcq_payload": {"options": []}}
    assert normalise_candidate_citations(cand, []) == cand


def test_non_dict_candidate_passthrough() -> None:
    assert normalise_candidate_citations("raw-string", _FILES) == "raw-string"  # type: ignore[arg-type]
