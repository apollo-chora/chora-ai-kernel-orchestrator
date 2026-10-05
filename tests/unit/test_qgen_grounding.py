"""Unit tests — Lane 1c multi-file + rubric grounding helpers (CHO-1703 W2).

Pure-function coverage for ``orchestrators/qgen_grounding.py``:

* ``normalise_source_files`` — defensive wire→typed list parse
* ``effective_source_files`` — prefer source_files[]; fall back to the
  f17/18 single-file mirror; [] when ungrounded
* ``split_source_files`` — role partition ("source" files vs the ≤1 "rubric"
  mark-scheme file)
* ``grounding_prompt_block`` — the generation-prompt extension that lists the
  grounding corpus, demands per-candidate model-reported citations
  {source_file, page, excerpt}, and (when a rubric file rides the job) the
  explicit mark-scheme alignment instructions (OE rubric criterion weights +
  marks cues like "[5 marks]" / "Question 3 (10 points)").
"""

from __future__ import annotations

from chora_ai_kernel_orchestrator.orchestrators.qgen_grounding import (
    ROLE_RUBRIC,
    ROLE_SOURCE,
    effective_source_files,
    grounding_prompt_block,
    normalise_source_files,
    split_source_files,
)

_PDF = {"blob_uri": "gs://b/exam.pdf", "mime_type": "application/pdf", "role": "source"}
_PNG = {"blob_uri": "gs://b/fig.png", "mime_type": "image/png", "role": "source"}
_RUBRIC = {"blob_uri": "gs://b/marks.pdf", "mime_type": "application/pdf", "role": "rubric"}


# -----------------------------------------------------------------------------
# normalise_source_files
# -----------------------------------------------------------------------------


def test_normalise_passes_well_formed_entries_in_order() -> None:
    out = normalise_source_files([_PDF, _PNG, _RUBRIC])
    assert out == [_PDF, _PNG, _RUBRIC]


def test_normalise_defaults_unknown_or_missing_role_to_source() -> None:
    out = normalise_source_files(
        [
            {"blob_uri": "gs://b/a.txt", "mime_type": "text/plain"},
            {"blob_uri": "gs://b/b.txt", "mime_type": "text/plain", "role": "WEIRD"},
            {"blob_uri": "gs://b/c.pdf", "mime_type": "application/pdf", "role": " RUBRIC "},
        ]
    )
    assert [f["role"] for f in out] == [ROLE_SOURCE, ROLE_SOURCE, ROLE_RUBRIC]


def test_normalise_drops_garbage_entries() -> None:
    out = normalise_source_files(
        [
            "not-a-dict",
            {"mime_type": "application/pdf"},  # no blob_uri
            {"blob_uri": "   "},  # blank blob_uri
            None,
            {"blob_uri": "gs://b/ok.pdf", "mime_type": "application/pdf", "role": "source"},
        ]
    )
    assert out == [{"blob_uri": "gs://b/ok.pdf", "mime_type": "application/pdf", "role": "source"}]


def test_normalise_tolerates_non_list_input() -> None:
    assert normalise_source_files(None) == []
    assert normalise_source_files("gs://b/x.pdf") == []
    assert normalise_source_files({"blob_uri": "gs://b/x.pdf"}) == []


# -----------------------------------------------------------------------------
# effective_source_files — the f17/18 back-compat resolution
# -----------------------------------------------------------------------------


def test_effective_prefers_source_files_when_non_empty() -> None:
    out = effective_source_files(
        source_files=[_PDF, _RUBRIC],
        source_blob_uri="gs://b/IGNORED.pdf",
        source_mime_type="application/pdf",
    )
    assert out == [_PDF, _RUBRIC]


def test_effective_falls_back_to_f17_f18_single_file() -> None:
    out = effective_source_files(
        source_files=[],
        source_blob_uri="gs://b/legacy.pdf",
        source_mime_type="application/pdf",
    )
    assert out == [{"blob_uri": "gs://b/legacy.pdf", "mime_type": "application/pdf", "role": ROLE_SOURCE}]


def test_effective_empty_when_ungrounded() -> None:
    assert effective_source_files(source_files=[], source_blob_uri="", source_mime_type="") == []


# -----------------------------------------------------------------------------
# split_source_files
# -----------------------------------------------------------------------------


def test_split_partitions_sources_and_first_rubric() -> None:
    sources, rubric = split_source_files([_PDF, _RUBRIC, _PNG])
    assert sources == [_PDF, _PNG]
    assert rubric == _RUBRIC


def test_split_without_rubric() -> None:
    sources, rubric = split_source_files([_PDF, _PNG])
    assert sources == [_PDF, _PNG]
    assert rubric is None


def test_split_extra_rubrics_beyond_first_are_dropped() -> None:
    # The upload handler caps at ≤1 rubric file; defensively the FIRST wins and
    # extras are dropped (a mark scheme must never masquerade as question
    # source material).
    second = {"blob_uri": "gs://b/marks2.pdf", "mime_type": "application/pdf", "role": "rubric"}
    sources, rubric = split_source_files([_RUBRIC, second, _PDF])
    assert rubric == _RUBRIC
    assert sources == [_PDF]


# -----------------------------------------------------------------------------
# grounding_prompt_block
# -----------------------------------------------------------------------------


def test_prompt_block_lists_every_file_uri() -> None:
    block = grounding_prompt_block(files=[_PDF, _PNG, _RUBRIC], grounding_mode="strict", question_type="mcq")
    assert "gs://b/exam.pdf" in block
    assert "gs://b/fig.png" in block
    assert "gs://b/marks.pdf" in block


def test_prompt_block_demands_citation_shape() -> None:
    block = grounding_prompt_block(files=[_PDF], grounding_mode="", question_type="mcq")
    # The exact JSON key shape the model must emit per candidate — creation's
    # verification pass (D15) + the FE citations sidebar parse these keys.
    assert '"citations"' in block
    assert '"source_file"' in block
    assert '"page"' in block
    assert '"excerpt"' in block
    assert "1024" in block  # verbatim-excerpt cap
    assert "1-based" in block or "1-indexed" in block


def test_prompt_block_strict_mode_framing() -> None:
    strict = grounding_prompt_block(files=[_PDF], grounding_mode="strict", question_type="mcq")
    seed = grounding_prompt_block(files=[_PDF], grounding_mode="starting_point", question_type="mcq")
    assert "ONLY" in strict
    assert strict != seed


def test_prompt_block_rubric_mark_scheme_instructions() -> None:
    block = grounding_prompt_block(files=[_PDF, _RUBRIC], grounding_mode="", question_type="oe")
    assert "mark scheme" in block.lower() or "mark-scheme" in block.lower()
    assert "[5 marks]" in block
    assert "(10 points)" in block
    # OE rubric alignment — criterion weights follow the rubric file.
    assert "weight" in block.lower()
    assert "gs://b/marks.pdf" in block


def test_prompt_block_no_rubric_instructions_without_rubric_file() -> None:
    block = grounding_prompt_block(files=[_PDF], grounding_mode="", question_type="oe")
    assert "[5 marks]" not in block


def test_prompt_block_empty_for_no_files() -> None:
    assert grounding_prompt_block(files=[], grounding_mode="strict", question_type="mcq") == ""
