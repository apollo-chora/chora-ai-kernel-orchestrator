"""CHO-2368 P2 — the 1.1.0 revision fixtures (safe segments only).

Pins the wave's content rules:
- every revised segment id is a NON-LOCKED catalogue segment of its agent;
- the qgen pair never revises `task` (a task override flattens the per-intent /
  per-question-type template dispatch);
- append-strategy revisions preserve the baseline verbatim as a prefix
  (meaningful INCREMENTAL change, per the owner ruling);
- every revision actually differs from the baseline;
- my added prose carries no em dashes (owner style rule; quoted baseline text
  keeps its source characters).
"""

from __future__ import annotations

import pytest

from chora_ai_kernel_orchestrator.domain.prompt_registry import baseline_seedspec
from chora_ai_kernel_orchestrator.domain.prompt_registry import revisions_110 as rev

EXPECTED_SEGMENT_IDS = {
    "qgen_question": {"role", "examples"},
    "qgen_critic": {"role", "examples"},
    "oe_evaluator": {"role", "task"},
    "oe_moderator": {"role", "task"},
    # P3 (CHO-2368): the familiar's 3 safe catalogue segments. Its composer
    # applies APPEND-AT-RENDER (per-instance dynamic blocks), so the registry
    # bodies are the additive craft text alone - strategy `replace` here means
    # "store the authored text verbatim", NOT baseline-concat.
    "familiar": {"role_frame", "examples_frame", "task_frame"},
}

_BASELINES = {(s.agent_id, s.segment_id): s for s in baseline_seedspec.baseline_segments()}


def test_version_label() -> None:
    assert rev.REVISION_VERSION_LABEL == "1.1.0"


def test_covers_exactly_the_four_wave_agents() -> None:
    segs = rev.revision_segments()
    assert {s.agent_id for s in segs} == set(EXPECTED_SEGMENT_IDS)


@pytest.mark.parametrize("agent_id", sorted(EXPECTED_SEGMENT_IDS))
def test_segment_ids_exact_and_safe(agent_id: str) -> None:
    ids = {s.segment_id for s in rev.revision_segments() if s.agent_id == agent_id}
    assert ids == EXPECTED_SEGMENT_IDS[agent_id]
    for segment_id in ids:
        baseline = _BASELINES[(agent_id, segment_id)]
        assert baseline.locked is False, f"{agent_id}/{segment_id} is locked"


def test_qgen_pair_never_revises_task() -> None:
    for s in rev.revision_segments():
        if s.agent_id in ("qgen_question", "qgen_critic"):
            assert s.segment_id != "task"


def test_revisions_differ_from_baseline() -> None:
    for s in rev.revision_segments():
        assert s.body != _BASELINES[(s.agent_id, s.segment_id)].body


def test_append_strategy_preserves_baseline_prefix() -> None:
    for s in rev.revision_segments():
        baseline = _BASELINES[(s.agent_id, s.segment_id)].body
        if s.strategy == "append":
            assert s.body.startswith(baseline), f"{s.agent_id}/{s.segment_id}"
            assert len(s.body) > len(baseline) + 40
        else:
            assert s.strategy == "replace"
            assert not s.body.startswith(baseline)


def test_added_prose_has_no_em_dashes() -> None:
    for s in rev.revision_segments():
        baseline = _BASELINES[(s.agent_id, s.segment_id)].body
        added = s.body[len(baseline) :] if s.strategy == "append" else s.body
        assert "—" not in added, f"{s.agent_id}/{s.segment_id} added prose has an em dash"


def test_notes_present_and_hashes_consistent() -> None:
    import hashlib

    for s in rev.revision_segments():
        assert s.note.strip(), f"{s.agent_id}/{s.segment_id} note empty"
        assert s.content_hash == hashlib.sha256(s.body.encode("utf-8")).hexdigest()


def test_overrides_map_shape() -> None:
    m = rev.overrides_for("qgen_question")
    assert set(m) == {"role", "examples"}
    assert all(isinstance(v, str) and v.strip() for v in m.values())
    assert rev.overrides_for("oe_moderator").keys() == {"role", "task"}
    with pytest.raises(KeyError):
        rev.overrides_for("weakness_analyser")


def test_familiar_joined_the_wave_with_additive_bodies() -> None:
    """P3: the familiar's revision bodies are pure additive craft text
    (append-at-render happens in the Go composer), so they must be `replace`
    strategy, must NOT start with the baseline template, and every note names
    the append-at-render semantics for the O+ catalogue reader."""
    m = rev.overrides_for("familiar")
    assert set(m) == {"role_frame", "examples_frame", "task_frame"}
    for s in rev.revision_segments():
        if s.agent_id != "familiar":
            continue
        assert s.strategy == "replace", f"{s.segment_id}: familiar bodies are standalone"
        baseline = _BASELINES[("familiar", s.segment_id)].body
        assert not s.body.startswith(baseline)
        assert "append" in s.note.lower(), f"{s.segment_id} note must flag append-at-render"
