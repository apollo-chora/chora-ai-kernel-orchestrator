"""CHO-2379 - the prompt version-label auto-bump (pure domain rule).

Duplicate ``(agent_id, version_label)`` rows are not cosmetic. The catalogue's
single-version read disambiguates with ``ORDER BY created_at DESC LIMIT 1``, so
every superseded attempt that reuses a label is UNREACHABLE in the O+ modal,
and the same reuse forced the eval-run 409 AlreadyExists suffix workaround in
CHO-2368. ARCHIVED is terminal (ADR-197), so re-promotion always mints a FRESH
plan and therefore always needs a FRESH label.

``next_version_label`` is the single place that picks that label. Pins:

- the first attempt in a family returns the base byte-identical (golden path:
  today's 1.1.0 promotions must not move);
- a later attempt takes family max patch + 1 and NEVER refills a hole, so a
  re-promotion always sorts ABOVE the attempt it supersedes;
- the seeded baseline label ('v1'), foreign families, ``None`` and garbage are
  ignored rather than crashing the promotion lane;
- a malformed base fails loud.
"""

from __future__ import annotations

from typing import Any

import pytest

from chora_ai_kernel_orchestrator.domain.prompt_registry.version_labels import (
    DEFAULT_BASE_LABEL,
    next_version_label,
)


def test_default_base_matches_the_wave_label() -> None:
    assert DEFAULT_BASE_LABEL == "1.1.0"


def test_first_attempt_returns_the_base() -> None:
    assert next_version_label([]) == "1.1.0"


def test_second_attempt_bumps_the_patch() -> None:
    assert next_version_label(["1.1.0"]) == "1.1.1"


def test_third_attempt_bumps_again() -> None:
    assert next_version_label(["1.1.0", "1.1.1"]) == "1.1.2"


def test_holes_are_never_refilled() -> None:
    # A re-promotion must sort ABOVE the attempt it supersedes, so a gap left
    # by a withdrawn 1.1.1 stays a gap.
    assert next_version_label(["1.1.0", "1.1.2"]) == "1.1.3"


def test_baseline_label_and_garbage_are_ignored() -> None:
    # 'v1' is the seeded immutable baseline, not a promotion attempt.
    assert next_version_label(["v1", None, "not-a-version", ""]) == "1.1.0"


def test_foreign_family_never_raises_the_patch() -> None:
    assert next_version_label(["v1", "2.7.9", "1.1.0"]) == "1.1.1"


def test_duplicate_input_labels_collapse() -> None:
    # The live defect itself: three qgen_question plans all labelled 1.1.0.
    assert next_version_label(["1.1.0", "1.1.0", "1.1.0"]) == "1.1.1"


def test_unsorted_input_takes_the_max_not_the_last_seen() -> None:
    assert next_version_label(["1.1.3", "1.1.0", "1.1.1"]) == "1.1.4"


def test_patches_compare_numerically_not_lexically() -> None:
    # Lexical max over ['1.1.0'..'1.1.9'] is '1.1.9' either way; the trap only
    # opens past ten, where '1.1.9' > '1.1.10' as text.
    assert next_version_label([f"1.1.{n}" for n in range(10)]) == "1.1.10"
    assert next_version_label(["1.1.9", "1.1.10"]) == "1.1.11"


def test_custom_base_selects_its_own_family() -> None:
    assert next_version_label(["1.1.0", "1.1.1"], base="2.0.0") == "2.0.0"
    assert next_version_label(["2.0.0", "1.1.9"], base="2.0.0") == "2.0.1"


def test_base_is_the_floor_never_a_lower_label() -> None:
    # A base above the family max must not hand back something below itself.
    assert next_version_label(["1.1.0"], base="1.1.5") == "1.1.5"


def test_accepts_any_iterable_not_just_a_list() -> None:
    assert next_version_label(v for v in ("1.1.0", "v1")) == "1.1.1"


@pytest.mark.parametrize("bad", ["", "1.1", "v1", "1.1.0-rc1", "1.1.x", None, 110])
def test_malformed_base_fails_loud(bad: Any) -> None:
    with pytest.raises(ValueError):
        next_version_label([], base=bad)
