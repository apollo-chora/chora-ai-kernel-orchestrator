"""The Python port of consumption's `proseNarrative` predicate, pinned against
the SHARED vector file that the Go original reads too.

Why this test exists in this shape: the dose lane decides OK vs FAILED on
whether the recommender's rationale is genuine learner-facing prose. Consumption
drops a non-prose narrative server side, so a kennel that said OK on a
JSON-shaped rationale would hand the learner a silently empty dose, which is
exactly the outcome the ruling exists to prevent.

The predicate therefore exists twice, in two languages, which is a drift risk.
The mitigation is that there is ONE vector file and both sides read it: adding a
vector must fail whichever side does not handle it. The vectors are the
contract; the port is just code.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chora_ai_kernel_orchestrator.orchestrators.prose_narrative import prose_narrative


def _vectors_path() -> Path:
    """Read the ONE shared vector file, vendored into this repo at
    ``testdata/prose_narrative_vectors.json``.

    The standalone repo vendors the shared vector file (it was originally read
    from the chora-contracts checkout) so there is no cross-repo reference. The
    vectors are the contract; the port is just code.
    """
    path = Path(__file__).resolve().parents[2] / "testdata" / "prose_narrative_vectors.json"
    assert path.is_file(), f"shared prose vector file not found: {path}"
    return path


def _load_vectors() -> list[dict[str, str]]:
    data = json.loads(_vectors_path().read_text(encoding="utf-8"))
    vectors = data["vectors"]
    assert vectors, "the shared vector file is empty; the test would be vacuous"
    return vectors


def test_the_vector_file_is_the_shared_one_not_a_local_copy() -> None:
    """Positive control on the instrument itself: a vector file this test
    invented would make every assertion below vacuous."""
    path = _vectors_path()
    assert path.name == "prose_narrative_vectors.json"
    assert path.parent.name == "testdata"
    assert len(_load_vectors()) >= 10, "too few vectors to discriminate"


@pytest.mark.parametrize("vector", _load_vectors(), ids=lambda v: v["name"])
def test_prose_narrative_matches_the_shared_vectors(vector: dict[str, str]) -> None:
    assert prose_narrative(vector["in"]) == vector["out"]


def test_the_fence_unwrap_is_not_optional() -> None:
    """A port that skips the markdown-fence unwrap still passes the naive
    bare-JSON vectors while accepting FENCED json at a learner. Pinned
    separately so that regression cannot hide behind the parametrised set."""
    assert prose_narrative('```json\n{"a":1}\n```') == ""
    assert prose_narrative("```\nYou are doing well.\n```") == "You are doing well."
