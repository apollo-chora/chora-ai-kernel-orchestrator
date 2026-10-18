"""Shared unit-test doubles.

The qgen crew resolves a job's grounding text from chora-creation BEFORE it
dispatches, and refuses to dispatch when material is declared but unavailable.
A unit test must not perform network I/O and must not depend on
``CREATION_BASE_URL`` being set, so the resolver is replaced here with canned
material.

This is autouse on purpose: every crew unit test that happens to build grounded
state would otherwise reach for a real client. Tests that specifically assert
the REFUSAL path (or that exercise the resolver itself) override this fixture
locally by patching the same target, which wins because their ``monkeypatch``
runs after this one.
"""

from __future__ import annotations

from typing import Any

import pytest

# A shape the Go grounding plugin accepts verbatim: a page-labelled SOURCE block.
STUB_SOURCE_MATERIAL = '<SOURCE file="material.pdf" page="1">material</SOURCE>'


@pytest.fixture(autouse=True)
def stub_source_material(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the creation-backed resolver with canned in-process material."""

    async def _resolved(state: Any, **_kwargs: Any) -> tuple[str, bool]:
        return STUB_SOURCE_MATERIAL, True

    monkeypatch.setattr(
        "chora_ai_kernel_orchestrator.orchestrators.qgen_crew.source_material_text_for_state",
        _resolved,
    )
