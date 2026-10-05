"""RED: ADR-254 D5, an ENABLED lane that cannot build refuses startup.

``main.py`` used to catch a lane's startup failure, log it, set the components
to ``None`` and carry on; ``/readyz`` then reported ready on the guardrail
alone. ``require_lane_built`` is the rule the lifespan applies to every enabled
lane: ``None`` from the builder is a configuration error and the process exits.
"""

from __future__ import annotations

import pytest

from chora_ai_kernel_orchestrator.adapter.pubsub.kennel_runtime import (
    require_lane_built,
)


def test_a_built_lane_passes_through_unchanged() -> None:
    components = object()
    assert require_lane_built("oe_grading_crew", components) is components


def test_none_from_an_enabled_lane_builder_is_refused_by_name() -> None:
    with pytest.raises(RuntimeError, match="oe_grading_crew is enabled"):
        require_lane_built("oe_grading_crew", None)
