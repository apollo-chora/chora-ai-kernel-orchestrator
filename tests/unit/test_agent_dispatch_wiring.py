"""RED: composition of the ADR-253 dispatch lane.

The rule this file exists to enforce: an agent-dispatch lane must NEVER fall
back to the ordinary shared checkpointer. That checkpointer opens its own
autocommit connection, so a lane running on it would write dispatch rows in a
separate transaction from the park, which is exactly the shape ADR-253 D3a was
ruled against. It would also still work, most of the time, which is what makes
the silent fallback the dangerous one: the ratified property would simply be
false while every test and every screenshot looked fine.

2026-08-23 (RULING A): the per-deployment transport switch is GONE with the
HTTP executor. There is no `http` arm to fall back to any more, so the guard is
unconditional: no `transport` argument, and every caller is refused a saver
that is not the transactional one.
"""

from __future__ import annotations

import pytest


def _mod():
    from chora_ai_kernel_orchestrator.adapter.pubsub import agent_dispatch_wiring

    return agent_dispatch_wiring


def test_the_transport_switch_is_gone() -> None:
    """No env flip can select a transport any more: Pub/Sub dispatch is the
    only one there is. A leftover switch would be a hatch to a deleted arm."""
    mod = _mod()
    for name in (
        "dispatch_transport_from_env",
        "TRANSPORT_HTTP",
        "TRANSPORT_PUBSUB",
        "ENV_DISPATCH_TRANSPORT",
        "ENV_WEAKNESS_DISPATCH_TRANSPORT",
    ):
        assert not hasattr(mod, name), (
            f"{name} survived the HTTP executor deletion; the transport switch must not outlive the arm it selected"
        )


async def test_a_non_transactional_saver_is_refused() -> None:
    """The load-bearing guard. Handing a lane an ordinary checkpointer would
    silently split the park from the dispatch row."""
    from chora_ai_kernel_orchestrator.adapter.checkpointer.factory import (
        LazyPostgresSaver,
    )

    with pytest.raises(ValueError, match="TransactionalDispatchSaver"):
        _mod().require_transactional_saver(LazyPostgresSaver(dsn="postgresql://x/y", builder=object()))


async def test_a_missing_saver_is_refused() -> None:
    with pytest.raises(ValueError, match="TransactionalDispatchSaver"):
        _mod().require_transactional_saver(None)


async def test_the_transactional_saver_passes_through() -> None:
    """The one shape that is allowed to start a lane."""
    from unittest.mock import MagicMock

    from chora_ai_kernel_orchestrator.adapter.checkpointer.transactional_dispatch_saver import (  # noqa: E501
        TransactionalDispatchSaver,
    )

    saver = TransactionalDispatchSaver(
        inner=MagicMock(),
        conn=MagicMock(),
        outbox_writer=MagicMock(),
    )
    assert _mod().require_transactional_saver(saver) is saver


def test_the_lane_is_named_in_the_refusal() -> None:
    """The qgen, OE, weakness and single-agent lanes share this guard, so the
    refusal has to say which one refused to start."""
    with pytest.raises(ValueError, match="growth edge"):
        _mod().require_transactional_saver(None, lane="growth edge")


def test_the_dispatch_roles_for_the_oe_lane_are_the_executor_roles() -> None:
    """The topics are keyed on the ROLE (oe_evaluate), not the agent id
    (oe_evaluator). Mixing them subscribes to topics nobody publishes to."""
    from chora_ai_kernel_orchestrator.adapter.agent_io import (
        ROLE_OE_EVALUATE,
        ROLE_OE_MODERATE,
    )

    assert _mod().OE_DISPATCH_ROLES == (ROLE_OE_EVALUATE, ROLE_OE_MODERATE)
    assert _mod().OE_DISPATCH_ROLES == ("oe_evaluate", "oe_moderate")
