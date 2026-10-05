"""RED: ADR-254 D5: every parked run carries a deadline.

The deadline is computed at park time from a policy, never left implicit:

    deadline_at = parked_at + min(RETENTION, role_deadline)

``role_deadline`` defaults to the lane's message retention (604800 s, the
``message_retention_duration`` every agent-dispatch subscription carries in
``agent_dispatch_lanes.tf``). Past retention the request is CERTAINLY gone, so
that is the physical truth the default encodes; a tighter per-role value is an
explicit choice for lanes where a learner is waiting (ADR-254 D5: typed
``companion_chat`` = 120 s) so they get ``FAILED`` rather than silence.

The policy is pure domain logic: no clock, no I/O. Time is passed in.
"""

from __future__ import annotations

import datetime as _dt

import pytest

from chora_ai_kernel_orchestrator.domain.agent_dispatch.deadline_policy import (
    ENV_PARK_DEADLINE_DEFAULT,
    ENV_PARK_DEADLINE_ROLE_PREFIX,
    RETENTION_SECONDS,
    ParkDeadlinePolicy,
)

_PARKED_AT = _dt.datetime(2026, 8, 22, 12, 0, 0, tzinfo=_dt.UTC)


def test_retention_is_the_lane_shape_seven_days() -> None:
    """604800 s is read off agent_dispatch_lanes.tf, not invented here."""
    assert RETENTION_SECONDS == 604_800


def test_default_deadline_is_the_retention() -> None:
    policy = ParkDeadlinePolicy()
    assert policy.seconds_for("oe_evaluate") == RETENTION_SECONDS
    assert policy.deadline_for("oe_evaluate", parked_at=_PARKED_AT) == (
        _PARKED_AT + _dt.timedelta(seconds=RETENTION_SECONDS)
    )


def test_per_role_tightening_applies_only_to_that_role() -> None:
    policy = ParkDeadlinePolicy(per_role_seconds={"companion_chat": 120})
    assert policy.seconds_for("companion_chat") == 120
    assert policy.seconds_for("oe_evaluate") == RETENTION_SECONDS
    assert policy.deadline_for("companion_chat", parked_at=_PARKED_AT) == (_PARKED_AT + _dt.timedelta(seconds=120))


def test_a_tighter_default_applies_to_every_unlisted_role() -> None:
    policy = ParkDeadlinePolicy(default_seconds=3600)
    assert policy.seconds_for("recommend") == 3600


@pytest.mark.parametrize("bad", [0, -5, RETENTION_SECONDS + 1])
def test_out_of_range_seconds_are_refused_at_construction(bad: int) -> None:
    """A deadline past retention is a lie (the message is already gone) and a
    non-positive one parks nothing; both are configuration errors, refused loud."""
    with pytest.raises(ValueError):
        ParkDeadlinePolicy(per_role_seconds={"recommend": bad})
    with pytest.raises(ValueError):
        ParkDeadlinePolicy(default_seconds=bad)


def test_a_naive_parked_at_is_refused() -> None:
    policy = ParkDeadlinePolicy()
    with pytest.raises(ValueError):
        policy.deadline_for("recommend", parked_at=_dt.datetime(2026, 8, 22, 12, 0, 0))


def test_from_env_reads_the_default_and_per_role_overrides() -> None:
    env = {
        ENV_PARK_DEADLINE_DEFAULT: "3600",
        f"{ENV_PARK_DEADLINE_ROLE_PREFIX}COMPANION_CHAT": "120",
    }
    policy = ParkDeadlinePolicy.from_env(env)
    assert policy.seconds_for("companion_chat") == 120
    assert policy.seconds_for("oe_evaluate") == 3600


def test_from_env_unset_or_blank_means_the_retention() -> None:
    assert ParkDeadlinePolicy.from_env({}).seconds_for("x") == RETENTION_SECONDS
    assert ParkDeadlinePolicy.from_env({ENV_PARK_DEADLINE_DEFAULT: "  "}).seconds_for("x") == RETENTION_SECONDS


@pytest.mark.parametrize("raw", ["abc", "0", "-1", "999999999", "12.5"])
def test_from_env_refuses_a_bad_value_instead_of_silently_defaulting(raw: str) -> None:
    with pytest.raises(ValueError):
        ParkDeadlinePolicy.from_env({ENV_PARK_DEADLINE_DEFAULT: raw})
    with pytest.raises(ValueError):
        ParkDeadlinePolicy.from_env({f"{ENV_PARK_DEADLINE_ROLE_PREFIX}RECOMMEND": raw})


def test_env_names_are_the_documented_ones() -> None:
    assert ENV_PARK_DEADLINE_DEFAULT == "AGENT_DISPATCH_PARK_DEADLINE_SECONDS"
    assert ENV_PARK_DEADLINE_ROLE_PREFIX == "AGENT_DISPATCH_PARK_DEADLINE_SECONDS_"
