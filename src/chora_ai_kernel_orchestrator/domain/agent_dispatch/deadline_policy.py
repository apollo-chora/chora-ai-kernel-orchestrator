"""ParkDeadlinePolicy: ADR-254 D5: every parked run carries a deadline.

    deadline_at = parked_at + min(RETENTION_SECONDS, role_deadline)

``role_deadline`` defaults to the lane's message retention (604800 s, the
``message_retention_duration`` every agent-dispatch subscription carries in
``chora-infra/terraform/environments/dev/agent_dispatch_lanes.tf``). Past
retention the request is CERTAINLY gone, so the default is the physical truth
rather than a guess, and work parked during an agent outage (the OE requests
of 2026-08-21) drains inside the window instead of being reaped. A tighter
value is an explicit per-role choice for lanes where a learner is waiting
(typed ``companion_chat`` = 120 s per ADR-254 D5), configured through env so a
deployment can tune it without a rebuild ([[secrets-and-env]]).

Pure: no clock, no I/O. ``from_env`` takes a mapping so it is testable.
"""

from __future__ import annotations

import datetime as _dt
import os
from collections.abc import Mapping
from dataclasses import dataclass, field

# agent_dispatch_lanes.tf: message_retention_duration = "604800s" (7 days).
RETENTION_SECONDS = 604_800

ENV_PARK_DEADLINE_DEFAULT = "AGENT_DISPATCH_PARK_DEADLINE_SECONDS"
ENV_PARK_DEADLINE_ROLE_PREFIX = "AGENT_DISPATCH_PARK_DEADLINE_SECONDS_"


def _validate_seconds(value: int, *, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{what}: deadline seconds must be an int, got {value!r}")
    if value < 1:
        raise ValueError(f"{what}: deadline seconds must be >= 1, got {value}")
    if value > RETENTION_SECONDS:
        raise ValueError(
            f"{what}: deadline seconds {value} exceeds the lane retention "
            f"{RETENTION_SECONDS}; a park cannot outlive its request"
        )
    return value


@dataclass(frozen=True)
class ParkDeadlinePolicy:
    default_seconds: int = RETENTION_SECONDS
    per_role_seconds: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_seconds(self.default_seconds, what="default")
        for role, seconds in self.per_role_seconds.items():
            if not (role or "").strip():
                raise ValueError("per-role deadline: blank role")
            _validate_seconds(seconds, what=f"role {role!r}")

    def seconds_for(self, agent_role: str) -> int:
        return self.per_role_seconds.get(agent_role, self.default_seconds)

    def deadline_for(self, agent_role: str, *, parked_at: _dt.datetime) -> _dt.datetime:
        if parked_at.tzinfo is None or parked_at.utcoffset() is None:
            raise ValueError("parked_at must be timezone-aware")
        seconds = min(RETENTION_SECONDS, self.seconds_for(agent_role))
        return parked_at + _dt.timedelta(seconds=seconds)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> ParkDeadlinePolicy:
        """Build from ``AGENT_DISPATCH_PARK_DEADLINE_SECONDS`` (default) and
        ``AGENT_DISPATCH_PARK_DEADLINE_SECONDS_<ROLE>`` (per role, role upper-cased).

        Unset or blank means the retention. A present-but-bad value is refused
        loud: a typo that silently fell back to seven days would look like a
        configured tight deadline that never fires.
        """
        source = os.environ if env is None else env
        default_seconds = RETENTION_SECONDS
        raw_default = (source.get(ENV_PARK_DEADLINE_DEFAULT) or "").strip()
        if raw_default:
            default_seconds = _parse_seconds(raw_default, what=ENV_PARK_DEADLINE_DEFAULT)
        per_role: dict[str, int] = {}
        for key, raw in source.items():
            if not key.startswith(ENV_PARK_DEADLINE_ROLE_PREFIX):
                continue
            role = key[len(ENV_PARK_DEADLINE_ROLE_PREFIX) :].lower()
            value = (raw or "").strip()
            if not role:
                raise ValueError(f"{key}: blank role suffix")
            if not value:
                continue
            per_role[role] = _parse_seconds(value, what=key)
        return cls(default_seconds=default_seconds, per_role_seconds=per_role)


def _parse_seconds(raw: str, *, what: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{what}={raw!r} is not an integer number of seconds") from exc
    return _validate_seconds(value, what=what)


__all__ = [
    "ENV_PARK_DEADLINE_DEFAULT",
    "ENV_PARK_DEADLINE_ROLE_PREFIX",
    "RETENTION_SECONDS",
    "ParkDeadlinePolicy",
]
