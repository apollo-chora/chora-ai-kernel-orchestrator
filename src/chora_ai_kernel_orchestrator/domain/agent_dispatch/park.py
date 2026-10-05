"""ParkRecord: the durable fact "this thread is parked on role R since T, deadline D".

ADR-254 D5: every parked run carries a deadline. The record is written by the
transactional dispatch saver in the SAME transaction as the LangGraph park and
the dispatch outbox row (migration ``0056_agent_dispatch_parks``), settled by
the completion router when the agent answers, and by the reaper when it does
not. It is the only place a park is queryable by age: LangGraph's checkpoint
tables carry no application timestamp and the outbox row's ``occurred_at``
vanishes under any pruning.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from enum import StrEnum

_KEY_PREFIX = "agent_dispatch."


class ParkState(StrEnum):
    """Lifecycle of one park. ``parked`` -> ``completed`` (the agent answered)
    or ``parked`` -> ``reaped`` (the reaper settled it FAILED). Terminal
    states never go back to ``parked``; a completion that arrives after a reap
    is recorded on the row (``late_completion_at``) and never resumes."""

    PARKED = "parked"
    COMPLETED = "completed"
    REAPED = "reaped"


@dataclass(frozen=True, slots=True)
class ParkRecord:
    idempotency_key: str
    workflow_id: str
    thread_id: str
    crew: str
    tenant_id: str
    gcid: str
    agent_role: str
    request_topic: str
    completion_topic: str
    traceparent: str
    tracestate: str
    parked_at: _dt.datetime
    deadline_at: _dt.datetime
    state: ParkState = ParkState.PARKED
    settled_at: _dt.datetime | None = None
    settled_by: str = ""
    late_completion_at: _dt.datetime | None = None

    @property
    def is_parked(self) -> bool:
        return self.state is ParkState.PARKED

    @property
    def execution_id(self) -> str:
        """The dispatch's ``execution_id``, recovered from the request key.

        ``agent_dispatch.dispatch_idempotency_key`` builds the key as
        ``agent_dispatch.{role}.{execution_id}``; a role never contains a dot,
        so the third dot-separated segment onward IS the execution id (which
        may itself contain dots).
        """
        key = self.idempotency_key
        if not key.startswith(_KEY_PREFIX):
            raise ValueError(f"not an agent-dispatch key: {key!r}")
        rest = key[len(_KEY_PREFIX) :]
        role, sep, execution_id = rest.partition(".")
        if not sep or not role or not execution_id:
            raise ValueError(f"malformed agent-dispatch key: {key!r}")
        return execution_id


__all__ = ["ParkRecord", "ParkState"]
