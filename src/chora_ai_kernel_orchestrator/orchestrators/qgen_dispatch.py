"""QGenDispatchAdapter: the kennel side of the qgen Pub/Sub lanes (ADR-254 D2).

The qgen graph nodes keep the ``_ExecutorLike`` call shape they have always
had (``executor.execute(agent_role=qgen_question | qgen_critic, ...)``). This
adapter sits between them and the ``PubSubAgentExecutor`` and does the three
things the flip needs, in one place rather than in five nodes:

1. maps the AGENT id the node names onto the LANE role the topics are keyed on
   (``qgen_question`` -> ``qgen_generate``, ``qgen_critic`` -> ``qgen_critique``;
   ``qgen_render`` is already a lane role) while keeping ``agid`` = the agent id
   (the Armor tier key and the decision-log identity);
2. builds the session-state-shaped payload the subscriber-only Go binaries
   merge verbatim (``build_session_state``, the transform the HTTP executor
   used to apply on its side of the wire), keeping ``gcid`` / trace context in
   the body so the executor lifts them into the envelope;
3. requires an explicit ``workflow_id``: qgen thread ids are assist ids, and the
   outbox's ``workflow_id`` column is UUID NOT NULL, so the job id is passed
   rather than guessed from the thread.

It is deliberately NOT an ``interrupt()`` site itself: the park belongs to the
inner executor, so everything here is pure and safe to re-run on resume.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from chora_ai_kernel_orchestrator.adapter.agent_io import (
    CRITIC_ROLES,
    LANE_ROLE_CRITIQUE,
    LANE_ROLE_GENERATE,
    LANE_ROLE_RENDER,
    QUESTION_ROLES,
    build_session_state,
    decode_input,
)
from chora_ai_kernel_orchestrator.adapter.agent_io.agent_response import (
    AgentExecutorResponse,
)

logger = logging.getLogger(__name__)

# The three roles the qgen lane dispatches; the runtime binds all three to the
# qgen runner so a completion on any of them resumes the parked job.
QGEN_LANE_ROLES: tuple[str, ...] = (LANE_ROLE_GENERATE, LANE_ROLE_CRITIQUE, LANE_ROLE_RENDER)

# Keys the Pub/Sub executor lifts into the envelope; kept in the body as well
# so the executor can read them (the Go merge skips them from the body).
_ENVELOPE_KEYS = ("gcid", "author_gcid", "traceparent", "tracestate")


def lane_role_for(agent_role: str) -> str:
    """The dispatch lane for a crew role; KeyError for anything not qgen's."""
    role = (agent_role or "").strip()
    if role in QUESTION_ROLES:
        return LANE_ROLE_GENERATE
    if role in CRITIC_ROLES:
        return LANE_ROLE_CRITIQUE
    if role == LANE_ROLE_RENDER:
        return LANE_ROLE_RENDER
    raise KeyError(f"QGenDispatchAdapter: {agent_role!r} is not a qgen role; lanes: {list(QGEN_LANE_ROLES)}")


class QGenDispatchAdapter:
    """``_ExecutorLike`` over a ``PubSubAgentExecutor`` pinned to the qgen lanes."""

    def __init__(self, inner: Any) -> None:
        if inner is None or not hasattr(inner, "execute"):
            raise ValueError("QGenDispatchAdapter: an inner executor exposing execute() is required")
        self._inner = inner

    async def execute(
        self,
        *,
        execution_id: str,
        tenant_id: str,
        agid: str,
        agent_role: str,
        input_payload: str,
        workflow_id: str = "",
        prompt_template_id: str = "",
        context_window: list[dict[str, str]] | None = None,
        available_tools: list[str] | None = None,
    ) -> AgentExecutorResponse:
        lane_role = lane_role_for(agent_role)
        if not (workflow_id or "").strip():
            raise ValueError(
                f"QGenDispatchAdapter: execution {execution_id!r} on {agent_role!r} carries no "
                "workflow_id; the qgen thread id is not the outbox's workflow key, pass the job id"
            )
        if context_window or available_tools:
            raise ValueError(
                "QGenDispatchAdapter: context_window / available_tools are not carried on the "
                "dispatch wire; thread them via input_payload"
            )
        input_obj = decode_input(input_payload)
        if not isinstance(input_obj, dict):
            input_obj = {"raw": input_payload}
        state = build_session_state(
            agent_role=lane_role,
            execution_id=execution_id,
            tenant_id=tenant_id,
            input_obj=input_obj,
        )
        for key in _ENVELOPE_KEYS:
            value = input_obj.get(key)
            if value and key not in state:
                state[key] = value
        return await self._inner.execute(
            execution_id=execution_id,
            tenant_id=tenant_id,
            agid=agid,
            agent_role=lane_role,
            input_payload=json.dumps(state, sort_keys=True),
            workflow_id=workflow_id,
            prompt_template_id=prompt_template_id,
        )


__all__ = ["QGEN_LANE_ROLES", "QGenDispatchAdapter", "lane_role_for"]
