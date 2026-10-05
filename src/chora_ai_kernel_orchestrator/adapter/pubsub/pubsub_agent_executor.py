"""PubSubAgentExecutor — the suspend-and-resume dispatch adapter (ADR-253 D1/D2).

Satisfies the ``_ExecutorLike`` protocol the crew nodes call
(``orchestrators/qgen_crew.py``), so **crew nodes are not rewritten**: a node
still writes ``resp = await executor.execute(...)`` and still reads an
``AgentExecutorResponse``. What changes is underneath. Instead of awaiting an
HTTP POST for the length of a model call (the executor that did, deleted
2026-08-23), ``execute``:

  1. builds the dispatch request (mandatory envelope + body + topic), then
  2. calls LangGraph's ``interrupt()``, which **parks the run**. The saver
     commits the park and the dispatch outbox row in one transaction
     (ADR-253 D3a), the dispatcher publishes, and this coroutine ends.
  3. When the completion consumer resumes the thread, ``interrupt()`` returns
     the completion payload and this function maps it to an
     ``AgentExecutorResponse`` as if the call had simply taken a long time.

⚠ Node re-execution is not a bug to work around, it is the contract. LangGraph
re-runs a node from the top on resume, so everything before ``interrupt()``
runs twice. That is safe here ONLY because this method's pre-park work is pure:
it builds a request and does not publish. The publish is the saver's, off the
committed park. The idempotency key is derived from ``execution_id`` so the
rebuilt request collides with the queued row rather than dispatching twice.

⚠ Failure is a status, not an exception on the wire (ADR-253 D4). A completion
carrying ``status=FAILED`` is raised HERE, in the node, so the crew's existing
error handling sees an executor failure exactly as it would have seen an HTTP
failure. A missing status is refused rather than read as success by omission.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from typing import Any

from langgraph.types import interrupt

from chora_ai_kernel_orchestrator.adapter.agent_io import map_agent_response
from chora_ai_kernel_orchestrator.adapter.agent_io.agent_response import (
    AgentExecutorResponse,
)
from chora_ai_kernel_orchestrator.adapter.pubsub.agent_dispatch import (
    STATUS_OK,
    build_dispatch_request,
    wrap_for_interrupt,
)

logger = logging.getLogger(__name__)

FINAL_STATE_SUCCEEDED = "EXECUTION_FINAL_STATE_SUCCEEDED"


class AgentDispatchError(RuntimeError):
    """An agent reported that it could not complete the dispatched work."""


class PubSubAgentExecutor:
    """Pub/Sub request/completion implementation of the ``_ExecutorLike`` seam."""

    def __init__(
        self,
        *,
        source_project: str,
        allowed_roles: Iterable[str] | None = None,
        source_service: str = "chora-ai-kernel-orchestrator",
    ) -> None:
        if not (source_project or "").strip():
            raise ValueError("source_project required")
        self._source_project = source_project
        self._source_service = source_service
        # None means "any role"; a configured set makes an unrouteable role fail
        # loud BEFORE the park, because a run parked on a topic pair that does
        # not exist can never be resumed by anything.
        self._allowed_roles = set(allowed_roles) if allowed_roles is not None else None

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
        if self._allowed_roles is not None and agent_role not in self._allowed_roles:
            raise KeyError(
                f"unknown agent_role={agent_role!r} for Pub/Sub dispatch; "
                f"configured roles: {sorted(self._allowed_roles)}"
            )
        if context_window or available_tools:
            # The HTTP executor ignores both today (the crews thread everything
            # through input_payload). Refusing loudly beats silently dropping a
            # caller's context on a new transport.
            raise ValueError(
                "PubSubAgentExecutor: context_window / available_tools are not "
                "carried on the dispatch wire; thread them via input_payload"
            )

        input_obj = _decode(input_payload)
        request = build_dispatch_request(
            agent_role=agent_role,
            execution_id=execution_id,
            tenant_id=tenant_id,
            gcid=str(input_obj.get("gcid") or input_obj.get("author_gcid") or ""),
            thread_id=_thread_id(),
            input_payload=input_payload,
            # Defaults to thread_id, which is what OE and qgen rely on. A lane
            # whose thread key is not a UUID must pass its own aggregate root,
            # because ai_kernel_outbox_events.workflow_id is UUID NOT NULL.
            workflow_id=workflow_id,
            prompt_template_id=prompt_template_id,
            agid=agid,
            traceparent=str(input_obj.get("traceparent") or ""),
            tracestate=str(input_obj.get("tracestate") or ""),
            source_project=self._source_project,
            source_service=self._source_service,
        )

        # ---- park. Everything above is pure and safe to repeat. ----
        completion = interrupt(wrap_for_interrupt(request))
        # ---- resumed. Below runs only once, on the completion. ----

        return _map_completion(completion, execution_id=execution_id, agent_role=agent_role)


def _thread_id() -> str:
    """The thread the run is parked on, read from LangGraph's own config.

    Taken from the runtime rather than passed in, so the value the completion
    consumer must resume is by construction the value LangGraph parked under —
    a threaded-through copy could drift from it.
    """
    from langgraph.config import get_config

    config = get_config() or {}
    thread_id = str((config.get("configurable") or {}).get("thread_id") or "")
    if not thread_id:
        raise ValueError(
            "PubSubAgentExecutor: no thread_id on the graph config; a dispatch "
            "parked without one could never be resumed"
        )
    return thread_id


def _decode(payload: str) -> dict[str, Any]:
    try:
        obj = json.loads(payload or "{}")
    except (TypeError, ValueError):
        return {}
    return obj if isinstance(obj, dict) else {}


def _map_completion(completion: Any, *, execution_id: str, agent_role: str) -> AgentExecutorResponse:
    """Map a completion payload to the response the crew nodes already read."""
    if not isinstance(completion, dict):
        raise AgentDispatchError(
            f"{agent_role} completion for {execution_id} was {type(completion).__name__}, expected an object"
        )
    status = str(completion.get("status") or "").strip().upper()
    if not status:
        raise AgentDispatchError(
            f"{agent_role} completion for {execution_id} carries no status; "
            "refusing to read a missing status as success"
        )
    if status != STATUS_OK:
        raise AgentDispatchError(
            f"{agent_role} dispatch {execution_id} returned status={status}: "
            f"{completion.get('error_message') or 'no error_message'}"
        )
    # Transport parity (ADR-253 D6). The agent's terminal JSON is normalised by
    # the SAME mapper the HTTP executor uses, so both transports produce an
    # identical AgentExecutorResponse from identical agent output. That is what
    # makes "select the HTTP executor and redeploy" a real rollback rather than
    # a behaviour change: the token split in particular is derived from the
    # agent's own emitted JSON, and re-deriving it here would quietly rewrite
    # every pipeline_trace row and every O+ per-agent token tile.
    mapped = map_agent_response(
        execution_id=execution_id,
        terminal_text=str(completion.get("output_payload") or ""),
        agent_role=agent_role,
    )
    # Envelope-level token counts are a FALLBACK, used only when the agent's
    # payload carried none. The payload is authoritative because that is where
    # the HTTP path reads them; two live sources of truth would drift.
    if mapped.input_tokens or mapped.output_tokens or mapped.tokens_consumed_total:
        return mapped
    wire_total = _int(completion.get("tokens_consumed_total"))
    wire_in = _int(completion.get("input_tokens"))
    wire_out = _int(completion.get("output_tokens"))
    if not (wire_total or wire_in or wire_out):
        return mapped
    return AgentExecutorResponse(
        execution_id=mapped.execution_id,
        output_payload=mapped.output_payload,
        tokens_consumed_total=wire_total or (wire_in + wire_out),
        cost_micros_total=mapped.cost_micros_total or _int(completion.get("cost_micros_total")),
        final_state=mapped.final_state,
        error_message=mapped.error_message,
        input_tokens=wire_in,
        output_tokens=wire_out,
    )


def _int(value: Any) -> int:
    try:
        return max(int(value), 0)
    except (TypeError, ValueError):
        return 0


__all__ = ["AgentDispatchError", "PubSubAgentExecutor"]
