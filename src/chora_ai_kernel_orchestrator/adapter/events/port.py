"""AIKernelEventPublisher port (Protocol).

The LangGraph orchestrator emits invocation lifecycle + crew lifecycle +
guardrail events to Pub/Sub. This Protocol is the seam between the
orchestrator state machine and the outbox / pubsub adapters.

Mirrors the closure-orchestrator ``ClosureEventPublisher`` shape:
* every method is async
* every method takes a typed payload dataclass
* implementations: ``TransactionalOutboxPublisher`` (production) +
  in-memory test doubles (added per-test).
"""

from __future__ import annotations

from typing import Protocol

from chora_ai_kernel_orchestrator.adapter.events.payloads import (
    AgentTerminated,
    CrewComposed,
    CrewExecuted,
    GuardrailEvaluated,
    ModelInvocationCompleted,
    ModelInvocationFailed,
    ModelInvoked,
)


class AIKernelEventPublisher(Protocol):
    """Orchestrator-side event publisher port."""

    async def publish_model_invoked(self, e: ModelInvoked) -> None: ...

    async def publish_model_invocation_completed(self, e: ModelInvocationCompleted) -> None: ...

    async def publish_model_invocation_failed(self, e: ModelInvocationFailed) -> None: ...

    async def publish_crew_composed(self, e: CrewComposed) -> None: ...

    async def publish_crew_executed(self, e: CrewExecuted) -> None: ...

    async def publish_guardrail_evaluated(self, e: GuardrailEvaluated) -> None: ...

    async def publish_agent_terminated(self, e: AgentTerminated) -> None: ...
