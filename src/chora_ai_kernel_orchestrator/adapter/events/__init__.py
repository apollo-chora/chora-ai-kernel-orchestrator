"""AI Kernel orchestrator event publisher port + payload types + outbox adapter.

Topics emitted by the LangGraph orchestrator (canonical chora.ai_kernel.*
per chora-contracts/CLAUDE.md taxonomy):

- ``chora.ai_kernel.invocation.invoked.v1`` — model invocation start
- ``chora.ai_kernel.invocation.completed.v1`` — model invocation success
- ``chora.ai_kernel.invocation.failed.v1`` — model invocation failure
- ``chora.ai_kernel.crew.composed.v1`` — crew instantiated for workflow
- ``chora.ai_kernel.crew.executed.v1`` — crew run completed
- ``chora.ai_kernel.guardrail.evaluated.v1`` — guardrail tier outcome

Wire contract: ``chora-contracts/proto/events/ai_kernel/*.proto``.
"""

from chora_ai_kernel_orchestrator.adapter.events.payloads import (
    AgentTerminated,
    CrewComposed,
    CrewExecuted,
    GuardrailEvaluated,
    ModelInvocationCompleted,
    ModelInvocationFailed,
    ModelInvoked,
)
from chora_ai_kernel_orchestrator.adapter.events.port import (
    AIKernelEventPublisher,
)
from chora_ai_kernel_orchestrator.adapter.events.publisher_outbox import (
    SCHEMA_VERSION,
    TransactionalOutboxPublisher,
)

__all__ = [
    "AIKernelEventPublisher",
    "AgentTerminated",
    "CrewComposed",
    "CrewExecuted",
    "GuardrailEvaluated",
    "ModelInvocationCompleted",
    "ModelInvocationFailed",
    "ModelInvoked",
    "SCHEMA_VERSION",
    "TransactionalOutboxPublisher",
]
