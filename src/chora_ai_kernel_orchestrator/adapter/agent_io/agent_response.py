"""The mapped agent response every crew node consumes.

Wire history: this DTO began as the mapped ``ExecuteAgentResponse`` of the
gRPC ``chora-services-agent-executor`` (Tier 2 D5), then served the HTTP
Vertex Agent Engine executor, and now carries the Pub/Sub completion payload.
Three transports, one shape, which is exactly why it outlived all three of
their clients: the gRPC executors were deleted with ADR-145, the HTTP executor
with RULING A (2026-08-23), and what the crew nodes read never changed.

It lives in ``adapter/agent_io`` with the session-state builder and the
response mapper that produces it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentExecutorResponse:
    """What the orchestrator nodes consume after a hop.

    C3 (CR qgen 2026-06-01): ``input_tokens`` + ``output_tokens`` carry the
    per-hop split the qgen agent JSON now reports at the top level. The
    qgen_crew ``generate`` / ``critique`` nodes stamp them onto the
    pipeline_trace rows so the W6 managed-Eval token-compounding criterion
    reads real numbers (it previously read 0 because only
    ``tokens_consumed_total`` was surfaced). Both default 0 for the M11
    baseline shape (executors that only ever emitted the total).
    """

    execution_id: str
    output_payload: str
    tokens_consumed_total: int = 0
    cost_micros_total: int = 0
    final_state: str = "EXECUTION_FINAL_STATE_UNSPECIFIED"
    error_message: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
