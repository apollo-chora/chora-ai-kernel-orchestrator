"""Per-agent domain logic packages.

Each subpackage hosts the pure business logic for one of the 24 agents
(see `chora_ai_kernel_orchestrator.domain.registry.agents`). Agent
modules MUST be free of LLM SDK / HTTP imports — they compose prompt
strings or deterministic outputs from typed inputs and return typed
outputs. The orchestrator pipeline + Model Broker adapters wrap them at
runtime.

Per `feedback_familiar_vs_agent`: a Chora "agent" here is the LLM-aware
component that powers a feature; the in-game Familiar entity (Content
Consumption domain) is a separate aggregate. This package owns the
agent half of that pairing.

Per Tier 5 D20 the runtime of each agent ultimately lives in a Go
executor (M14 BLANKET migration); this package holds the
*orchestration-side* primitives (HITL state machines, study-plan
composition, etc.) the LangGraph nodes exercise before dispatching to
the executor.
"""
