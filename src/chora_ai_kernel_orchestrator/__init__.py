"""Chora AI Kernel Orchestrator (Python LangGraph orchestrator).

Hybrid kernel per Tier 2 D5: this Python service holds the LangGraph state
machine + checkpointer; Go executors handle stateless agent execution and
are called via gRPC. The orchestrator also speaks HTTP to the Guardrail
Service for input/output screening.
"""

__version__ = "0.1.0"
