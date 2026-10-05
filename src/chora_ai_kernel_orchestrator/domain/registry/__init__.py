"""24-agent registry domain package.

Exposes the immutable cast of 24 AI agents from the Sequel comic alongside
the capability taxonomy used to filter them. Adapter layers (HTTP / gRPC)
import from this package only — they never mutate the registry.
"""

from chora_ai_kernel_orchestrator.domain.registry.agents import (
    AGENT_REGISTRY,
    Agent24Registry,
    AgentDescriptor,
    AgentType,
    RiskTier,
)
from chora_ai_kernel_orchestrator.domain.registry.capabilities import Capability

__all__ = [
    "AGENT_REGISTRY",
    "Agent24Registry",
    "AgentDescriptor",
    "AgentType",
    "Capability",
    "RiskTier",
]
