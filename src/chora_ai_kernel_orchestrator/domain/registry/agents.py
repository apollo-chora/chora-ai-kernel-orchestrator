"""24-agent registry — Sequel comic cast.

Source: docs/comic/Sequel_Chora-Agentic-Customer-journeys.md, "The 24 AI Agents"
section. Agents #1-#22 appear in the comic's "Behind the Button" cutaways;
#23 (MCP Tool Router) and #24 (Story Point Estimator) extend the cast for
A2A + Campus Ops value streams referenced in Acts 13-16.

The registry is a read-only domain object — adapter layers (HTTP / gRPC) read
from `AGENT_REGISTRY` but never mutate it. Per Tier 5 D20 the *runtime* of
each agent lives in a Go executor (M14 BLANKET migration); this registry
gives orchestrator + Model Broker a stable id-and-metadata surface to plan
crew composition + cost forecasting against.

Per `feedback_familiar_vs_agent`: #19 Familiar here is the **AI agent that
powers** the in-game Familiar entity (which lives in Content Consumption
domain). The two are deliberately distinct: registry = adapter concern;
Familiar entity = aggregate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from chora_ai_kernel_orchestrator.domain.registry.capabilities import Capability


class AgentType(StrEnum):
    """High-level agent flavour used by the Model Broker for routing.

    LLM     = generative model call (Gemini / Gemma) — billable.
    ML      = classical ML / embedding / vector / classifier — usually cheap.
    SYSTEM  = infra-level routing or tool dispatch — non-billable on its own.
    """

    LLM = "LLM"
    ML = "ML"
    SYSTEM = "System"


class RiskTier(StrEnum):
    """Per-agent guardrail risk class — drives YAML config selection.

    Aligned with `ai-runtime-guardrails` skill — each tier maps to a chain of
    regex / DLP / Model Armor / output-validator screens. CRITICAL is reserved
    for agents that gate platform-wide trust (#10 AI Governance).
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True)
class AgentDescriptor:
    """Immutable metadata for one of the 24 agents.

    Attributes:
        id:                          kebab-case stable identifier; used as the
                                     agent_id on the orchestrate request body.
        comic_number:                Sequel comic numbering (#1-#24).
        display_name:                Human-readable title.
        tier:                        1 (cheapest) -> 4 (highest oversight).
        agent_type:                  LLM | ML | System (see AgentType).
        description:                 One-sentence purpose.
        capabilities:                Capability tags this agent satisfies.
        risk_tier:                   Guardrail risk class.
        cost_estimate_per_call_sgd:  Forecast SGD cost per call (Model Broker).
    """

    id: str
    comic_number: int
    display_name: str
    tier: int
    agent_type: AgentType
    description: str
    capabilities: tuple[Capability, ...]
    risk_tier: RiskTier
    cost_estimate_per_call_sgd: float = 0.0


# ---------------------------------------------------------------------------
# Concrete cast — order matches comic numbering #1 -> #24.
# ---------------------------------------------------------------------------


_AGENTS: tuple[AgentDescriptor, ...] = (
    AgentDescriptor(
        id="orchestrator",
        comic_number=1,
        display_name="Orchestrator",
        tier=2,
        agent_type=AgentType.LLM,
        description=(
            "Conducts the agent crew per request — picks crew, sequences calls, terminates on guardrail refusal."
        ),
        capabilities=(Capability.ATOM_ORCHESTRATION,),
        risk_tier=RiskTier.HIGH,
        cost_estimate_per_call_sgd=0.0040,
    ),
    AgentDescriptor(
        id="tutor",
        comic_number=2,
        display_name="Tutor",
        tier=2,
        agent_type=AgentType.LLM,
        description="Explains atoms in a learner's preferred style — partners with Explainer + Familiar.",
        capabilities=(Capability.TUTORING, Capability.EXPLANATION),
        risk_tier=RiskTier.MEDIUM,
        cost_estimate_per_call_sgd=0.0030,
    ),
    AgentDescriptor(
        id="weakness-analyzer",
        comic_number=3,
        display_name="Weakness Analyzer",
        tier=1,
        agent_type=AgentType.ML,
        description="Diagnoses retention gaps and surfaces topics whose Ebbinghaus decay is critical.",
        capabilities=(Capability.WEAKNESS_ANALYSIS,),
        risk_tier=RiskTier.MEDIUM,
        cost_estimate_per_call_sgd=0.0005,
    ),
    AgentDescriptor(
        id="learning-path",
        comic_number=4,
        display_name="Learning Path",
        tier=1,
        agent_type=AgentType.ML,
        description="Assembles personalised Daily Doses + revision plans from atom mix (decay/curiosity/weakness).",
        capabilities=(Capability.LEARNING_PATH,),
        risk_tier=RiskTier.MEDIUM,
        cost_estimate_per_call_sgd=0.0008,
    ),
    AgentDescriptor(
        id="adaptive-difficulty",
        comic_number=5,
        display_name="Adaptive Difficulty",
        tier=1,
        agent_type=AgentType.ML,
        description="Auto-tunes difficulty during sessions and balances PvP duels.",
        capabilities=(Capability.ADAPTIVE_DIFFICULTY,),
        risk_tier=RiskTier.LOW,
        cost_estimate_per_call_sgd=0.0003,
    ),
    AgentDescriptor(
        id="rag-memory",
        comic_number=6,
        display_name="RAG Memory",
        tier=1,
        agent_type=AgentType.ML,
        description="Long-term learner memory store (per-tenant pgvector) consulted on every session.",
        capabilities=(Capability.RAG_MEMORY,),
        risk_tier=RiskTier.HIGH,
        cost_estimate_per_call_sgd=0.0006,
    ),
    AgentDescriptor(
        id="learning-analytics",
        comic_number=7,
        display_name="Learning Analytics",
        tier=1,
        agent_type=AgentType.ML,
        description="Predicts dropout risk and grade trajectories — flags at-risk learners for nudges.",
        capabilities=(Capability.LEARNING_ANALYTICS,),
        risk_tier=RiskTier.MEDIUM,
        cost_estimate_per_call_sgd=0.0007,
    ),
    AgentDescriptor(
        id="content-analyst",
        comic_number=8,
        display_name="Content Analyst",
        tier=1,
        agent_type=AgentType.ML,
        description="Classifies, deduplicates, and files atoms into the Knowledge Graph.",
        capabilities=(Capability.CONTENT_ANALYSIS,),
        risk_tier=RiskTier.LOW,
        cost_estimate_per_call_sgd=0.0004,
    ),
    AgentDescriptor(
        id="explainer",
        comic_number=9,
        display_name="Explainer",
        tier=2,
        agent_type=AgentType.LLM,
        description="Generates per-learner analogies and visual explanations for stuck concepts.",
        capabilities=(Capability.EXPLANATION,),
        risk_tier=RiskTier.MEDIUM,
        cost_estimate_per_call_sgd=0.0035,
    ),
    AgentDescriptor(
        id="ai-governance",
        comic_number=10,
        display_name="AI Governance",
        tier=4,
        agent_type=AgentType.LLM,
        description="Tier-4 safety gate — PII / bias / unsafe-content scan; stamps SAFE on every produced atom.",
        capabilities=(Capability.AI_GOVERNANCE,),
        risk_tier=RiskTier.CRITICAL,
        cost_estimate_per_call_sgd=0.0090,
    ),
    AgentDescriptor(
        id="company-policy",
        comic_number=11,
        display_name="Company Policy",
        tier=3,
        agent_type=AgentType.LLM,
        description="Per-tenant policy adherence — competitor refs, copyright, brand voice, manipulative language.",
        capabilities=(Capability.COMPANY_POLICY,),
        risk_tier=RiskTier.HIGH,
        cost_estimate_per_call_sgd=0.0055,
    ),
    AgentDescriptor(
        id="content-gen",
        comic_number=12,
        display_name="Content Gen",
        tier=2,
        agent_type=AgentType.LLM,
        description="Generates atoms (MCQ / explainers / care packages) from upstream transcripts and rubrics.",
        capabilities=(Capability.CONTENT_GENERATION,),
        risk_tier=RiskTier.HIGH,
        cost_estimate_per_call_sgd=0.0050,
    ),
    AgentDescriptor(
        id="content-review",
        comic_number=13,
        display_name="Content Review",
        tier=2,
        agent_type=AgentType.LLM,
        description="Pedantic editor — checks Bloom's level, factual claims, language quality before publish.",
        capabilities=(Capability.CONTENT_REVIEW,),
        risk_tier=RiskTier.HIGH,
        cost_estimate_per_call_sgd=0.0040,
    ),
    AgentDescriptor(
        id="training-compliance",
        comic_number=14,
        display_name="Training Compliance",
        tier=3,
        agent_type=AgentType.LLM,
        description="Maps atoms to SSG SkillsFuture frameworks and other regulator taxonomies; auto-corrects labels.",
        capabilities=(Capability.TRAINING_COMPLIANCE,),
        risk_tier=RiskTier.HIGH,
        cost_estimate_per_call_sgd=0.0050,
    ),
    AgentDescriptor(
        id="media-transcription",
        comic_number=15,
        display_name="Media Transcription",
        tier=1,
        agent_type=AgentType.ML,
        description="Speech-to-text for video/audio uploads — multi-speaker diarisation + timestamps.",
        capabilities=(Capability.MEDIA_TRANSCRIPTION,),
        risk_tier=RiskTier.MEDIUM,
        cost_estimate_per_call_sgd=0.0020,
    ),
    AgentDescriptor(
        id="translation",
        comic_number=16,
        display_name="Translation",
        tier=2,
        agent_type=AgentType.LLM,
        description="Translates atoms into MTM target locales (Malay, Mandarin, Tamil, etc.) preserving tone.",
        capabilities=(Capability.TRANSLATION,),
        risk_tier=RiskTier.MEDIUM,
        cost_estimate_per_call_sgd=0.0030,
    ),
    AgentDescriptor(
        id="trigger-optimizer",
        comic_number=17,
        display_name="Trigger Optimizer",
        tier=1,
        agent_type=AgentType.ML,
        description="Picks optimal channel + time-of-day for nudges based on per-learner engagement curves.",
        capabilities=(Capability.TRIGGER_OPTIMIZATION,),
        risk_tier=RiskTier.LOW,
        cost_estimate_per_call_sgd=0.0004,
    ),
    AgentDescriptor(
        id="exam-marking",
        comic_number=18,
        display_name="Exam Marking",
        tier=3,
        agent_type=AgentType.LLM,
        description="Auto-marks free-text essays against rubrics; generates per-student remarks.",
        capabilities=(Capability.EXAM_MARKING,),
        risk_tier=RiskTier.HIGH,
        cost_estimate_per_call_sgd=0.0070,
    ),
    AgentDescriptor(
        id="familiar",
        comic_number=19,
        display_name="Familiar",
        tier=2,
        agent_type=AgentType.LLM,
        description="In-game RPG companion's voice — adjusts tone, remembers preferences within consent scope.",
        capabilities=(Capability.FAMILIAR, Capability.EXPLANATION),
        risk_tier=RiskTier.MEDIUM,
        cost_estimate_per_call_sgd=0.0030,
    ),
    AgentDescriptor(
        id="model-broker",
        comic_number=20,
        display_name="Model Broker",
        tier=1,
        agent_type=AgentType.SYSTEM,
        description="Routes inference between Gemini tiers / Gemma tenants — enforces per-tenant cost ceilings.",
        capabilities=(Capability.MODEL_BROKER,),
        risk_tier=RiskTier.HIGH,
        cost_estimate_per_call_sgd=0.0,
    ),
    AgentDescriptor(
        id="pvp-screener",
        comic_number=21,
        display_name="PvP Screener",
        tier=2,
        agent_type=AgentType.LLM,
        description="Inspects learner-submitted PvP duel questions for trolls, duplicates, and difficulty fit.",
        capabilities=(Capability.PVP_SCREENING,),
        risk_tier=RiskTier.HIGH,
        cost_estimate_per_call_sgd=0.0030,
    ),
    AgentDescriptor(
        id="assessment",
        comic_number=22,
        display_name="Assessment",
        tier=1,
        agent_type=AgentType.ML,
        description="Calibrates per-atom difficulty scores via IRT-style statistical models on historical answers.",
        capabilities=(Capability.ASSESSMENT,),
        risk_tier=RiskTier.MEDIUM,
        cost_estimate_per_call_sgd=0.0006,
    ),
    AgentDescriptor(
        id="mcp-tool-router",
        comic_number=23,
        display_name="MCP Tool Router",
        tier=2,
        agent_type=AgentType.SYSTEM,
        description=(
            "Routes Model Context Protocol tool calls to internal A2A endpoints — sanctioned external sync path."
        ),
        capabilities=(Capability.MCP_TOOL_ROUTING,),
        risk_tier=RiskTier.HIGH,
        cost_estimate_per_call_sgd=0.0010,
    ),
    AgentDescriptor(
        id="story-point-estimator",
        comic_number=24,
        display_name="Story Point Estimator",
        tier=2,
        agent_type=AgentType.LLM,
        description="Suggests Scrum story-point estimates from user-story text — used by Campus Ops auto-marking.",
        capabilities=(Capability.STORY_POINT_ESTIMATION, Capability.EXAM_MARKING),
        risk_tier=RiskTier.MEDIUM,
        cost_estimate_per_call_sgd=0.0028,
    ),
)


@dataclass(frozen=True)
class Agent24Registry:
    """Read-only registry over the 24 agents.

    The instance is constructed once at module import (`AGENT_REGISTRY`) and
    is safe to share across requests — agents are immutable descriptors.
    """

    _agents: tuple[AgentDescriptor, ...] = field(default_factory=lambda: _AGENTS)

    def all(self) -> tuple[AgentDescriptor, ...]:
        """Return all 24 agents in comic-numbering order."""
        return self._agents

    def get(self, agent_id: str) -> AgentDescriptor:
        """Look up an agent by id; KeyError if unknown."""
        for a in self._agents:
            if a.id == agent_id:
                return a
        raise KeyError(f"unknown agent id: {agent_id!r}")

    def filter_by_capability(self, capability: Capability | str) -> tuple[AgentDescriptor, ...]:
        """Return all agents declaring the given capability.

        Accepts either a `Capability` enum or its string value (HTTP query
        convenience). Unknown string values surface a ValueError so callers
        can distinguish "valid but unmapped" from "garbage input".
        """
        cap: Capability = capability if isinstance(capability, Capability) else Capability(capability)
        return tuple(a for a in self._agents if cap in a.capabilities)


AGENT_REGISTRY: Agent24Registry = Agent24Registry()
