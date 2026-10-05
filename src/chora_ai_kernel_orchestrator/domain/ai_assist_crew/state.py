"""AIAssistState TypedDict + per-stage value objects.

Per Phyllis MVP §5.7 the 6-agent content gate carries a richer state than
the generic OrchestratorState — each stage owns its own typed slot so
subsequent nodes can branch on prior verdicts without re-parsing JSON
payloads.

Pure-domain module: no infra imports (no httpx / grpc / langgraph). All
side effects live in the adapter layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypedDict

# Default evaluator threshold for the RETRY conditional edge.
DEFAULT_EVALUATOR_THRESHOLD: float = 0.70

# Allowed terminal governance_status values per ADR-141 + Phyllis MVP §3 step 4.
_ALLOWED_GOVERNANCE_STATUSES: frozenset[str] = frozenset({"approved", "remediated", "interrupted"})


@dataclass(frozen=True)
class ValidatorResult:
    """Validator stage output — drives the conditional REVISE branch."""

    invalid: bool
    violations: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass(frozen=True)
class Classification:
    """Classifier stage output — topic, difficulty, locale."""

    topic: str
    difficulty: int
    locale: str


@dataclass(frozen=True)
class WebResearch:
    """Web Researcher stage output — facts + sources for grounding."""

    facts: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class QAPair:
    """Q&A Generator output — one MCQ / short answer."""

    id: str
    stem: str
    answer: str
    distractors: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Evaluation:
    """Evaluator stage output — drives the conditional RETRY branch."""

    score: float
    rationale: str = ""
    flagged: list[str] = field(default_factory=list)

    def is_below_threshold(self, threshold: float = DEFAULT_EVALUATOR_THRESHOLD) -> bool:
        return self.score < threshold


@dataclass(frozen=True)
class Report:
    """Reporter stage output — terminal artefact for the AI Assist run."""

    atoms: list[dict[str, Any]] = field(default_factory=list)
    governance_status: str = "approved"
    explanation: str = ""
    imda_evidence_id: str = ""
    requires_hitl: bool = False

    def __post_init__(self) -> None:
        if self.governance_status not in _ALLOWED_GOVERNANCE_STATUSES:
            raise ValueError(
                f"governance_status must be one of {sorted(_ALLOWED_GOVERNANCE_STATUSES)}, "
                f"got {self.governance_status!r}"
            )


class AIAssistState(TypedDict, total=False):
    """LangGraph state for the AI Assist crew.

    `total=False` so partial-dict node returns merge cleanly.
    """

    # Caller-supplied
    run_id: str
    tenant_id: str
    gcid: str
    agent_id: str
    prompt: str
    context: dict[str, str]

    # Stage outputs (Optional until that stage runs)
    candidate_atoms: list[dict[str, Any]]
    validator_result: ValidatorResult | None
    classifier_result: Classification | None
    web_research: WebResearch | None
    qa_pairs: list[QAPair]
    evaluation: Evaluation | None
    report: Report | None
    guardrail_verdict: str | None  # 'approved' | 'remediated' | 'blocked'

    # Loop / book-keeping
    retry_count: int
    errors: list[str]

    # Per-node trace (audit + admin trace endpoint)
    trace: list[dict[str, Any]]

    # Final terminal state — set by the Reporter or HITL resume.
    governance_status: str | None


def new_ai_assist_state(
    *,
    run_id: str,
    tenant_id: str,
    gcid: str,
    agent_id: str,
    prompt: str,
    context: dict[str, str] | None = None,
) -> AIAssistState:
    """Build a fresh AIAssistState — all loop counters zeroed, lists empty."""
    return AIAssistState(
        run_id=run_id,
        tenant_id=tenant_id,
        gcid=gcid,
        agent_id=agent_id,
        prompt=prompt,
        context=dict(context or {}),
        candidate_atoms=[],
        validator_result=None,
        classifier_result=None,
        web_research=None,
        qa_pairs=[],
        evaluation=None,
        report=None,
        guardrail_verdict=None,
        retry_count=0,
        errors=[],
        trace=[],
        governance_status=None,
    )
