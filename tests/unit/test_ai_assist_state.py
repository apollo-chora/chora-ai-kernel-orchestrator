"""Tests for AIAssistState TypedDict + helpers (S3.4 P0 RED).

The 6-agent content gate (Validator -> Classifier -> Web Researcher -> Q&A
Generator -> Evaluator -> Reporter) per Phyllis MVP §5.7 carries a richer
state than the generic OrchestratorState — each gate stage produces its
own typed slot so downstream nodes can branch on prior verdicts.

Per `tdd-blanket` skill RED phase: implementation does not yet exist.
"""

from __future__ import annotations

import pytest


def test_ai_assist_state_typeddict_has_required_keys() -> None:
    """AIAssistState declares the 11 keys mandated by the brief."""
    from chora_ai_kernel_orchestrator.domain.ai_assist_crew.state import (
        AIAssistState,
    )

    annotations = AIAssistState.__annotations__
    expected = {
        # Caller-supplied
        "run_id",
        "tenant_id",
        "gcid",
        "agent_id",
        "prompt",
        "context",
        # Stage outputs
        "candidate_atoms",
        "validator_result",
        "classifier_result",
        "web_research",
        "qa_pairs",
        "evaluation",
        "report",
        "guardrail_verdict",
        # Loop / error book-keeping
        "retry_count",
        "errors",
        # Trace + final state
        "trace",
        "governance_status",
    }
    missing = expected - set(annotations.keys())
    assert not missing, f"AIAssistState missing keys: {missing}"


def test_validator_result_carries_invalid_flag() -> None:
    """ValidatorResult exposes a boolean `invalid` so the conditional REVISE
    edge can branch deterministically.
    """
    from chora_ai_kernel_orchestrator.domain.ai_assist_crew.state import (
        ValidatorResult,
    )

    bad = ValidatorResult(
        invalid=True,
        violations=["no learning_objective"],
        notes="prompt missing required scaffolding",
    )
    assert bad.invalid is True
    assert "no learning_objective" in bad.violations

    good = ValidatorResult(invalid=False, violations=[], notes="clean")
    assert good.invalid is False


def test_evaluation_threshold_branching() -> None:
    """Evaluation.is_below_threshold() drives the RETRY conditional edge."""
    from chora_ai_kernel_orchestrator.domain.ai_assist_crew.state import (
        Evaluation,
    )

    low = Evaluation(score=0.55, rationale="weak distractors", flagged=[])
    high = Evaluation(score=0.92, rationale="strong", flagged=[])

    assert low.is_below_threshold(threshold=0.70) is True
    assert high.is_below_threshold(threshold=0.70) is False
    # Default threshold = 0.70 per skill / brief.
    assert low.is_below_threshold() is True


def test_report_terminal_governance_statuses() -> None:
    """Report.governance_status is restricted to approved | remediated."""
    from chora_ai_kernel_orchestrator.domain.ai_assist_crew.state import (
        Report,
    )

    approved = Report(
        atoms=[{"id": "a1", "stem": "Q1"}],
        governance_status="approved",
        explanation="passes all gates",
        imda_evidence_id="trace-01",
    )
    remediated = Report(
        atoms=[{"id": "a1", "stem": "Q1"}],
        governance_status="remediated",
        explanation="rewritten by guardrail",
        imda_evidence_id="trace-01",
    )
    assert approved.governance_status == "approved"
    assert remediated.governance_status == "remediated"

    with pytest.raises(ValueError):
        Report(
            atoms=[],
            governance_status="rejected",  # not allowed
            explanation="bad",
            imda_evidence_id="x",
        )


def test_new_ai_assist_state_initialises_loop_counters() -> None:
    """new_ai_assist_state() seeds retry_count=0 + empty error/trace lists."""
    from chora_ai_kernel_orchestrator.domain.ai_assist_crew.state import (
        new_ai_assist_state,
    )

    s = new_ai_assist_state(
        run_id="01975c83-0000-7000-8000-000000000000",
        tenant_id="00000000-0000-7000-8000-000000000001",
        gcid="00000000-0000-7000-8000-000000000002",
        agent_id="ai-assist-creation",
        prompt="Generate 10 MCQ on Agile Estimation",
        context={"course_id": "csm-101"},
    )

    assert s["retry_count"] == 0
    assert s["errors"] == []
    assert s["trace"] == []
    assert s["candidate_atoms"] == []
    assert s["validator_result"] is None
    assert s["classifier_result"] is None
    assert s["web_research"] is None
    assert s["qa_pairs"] == []
    assert s["evaluation"] is None
    assert s["report"] is None
    assert s["guardrail_verdict"] is None
    assert s["governance_status"] is None
    assert s["context"]["course_id"] == "csm-101"
