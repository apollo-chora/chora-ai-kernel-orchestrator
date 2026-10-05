"""Capability tags for the 24-agent registry.

Each capability is a kebab-case string the registry uses for filtering. They
correspond 1:1 with the value streams exercised by the Sequel comic's 16 acts:
atom orchestration, content generation, translation, exam marking, and so on.
The enum is the single canonical taxonomy — agents declare which capabilities
they own, and HTTP callers filter by the same string values.

Source: docs/comic/Sequel_Chora-Agentic-Customer-journeys.md ("Behind the
Button" cast). One capability tag may be owned by multiple agents (e.g. both
Familiar and Tutor handle tutoring-style explanations).
"""

from __future__ import annotations

from enum import StrEnum


class Capability(StrEnum):
    """Filterable capability tags for the 24-agent registry."""

    ATOM_ORCHESTRATION = "atom-orchestration"
    TUTORING = "tutoring"
    WEAKNESS_ANALYSIS = "weakness-analysis"
    LEARNING_PATH = "learning-path"
    ADAPTIVE_DIFFICULTY = "adaptive-difficulty"
    RAG_MEMORY = "rag-memory"
    LEARNING_ANALYTICS = "learning-analytics"
    CONTENT_ANALYSIS = "content-analysis"
    EXPLANATION = "explanation"
    AI_GOVERNANCE = "ai-governance"
    COMPANY_POLICY = "company-policy"
    CONTENT_GENERATION = "content-generation"
    CONTENT_REVIEW = "content-review"
    TRAINING_COMPLIANCE = "training-compliance"
    MEDIA_TRANSCRIPTION = "media-transcription"
    TRANSLATION = "translation"
    TRIGGER_OPTIMIZATION = "trigger-optimization"
    EXAM_MARKING = "exam-marking"
    FAMILIAR = "familiar"
    MODEL_BROKER = "model-broker"
    PVP_SCREENING = "pvp-screening"
    ASSESSMENT = "assessment"
    MCP_TOOL_ROUTING = "mcp-tool-routing"
    STORY_POINT_ESTIMATION = "story-point-estimation"
