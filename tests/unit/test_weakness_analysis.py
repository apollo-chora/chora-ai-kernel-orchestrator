"""Unit tests for the weakness-analyser pure analysis core (W2 of the 1b
Growth-Edge track).

Covers: per-upload_kind system prompts, the multimodal contents_json builder
(inlineData for binary, text-part for text docs), and the structured-output
parser (fence stripping, object/array tolerance, confidence-threshold drop of
noisy edges, concept_key derivation, clamping, cap, fail-soft on garbage). No
gRPC / Pub/Sub / GCS here — this is pure domain logic.
"""

from __future__ import annotations

import base64
import json

import pytest

from chora_ai_kernel_orchestrator.domain.weakness_analyser_crew.analysis import (
    MAX_EDGES,
    MIN_CONFIDENCE,
    UPLOAD_KIND_MARKED_TEST,
    UPLOAD_KIND_NOTES,
    UPLOAD_KIND_SCRIBBLE,
    AnalysisResult,
    CandidateStruggle,
    ExtractedGrowthEdge,
    build_contents_json,
    build_system_prompt,
    normalize_concept_key,
    parse_analysis,
)

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# system prompt
# --------------------------------------------------------------------------- #


def test_build_system_prompt_varies_by_kind() -> None:
    marked = build_system_prompt(UPLOAD_KIND_MARKED_TEST)
    notes = build_system_prompt(UPLOAD_KIND_NOTES)
    scribble = build_system_prompt(UPLOAD_KIND_SCRIBBLE)

    assert "per-item" in marked.lower() or "right" in marked.lower()
    assert "notes" in notes.lower()
    assert "scribble" in scribble.lower() or "drawn" in scribble.lower()
    # All must instruct strict JSON output with the contract keys.
    for p in (marked, notes, scribble):
        assert "json" in p.lower()
        assert "concept_label" in p
        assert "confidence" in p


def test_build_system_prompt_unknown_kind_falls_back() -> None:
    # An unexpected kind must still yield a usable (marked-test default) prompt.
    p = build_system_prompt("bogus")
    assert "concept_label" in p


# --------------------------------------------------------------------------- #
# contents_json (multimodal)
# --------------------------------------------------------------------------- #


def test_build_contents_json_binary_uses_inline_data() -> None:
    blob = b"%PDF-1.7 fake pdf bytes"
    raw = build_contents_json(
        upload_kind=UPLOAD_KIND_MARKED_TEST,
        blob_bytes=blob,
        mime_type="application/pdf",
        context_hint="Secondary 3 geography",
    )
    doc = json.loads(raw)
    assert isinstance(doc, list) and len(doc) == 1
    parts = doc[0]["parts"]
    assert doc[0]["role"] == "user"
    # first part is the text instruction (incl. the context hint)
    assert "Secondary 3 geography" in parts[0]["text"]
    # second part is the inline binary
    inline = parts[1]["inlineData"]
    assert inline["mimeType"] == "application/pdf"
    assert base64.b64decode(inline["data"]) == blob


def test_build_contents_json_text_doc_uses_text_part() -> None:
    blob = b"I always mix up fluvial and pluvial flooding."
    raw = build_contents_json(
        upload_kind=UPLOAD_KIND_NOTES,
        blob_bytes=blob,
        mime_type="text/plain",
        context_hint="",
    )
    doc = json.loads(raw)
    parts = doc[0]["parts"]
    # text docs are inlined as a text part — no inlineData
    assert all("inlineData" not in p for p in parts)
    joined = " ".join(p.get("text", "") for p in parts)
    assert "fluvial and pluvial" in joined


# --------------------------------------------------------------------------- #
# parse_analysis
# --------------------------------------------------------------------------- #


def _edge(label: str, conf: float, strength: float | None = None) -> dict:
    e = {
        "concept_label": label,
        "confidence": conf,
        "category": "physical-geography",
        "tags": ["flooding"],
        "summary": f"gap in {label}",
    }
    if strength is not None:
        e["strength"] = strength
    return e


def test_parse_filters_low_confidence_and_builds_edges() -> None:
    payload = {
        "edges": [
            _edge("causes of riverine flooding", 0.9, 0.8),
            _edge("noise concept", 0.2, 0.9),  # below MIN_CONFIDENCE -> dropped
        ]
    }
    res = parse_analysis(json.dumps(payload), model_used="gemini-3-pro", input_tokens=10, output_tokens=20)
    assert isinstance(res, AnalysisResult)
    assert res.model_used == "gemini-3-pro"
    assert res.input_tokens == 10 and res.output_tokens == 20
    assert len(res.edges) == 1
    edge = res.edges[0]
    assert isinstance(edge, ExtractedGrowthEdge)
    assert edge.concept_label == "causes of riverine flooding"
    assert edge.concept_key == "causes-of-riverine-flooding"  # derived
    assert edge.category == "physical-geography"
    assert edge.tags == ["flooding"]
    assert edge.strength == 0.8
    # descriptor_json carries the distilled metadata as a JSON string
    desc = json.loads(edge.descriptor_json)
    assert desc.get("summary") == "gap in causes of riverine flooding"


def test_parse_strips_code_fence() -> None:
    fenced = "```json\n" + json.dumps({"edges": [_edge("photosynthesis", 0.7, 0.6)]}) + "\n```"
    res = parse_analysis(fenced, model_used="m", input_tokens=0, output_tokens=0)
    assert len(res.edges) == 1
    assert res.edges[0].concept_key == "photosynthesis"


def test_parse_accepts_bare_array() -> None:
    bare = json.dumps([_edge("mitosis vs meiosis", 0.8, 0.7)])
    res = parse_analysis(bare, model_used="m", input_tokens=0, output_tokens=0)
    assert len(res.edges) == 1
    assert res.edges[0].concept_key == "mitosis-vs-meiosis"


def test_parse_failsoft_on_garbage() -> None:
    # Unreadable / non-JSON -> empty edges (valid: "nothing weak detected").
    res = parse_analysis("the model said something non-JSON", model_used="m", input_tokens=1, output_tokens=2)
    assert res.edges == []
    assert res.model_used == "m"


def test_parse_skips_edges_without_label() -> None:
    payload = {"edges": [{"confidence": 0.9, "strength": 0.5}, _edge("ok concept", 0.9, 0.5)]}
    res = parse_analysis(json.dumps(payload), model_used="m", input_tokens=0, output_tokens=0)
    assert len(res.edges) == 1
    assert res.edges[0].concept_label == "ok concept"


def test_parse_clamps_and_defaults_strength() -> None:
    payload = {"edges": [{"concept_label": "x", "confidence": 1.5}]}  # no strength, conf >1
    res = parse_analysis(json.dumps(payload), model_used="m", input_tokens=0, output_tokens=0)
    assert len(res.edges) == 1
    e = res.edges[0]
    # confidence clamped to 1.0; strength defaults to confidence when absent
    assert 0.0 <= e.strength <= 1.0
    assert e.strength == 1.0


def test_parse_caps_edge_count() -> None:
    payload = {"edges": [_edge(f"concept number {i}", 0.9, 0.9) for i in range(MAX_EDGES + 10)]}
    res = parse_analysis(json.dumps(payload), model_used="m", input_tokens=0, output_tokens=0)
    assert len(res.edges) == MAX_EDGES


def test_parse_honours_explicit_concept_key() -> None:
    payload = {
        "edges": [
            {
                "concept_label": "Riverine flood causes",
                "concept_key": "riverine-flood-causes",
                "confidence": 0.9,
                "strength": 0.7,
            }
        ]
    }
    res = parse_analysis(json.dumps(payload), model_used="m", input_tokens=0, output_tokens=0)
    assert res.edges[0].concept_key == "riverine-flood-causes"


def test_parse_drops_edge_with_empty_normalised_key() -> None:
    payload = {"edges": [{"concept_label": "!!!", "confidence": 0.9}, _edge("real one", 0.9, 0.5)]}
    res = parse_analysis(json.dumps(payload), model_used="m", input_tokens=0, output_tokens=0)
    assert [e.concept_label for e in res.edges] == ["real one"]


def test_min_confidence_constant_is_sane() -> None:
    assert 0.0 < MIN_CONFIDENCE < 1.0


def test_parse_descriptor_carries_samples_and_per_item() -> None:
    payload = {
        "edges": [
            {
                "concept_label": "fluvial flooding",
                "confidence": 0.9,
                "strength": 0.7,
                "misconceptions": ["thinks all flooding is rainfall"],
                "suggested_angles": ["compare fluvial vs pluvial"],
                "sample_wrong": [{"prompt": "Q3", "why_wrong": "chose pluvial"}, "garbage-not-dict"],
                "per_item_correctness": [{"item": "Q3", "correct": False}, 42],
            }
        ]
    }
    res = parse_analysis(json.dumps(payload), model_used="m", input_tokens=0, output_tokens=0)
    desc = json.loads(res.edges[0].descriptor_json)
    assert desc["misconceptions"] == ["thinks all flooding is rainfall"]
    assert desc["suggested_angles"] == ["compare fluvial vs pluvial"]
    # non-dict members are filtered out of sample_wrong / per_item_correctness
    assert desc["sample_wrong"] == [{"prompt": "Q3", "why_wrong": "chose pluvial"}]
    assert desc["per_item_correctness"] == [{"item": "Q3", "correct": False}]


def test_parse_non_collection_is_empty() -> None:
    # A JSON scalar (not object/array) yields zero edges, not a crash.
    res = parse_analysis("42", model_used="m", input_tokens=0, output_tokens=0)
    assert res.edges == []


def test_parse_strips_plain_and_unclosed_fence() -> None:
    plain = "```\n" + json.dumps([_edge("erosion", 0.8, 0.6)]) + "\n```"
    assert len(parse_analysis(plain, model_used="m", input_tokens=0, output_tokens=0).edges) == 1
    unclosed = "```json\n" + json.dumps([_edge("deposition", 0.8, 0.6)])
    assert len(parse_analysis(unclosed, model_used="m", input_tokens=0, output_tokens=0).edges) == 1


# --------------------------------------------------------------------------- #
# below-threshold retention (CHO-1973 Q2 — candidate_struggles source)
# --------------------------------------------------------------------------- #
#
# The bounded HITL review panel offers the learner a short list of concepts the
# analyser DIAGNOSED but did NOT promote to a Growth Edge (confidence below
# MIN_CONFIDENCE). parse_analysis must RETAIN those sub-threshold concepts in
# ``below_threshold`` WITHOUT changing which concepts are promoted to ``edges``.


def test_parse_retains_below_threshold_concepts() -> None:
    payload = {
        "edges": [
            _edge("causes of riverine flooding", 0.9, 0.8),  # promoted -> edge
            _edge("coastal erosion", 0.3, 0.6),  # below -> retained struggle
            _edge("tectonic plates", 0.45, 0.4),  # below -> retained struggle
        ]
    }
    res = parse_analysis(json.dumps(payload), model_used="m", input_tokens=0, output_tokens=0)
    # promotion is UNCHANGED — only the above-threshold concept becomes an edge.
    assert [e.concept_label for e in res.edges] == ["causes of riverine flooding"]
    # the two sub-threshold concepts are RETAINED (not discarded as before).
    assert all(isinstance(s, CandidateStruggle) for s in res.below_threshold)
    by_key = {s.concept_key: s for s in res.below_threshold}
    assert set(by_key) == {"coastal-erosion", "tectonic-plates"}
    # each carries its label + confidence so the panel can rank + cap the picker.
    assert by_key["coastal-erosion"].concept_label == "coastal erosion"
    assert by_key["coastal-erosion"].confidence == 0.3


def test_below_threshold_retention_does_not_change_edges() -> None:
    # GUARD: retaining sub-threshold concepts must NOT alter the promoted edges.
    payload = {
        "edges": [
            _edge("promoted one", 0.9, 0.8),
            _edge("noise", 0.2, 0.9),
            _edge("promoted two", 0.7, 0.6),
        ]
    }
    res = parse_analysis(json.dumps(payload), model_used="m", input_tokens=0, output_tokens=0)
    assert [e.concept_key for e in res.edges] == ["promoted-one", "promoted-two"]
    assert {s.concept_key for s in res.below_threshold} == {"noise"}


def test_below_threshold_excludes_zero_missing_and_at_threshold() -> None:
    payload = {
        "edges": [
            _edge("zero conf", 0.0, 0.5),  # noise -> not retained
            {"concept_label": "missing conf", "strength": 0.5},  # default 0.0 -> not retained
            _edge("at threshold", MIN_CONFIDENCE, 0.5),  # >= floor -> promoted edge
            _edge("just below", MIN_CONFIDENCE - 0.01, 0.5),  # retained struggle
        ]
    }
    res = parse_analysis(json.dumps(payload), model_used="m", input_tokens=0, output_tokens=0)
    assert {s.concept_key for s in res.below_threshold} == {"just-below"}
    # the boundary concept promotes to an edge (the floor is unchanged) — it must
    # never be double-counted as a struggle.
    assert "at-threshold" in {e.concept_key for e in res.edges}


def test_below_threshold_skips_concept_without_label() -> None:
    payload = {
        "edges": [
            {"confidence": 0.3, "strength": 0.5},  # no label -> skipped, never fabricated
            _edge("real low concept", 0.3, 0.5),
        ]
    }
    res = parse_analysis(json.dumps(payload), model_used="m", input_tokens=0, output_tokens=0)
    assert [s.concept_label for s in res.below_threshold] == ["real low concept"]


def test_below_threshold_derives_key_from_label() -> None:
    payload = {"edges": [{"concept_label": "Long Division!", "confidence": 0.3}]}
    res = parse_analysis(json.dumps(payload), model_used="m", input_tokens=0, output_tokens=0)
    assert [s.concept_key for s in res.below_threshold] == ["long-division"]


def test_below_threshold_skips_unnormalisable_key() -> None:
    payload = {"edges": [{"concept_label": "!!!", "confidence": 0.3}]}
    res = parse_analysis(json.dumps(payload), model_used="m", input_tokens=0, output_tokens=0)
    assert res.below_threshold == []


def test_below_threshold_empty_when_all_promoted() -> None:
    payload = {"edges": [_edge("a", 0.9, 0.8), _edge("b", 0.8, 0.7)]}
    res = parse_analysis(json.dumps(payload), model_used="m", input_tokens=0, output_tokens=0)
    assert res.below_threshold == []
    assert len(res.edges) == 2


# --------------------------------------------------------------------------- #
# normalize_concept_key
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("raw", "want"),
    [
        ("Causes of Riverine Flooding", "causes-of-riverine-flooding"),
        ("  multiple   spaces  ", "multiple-spaces"),
        ("already-a-slug", "already-a-slug"),
        ("Photosynthesis (C3 vs C4)!", "photosynthesis-c3-vs-c4"),
        ("!!!", ""),
        ("UPPER_snake_Case", "upper-snake-case"),
    ],
)
def test_normalize_concept_key(raw: str, want: str) -> None:
    assert normalize_concept_key(raw) == want
