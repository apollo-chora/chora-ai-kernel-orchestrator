"""RED: the companion_diagnosis crew's three dispatched hops on the bus (ADR-254 D5/D6).

  * ``companion_extract``: the extractor reads the upload BY REFERENCE
    (``source_blob_uri`` + ``source_mime_type``); no bytes ride the bus, no
    bytes enter the checkpoint. The kennel keeps Cloud Vision SafeSearch
    before the dispatch (D12).
  * ``companion_diagnose`` with ``task_kind`` in {study_aids, practice_test}:
    the two learner outputs fold onto the diagnoser binary (no new lane);
    ``edges_json`` is the JSON array string of the published edges, each
    ``{concept_key, concept_label, descriptor_json}``; practice_test also
    carries ``max_questions``.

Every key here lands in ADK session state wholesale (agentdispatch merges
input_payload); the names are WP-A's Go constants (``internal/boot/agents.go``),
a wire contract, not a convention.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from chora_ai_kernel_orchestrator.adapter.agent_io.agent_response import (
    AgentExecutorResponse,
)
from chora_ai_kernel_orchestrator.adapter.weakness.pubsub_dispatch import (
    ROLE_COMPANION_DIAGNOSE,
    ROLE_COMPANION_EXTRACT,
    TASK_KIND_DIAGNOSE,
    TASK_KIND_PRACTICE_TEST,
    TASK_KIND_STUDY_AIDS,
    PubSubDiagnoserAdapter,
    PubSubExtractorAdapter,
)

_TENANT = "11111111-1111-7111-8111-111111111111"
_UPLOAD = "01a02062-e5b4-7870-8fca-53ce363cd542"
_GCID = "00000000-0000-7000-8000-000000001999"
_THREAD = f"{_TENANT}:{_UPLOAD}"
_EDGES = [
    {
        "concept_key": "bio.cells.osmosis",
        "concept_label": "Osmosis",
        "descriptor_json": "{}",
        "category": "concept",
        "confidence": 0.9,
        "strength": 0.4,
        "tags": [],
    },
]


class _RecordingExecutor:
    def __init__(self, output: str = "ok") -> None:
        self.calls: list[dict[str, Any]] = []
        self._output = output

    async def execute(self, **kwargs: Any) -> AgentExecutorResponse:
        self.calls.append(kwargs)
        return AgentExecutorResponse(
            execution_id=kwargs["execution_id"],
            output_payload=self._output,
            tokens_consumed_total=0,
            cost_micros_total=0,
            final_state="EXECUTION_FINAL_STATE_SUCCEEDED",
            error_message="",
            input_tokens=0,
            output_tokens=0,
        )


# --------------------------------------------------------------------------- #
# extract
# --------------------------------------------------------------------------- #


def _extractor(executor: Any, thread: str = _THREAD) -> PubSubExtractorAdapter:
    return PubSubExtractorAdapter(executor=executor, thread_id_provider=lambda: thread)


async def _extract(adapter: PubSubExtractorAdapter) -> Any:
    return await adapter.extract(
        source_blob_uri="gs://chora-consumption-growth-uploads-dev/smoke/marked_test_smoke.png",
        source_mime_type="image/png",
        tenant_id=_TENANT,
        gcid=_GCID,
        traceparent="00-trace-span-01",
        tracestate="chora=1",
    )


@pytest.mark.asyncio
async def test_extract_dispatches_by_reference_on_the_companion_extract_role() -> None:
    executor = _RecordingExecutor(output="Q1: 2+2=5 (marked wrong)")
    result = await _extract(_extractor(executor))

    call = executor.calls[0]
    assert call["agent_role"] == ROLE_COMPANION_EXTRACT == "companion_extract"
    payload = json.loads(call["input_payload"])
    assert payload["source_blob_uri"].startswith("gs://")
    assert payload["source_mime_type"] == "image/png"
    assert "blob_bytes" not in payload and "inline_data" not in payload
    # read back out for the ENVELOPE by PubSubAgentExecutor.execute
    assert payload["gcid"] == _GCID and payload["traceparent"] == "00-trace-span-01"
    assert call["workflow_id"] == _UPLOAD
    assert call["execution_id"] == f"{_UPLOAD}:extract"
    assert result.text == "Q1: 2+2=5 (marked wrong)"


@pytest.mark.asyncio
async def test_extract_re_execution_rebuilds_the_same_execution_id() -> None:
    executor = _RecordingExecutor()
    adapter = _extractor(executor)
    await _extract(adapter)
    await _extract(adapter)
    assert executor.calls[0]["execution_id"] == executor.calls[1]["execution_id"]


@pytest.mark.asyncio
async def test_extract_refuses_a_non_growth_edge_thread() -> None:
    with pytest.raises(ValueError, match="thread"):
        await _extract(_extractor(_RecordingExecutor(), thread="nope"))


@pytest.mark.asyncio
async def test_extract_refuses_a_non_gs_uri_before_dispatching() -> None:
    """The extractor injects a FileData part by reference; anything that is
    not a gs:// object is a permanent failure on the agent. Refuse here, with
    nothing parked and nothing metered."""
    executor = _RecordingExecutor()
    with pytest.raises(ValueError, match="gs://"):
        await _extractor(executor).extract(
            source_blob_uri="https://example.com/x.png",
            source_mime_type="image/png",
            tenant_id=_TENANT,
            gcid=_GCID,
            traceparent="",
            tracestate="",
        )
    assert executor.calls == []


# --------------------------------------------------------------------------- #
# task_kind outputs on the diagnoser role
# --------------------------------------------------------------------------- #


def _diagnoser(executor: Any) -> PubSubDiagnoserAdapter:
    return PubSubDiagnoserAdapter(
        executor=executor,
        model_id="gemini-2.5-pro",
        thread_id_provider=lambda: _THREAD,
    )


@pytest.mark.asyncio
async def test_study_aids_rides_the_diagnoser_role_with_edges_json() -> None:
    executor = _RecordingExecutor(output='{"advice":"keep going","glossary":[],"cheat_sheet":[]}')
    text = await _diagnoser(executor).run_task(
        task_kind=TASK_KIND_STUDY_AIDS,
        edges=_EDGES,
        tenant_id=_TENANT,
        gcid=_GCID,
        traceparent="00-t-s-01",
        tracestate="",
    )
    call = executor.calls[0]
    assert call["agent_role"] == ROLE_COMPANION_DIAGNOSE
    payload = json.loads(call["input_payload"])
    assert payload["task_kind"] == "study_aids"
    edges = json.loads(payload["edges_json"])
    # exactly the three keys the contract names, nothing else leaks (no confidence)
    assert edges == [{"concept_key": "bio.cells.osmosis", "concept_label": "Osmosis", "descriptor_json": "{}"}]
    assert "max_questions" not in payload
    assert call["workflow_id"] == _UPLOAD
    assert call["execution_id"].startswith(f"{_UPLOAD}:study_aids:")
    assert text == '{"advice":"keep going","glossary":[],"cheat_sheet":[]}'


@pytest.mark.asyncio
async def test_practice_test_carries_max_questions() -> None:
    executor = _RecordingExecutor(output='{"title":"t","questions":[]}')
    await _diagnoser(executor).run_task(
        task_kind=TASK_KIND_PRACTICE_TEST,
        edges=_EDGES,
        tenant_id=_TENANT,
        gcid=_GCID,
        traceparent="",
        tracestate="",
        max_questions=5,
    )
    payload = json.loads(executor.calls[0]["input_payload"])
    assert payload["task_kind"] == "practice_test"
    assert payload["max_questions"] == 5
    assert executor.calls[0]["execution_id"].startswith(f"{_UPLOAD}:practice_test:")


@pytest.mark.asyncio
async def test_the_two_task_kinds_never_share_an_execution_id() -> None:
    executor = _RecordingExecutor()
    adapter = _diagnoser(executor)
    await adapter.run_task(
        task_kind=TASK_KIND_STUDY_AIDS, edges=_EDGES, tenant_id=_TENANT, gcid=_GCID, traceparent="", tracestate=""
    )
    await adapter.run_task(
        task_kind=TASK_KIND_PRACTICE_TEST, edges=_EDGES, tenant_id=_TENANT, gcid=_GCID, traceparent="", tracestate=""
    )
    assert executor.calls[0]["execution_id"] != executor.calls[1]["execution_id"]


@pytest.mark.asyncio
async def test_an_unknown_task_kind_is_refused_before_dispatch() -> None:
    executor = _RecordingExecutor()
    with pytest.raises(ValueError, match="task_kind"):
        await _diagnoser(executor).run_task(
            task_kind="essay",
            edges=_EDGES,
            tenant_id=_TENANT,
            gcid=_GCID,
            traceparent="",
            tracestate="",
        )
    assert executor.calls == []


@pytest.mark.asyncio
async def test_diagnose_is_the_default_task_kind_and_is_stamped_explicitly() -> None:
    executor = _RecordingExecutor(output='{"edges": []}')
    await _diagnoser(executor).diagnose(
        extracted_text="x",
        structured_clues_block="",
        upload_kind="marked_test",
        tenant_id=_TENANT,
        gcid=_GCID,
        traceparent="",
        tracestate="",
    )
    assert json.loads(executor.calls[0]["input_payload"])["task_kind"] == TASK_KIND_DIAGNOSE == "diagnose"


def test_the_payload_keys_match_the_go_agents_state_keys() -> None:
    """Cross-language: the Go diagnoser reads these exact session-state keys
    (``internal/boot/agents.go``); a drift parks every run on a permanent
    FAILED completion. Read the Go source rather than trust two hand-typed lists."""
    import re
    from pathlib import Path

    # Vendored from the companion_diagnosis ADK agent (a separate repo).
    go = (
        Path(__file__).resolve().parents[2] / "testdata" / "agent_configs" / "companion_diagnosis_agents.go"
    ).read_text()
    keys = dict(re.findall(r'stateKey(\w+)\s*=\s*"([^"]+)"', go))
    assert keys.get("TaskKind") == "task_kind"
    assert keys.get("EdgesJSON") == "edges_json"
    assert keys.get("MaxQuestions") == "max_questions"
    assert keys.get("SourceBlobURI") == "source_blob_uri"
    assert keys.get("SourceMime") == "source_mime_type"
    assert keys.get("ExtractedText") == "extracted_text"
    assert keys.get("CluesBlock") == "clues_block"
