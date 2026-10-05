"""RED: the growth-edge diagnoser on the ADR-253 Pub/Sub dispatch lane.

The diagnoser is NOT a fourth caller of ``_ExecutorLike.execute``. It is a domain
PORT (``orchestrators/weakness_analyser_crew.Diagnoser``) whose HTTP
implementation speaks the ADK REST dialect directly, ``_create_session`` then
``_stream_query`` against a resolved ``gke://`` host. So this conversion builds a
seam rather than reusing one, and the adapter below is where the two shapes meet:
it satisfies the crew's ``Diagnoser`` port on the outside and parks through the
Pub/Sub executor on the inside.

Keeping the PORT is what makes the ADR-253 D6 rollback real here: selecting
``AdkDiagnoserAdapter`` again is a composition-root change, not a crew rewrite.
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
    PubSubDiagnoserAdapter,
)

_TENANT = "11111111-1111-7111-8111-111111111111"
_UPLOAD = "01a02062-e5b4-7870-8fca-53ce363cd542"
_GCID = "00000000-0000-7000-8000-000000001999"
_THREAD = f"{_TENANT}:{_UPLOAD}"


class _RecordingExecutor:
    """Records the dispatch and returns what the agent would have answered."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def execute(self, **kwargs: Any) -> AgentExecutorResponse:
        self.calls.append(kwargs)
        return AgentExecutorResponse(
            execution_id=kwargs["execution_id"],
            output_payload='{"edges": []}',
            tokens_consumed_total=300,
            cost_micros_total=0,
            final_state="EXECUTION_FINAL_STATE_SUCCEEDED",
            error_message="",
            input_tokens=200,
            output_tokens=100,
        )


def _adapter(executor: Any, thread: str = _THREAD) -> PubSubDiagnoserAdapter:
    return PubSubDiagnoserAdapter(
        executor=executor,
        model_id="gemini-2.5-pro",
        thread_id_provider=lambda: thread,
    )


async def _diagnose(adapter: PubSubDiagnoserAdapter, clues: str = "clue block") -> Any:
    return await adapter.diagnose(
        extracted_text="the learner confused mitosis with meiosis",
        structured_clues_block=clues,
        upload_kind="marked_test",
        tenant_id=_TENANT,
        gcid=_GCID,
        traceparent="00-trace-span-01",
        tracestate="chora=1",
    )


@pytest.mark.asyncio
async def test_the_payload_carries_every_key_the_agent_reads_from_session_state() -> None:
    """A missing key renders an EMPTY [TASK] block and the agent fails loud.

    ``weakness_adk_go/cmd/weakness_diagnoser/main.go`` composes its per-turn
    instruction from ``extracted_text`` (REQUIRED, it fatals without it) and the
    optional ``clues_block``. The Go side merges input_payload into ADK session
    state wholesale (``agentdispatch/dispatch.go:112`` SessionState), so these
    names are the wire contract and must match exactly.
    """
    executor = _RecordingExecutor()
    await _diagnose(_adapter(executor))

    payload = json.loads(executor.calls[0]["input_payload"])
    assert payload["extracted_text"].startswith("the learner confused")
    assert payload["clues_block"] == "clue block"
    assert payload["upload_kind"] == "marked_test"
    # ADR-254 D6: the task_kind discriminator, explicit (absent would also mean
    # diagnose on the wire, but an explicit value is what the tests can pin).
    assert payload["task_kind"] == "diagnose"
    # read back out for the ENVELOPE by PubSubAgentExecutor.execute
    assert payload["gcid"] == _GCID
    assert payload["traceparent"] == "00-trace-span-01"


@pytest.mark.asyncio
async def test_the_outbox_keys_on_the_upload_not_the_thread() -> None:
    """``workflow_id`` is UUID NOT NULL; the growth-edge thread key is not a UUID."""
    executor = _RecordingExecutor()
    await _diagnose(_adapter(executor))

    assert executor.calls[0]["workflow_id"] == _UPLOAD


@pytest.mark.asyncio
async def test_a_re_executed_node_rebuilds_an_identical_execution_id() -> None:
    """LangGraph re-runs a node from the top on resume, that is the contract.

    The idempotency key is derived from ``execution_id``, so a rebuilt request
    must collide with the queued row rather than dispatching the same diagnosis
    twice.
    """
    executor = _RecordingExecutor()
    adapter = _adapter(executor)
    await _diagnose(adapter)
    await _diagnose(adapter)

    assert executor.calls[0]["execution_id"] == executor.calls[1]["execution_id"]


@pytest.mark.asyncio
async def test_a_reiterate_gets_a_distinct_execution_id() -> None:
    """⚠ The crew LOOPS hitl_review -> diagnose on the same thread and upload.

    ``build_weakness_thread_id``'s docstring records that a "reiterate" stays
    in-thread. So an execution_id of ``{upload}:diagnose`` alone would give the
    second diagnosis the SAME idempotency key as the first, and the re-run would
    be silently deduped away, the learner asks for another look and receives the
    original answer. The reiterate folds its focus topics into the clue block
    before the port call, so the clue block is what discriminates the attempts.
    """
    executor = _RecordingExecutor()
    adapter = _adapter(executor)
    await _diagnose(adapter, clues="clue block")
    await _diagnose(adapter, clues="clue block + focus: meiosis")

    assert executor.calls[0]["execution_id"] != executor.calls[1]["execution_id"]


@pytest.mark.asyncio
async def test_the_agents_answer_maps_onto_the_port_result() -> None:
    executor = _RecordingExecutor()
    result = await _diagnose(_adapter(executor))

    assert result.diagnosis_json == '{"edges": []}'
    assert result.model_used == "gemini-2.5-pro"
    assert result.input_tokens == 200
    assert result.output_tokens == 100


@pytest.mark.asyncio
async def test_a_thread_that_is_not_the_growth_edge_shape_is_refused() -> None:
    """Fail loud rather than dispatch on a guessed upload id."""
    executor = _RecordingExecutor()
    with pytest.raises(ValueError, match="thread"):
        await _diagnose(_adapter(executor, thread="not-a-growth-edge-thread"))


# -----------------------------------------------------------------------------
# ⚠ The fakes above accept **kwargs, so they cannot see a signature mismatch
# against the REAL executor. Green on both sides means suspect the wire: bind the
# adapter's actual call to PubSubAgentExecutor.execute and let inspect refuse it.
# -----------------------------------------------------------------------------


def test_the_adapter_matches_the_real_executor_signature() -> None:
    """Every kwarg the adapter sends must be accepted by PubSubAgentExecutor.

    Without this, the adapter and its unit tests stay green while production
    raises TypeError on the first dispatch, the executor never grew the
    ``workflow_id`` parameter the growth-edge lane has to pass.
    """
    import inspect

    from chora_ai_kernel_orchestrator.adapter.pubsub.pubsub_agent_executor import (
        PubSubAgentExecutor,
    )

    sent = {
        "execution_id": f"{_UPLOAD}:diagnose:abc123",
        "tenant_id": _TENANT,
        "agid": "",
        "agent_role": "companion_diagnose",
        "input_payload": "{}",
        "workflow_id": _UPLOAD,
    }
    # raises TypeError naming the offending kwarg if the signature disagrees
    inspect.signature(PubSubAgentExecutor.execute).bind(PubSubAgentExecutor(source_project="chora-489812"), **sent)


def test_the_go_agent_subscribes_to_the_role_python_publishes() -> None:
    """Cross-language contract. A mismatch here is invisible on both sides.

    The Python producer keys the request topic on ``agent_role``; the Go
    subscriber derives its subscription from its own ``dispatchRole`` constant.
    If they disagree, the orchestrator publishes to a topic nobody consumes and
    the agent subscribes to a topic nobody publishes to. The pod reports healthy,
    the dispatch parks forever, and neither side logs an error.

    There is no shared artefact to generate this from, so the test reads the Go
    source. That is deliberate: the alternative is trusting two hand-typed
    strings in different languages to stay equal.
    """
    import re
    from pathlib import Path

    # Vendored from the companion_diagnosis ADK agent (a separate repo); the
    # standalone repo parity-pins against the vendored copy.
    go_dispatch = Path(__file__).resolve().parents[2] / "testdata" / "agent_configs" / "companion_diagnosis_dispatch.go"
    assert go_dispatch.is_file(), f"agent dispatch.go not found at {go_dispatch}"
    text = go_dispatch.read_text()

    diagnose = re.search(r'DispatchRoleDiagnose\s*=\s*"([^"]+)"', text)
    extract = re.search(r'DispatchRoleExtract\s*=\s*"([^"]+)"', text)
    assert diagnose and extract, "DispatchRoleDiagnose / DispatchRoleExtract not found in the Go agent"
    assert diagnose.group(1) == ROLE_COMPANION_DIAGNOSE, (
        f"Go DispatchRoleDiagnose={diagnose.group(1)!r} != Python ROLE_COMPANION_DIAGNOSE={ROLE_COMPANION_DIAGNOSE!r}"
    )
    from chora_ai_kernel_orchestrator.adapter.weakness.pubsub_dispatch import (
        ROLE_COMPANION_EXTRACT,
    )

    assert extract.group(1) == ROLE_COMPANION_EXTRACT, (
        f"Go DispatchRoleExtract={extract.group(1)!r} != Python ROLE_COMPANION_EXTRACT={ROLE_COMPANION_EXTRACT!r}"
    )


def test_the_role_constants_are_the_adr_254_names_and_the_legacy_one_is_gone() -> None:
    """``companion_diagnose`` is the lane the new kennel dispatches on. The
    legacy ``weakness_diagnose`` constant was deleted with its registration
    (coordinator "drop it", 2026-08-23): its request subscription read 0
    undelivered, so the drain window is closed and the lane is destroyed. A
    surviving constant would invite a re-registration nothing publishes to."""
    import chora_ai_kernel_orchestrator.adapter.weakness.pubsub_dispatch as mod

    assert ROLE_COMPANION_DIAGNOSE == "companion_diagnose"
    assert not hasattr(mod, "LEGACY_ROLE_WEAKNESS_DIAGNOSE")


# -----------------------------------------------------------------------------
# The DEFAULT thread provider, the one production actually uses. The tests above
# inject a provider, so none of them exercise it, and a real park reads the
# thread from LangGraph's runtime config.
# -----------------------------------------------------------------------------


def test_the_default_provider_refuses_a_config_with_no_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dispatch parked without a thread_id could never be resumed.

    This branch IS reachable in production: a graph invoked without a thread_id
    in ``configurable`` reaches the node and then the dispatch. Monkeypatching
    ``get_config`` is what exercises it, called for real outside a runnable
    context, LangGraph raises RuntimeError before this check is ever reached,
    which is fail-loud too but is a different (and unreachable) path.
    """
    import langgraph.config

    from chora_ai_kernel_orchestrator.adapter.weakness import pubsub_dispatch

    monkeypatch.setattr(langgraph.config, "get_config", lambda: {"configurable": {}})
    with pytest.raises(ValueError, match="thread_id"):
        pubsub_dispatch._thread_id_from_langgraph()


def test_upload_id_is_recovered_from_the_growth_edge_thread() -> None:
    from chora_ai_kernel_orchestrator.adapter.weakness.pubsub_dispatch import (
        upload_id_from_thread,
    )

    assert upload_id_from_thread(_THREAD) == _UPLOAD


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "only-one-segment",
        f"{_TENANT}:{_UPLOAD}:extra",  # the 3-segment qgen/OE shape
        f"{_TENANT}:",
        f":{_UPLOAD}",
    ],
)
def test_a_thread_of_the_wrong_shape_is_refused_rather_than_guessed(bad: str) -> None:
    """Including the 3-segment shape, which is a DIFFERENT crew's thread key.

    ``build_thread_id`` returns ``{tenant}:{workflow}:{run}`` for qgen / OE /
    ai-assist. Silently taking segment 1 of that would write the outbox row
    against a workflow id, not an upload id, and the row would land under the
    wrong aggregate.
    """
    from chora_ai_kernel_orchestrator.adapter.weakness.pubsub_dispatch import (
        upload_id_from_thread,
    )

    with pytest.raises(ValueError, match="growth-edge shape"):
        upload_id_from_thread(bad)
