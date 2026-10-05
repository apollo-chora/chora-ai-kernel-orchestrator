"""The companion_diagnosis crew's dispatched hops on the ADR-253 Pub/Sub lane.

ADR-254 D2/D5/D6 (2026-08-22). Three hops, two roles, one seam:

  * ``companion_extract``  ``PubSubExtractorAdapter`` fills the crew's
    ``Extractor`` port. The extractor reads the upload BY REFERENCE
    (``source_blob_uri`` + ``source_mime_type``; the agent injects a Gemini
    ``FileData`` part through the shared grounding plugin). No bytes ride the
    bus and no bytes enter the checkpoint; Cloud Vision SafeSearch stays in the
    kennel BEFORE this dispatch (D12).
  * ``companion_diagnose`` with ``task_kind=diagnose``  ``PubSubDiagnoserAdapter
    .diagnose`` fills the crew's ``Diagnoser`` port (the ex ``weakness_diagnose``
    hop, role renamed at the cut).
  * ``companion_diagnose`` with ``task_kind`` in {study_aids, practice_test}
    ``PubSubDiagnoserAdapter.run_task`` fills the crew's ``OutputTaskRunner``
    port: the two learner outputs fold onto the diagnoser binary, no new lane.

Why this is a seam rather than a fourth caller of ``_ExecutorLike.execute``:
the diagnoser and the extractor are domain PORTS the crew graph speaks; keeping
the ports and implementing them over the bus is what keeps the graph free of
transport (hexagonal), and the AST park guard enumerates these verbs.

Three things the growth-edge lane has to get right that the OE lane got free:

1. ``workflow_id`` is NOT the thread key. ``ai_kernel_outbox_events.workflow_id``
   is ``UUID NOT NULL`` (``migrations/0003_outbox.sql:39``); the growth-edge
   thread is ``{tenant_id}:{upload_id}`` (``checkpointer/factory.py:105``), so
   the outbox row keys on ``upload_id``.
2. The upload id is read from the LANGGRAPH THREAD, not passed in: taken from
   the runtime, the value is by construction the one the run is parked under.
3. The crew LOOPS ``hitl_review -> diagnose`` on a reiterate, so the diagnose
   execution id carries a digest of the request content (the reiterate folds
   its focus topics into the clue block BEFORE the port call). The output
   tasks carry a digest of the published edges for the same reason, and the
   extract runs once per upload so its id is plain ``{upload}:extract``.

Every key in an ``input_payload`` lands in ADK session state wholesale
(``agentdispatch/dispatch.go`` SessionState); the names are the Go agent's
state keys (``companion_diagnosis_adk_go/internal/boot/agents.go``), a wire
contract pinned by ``tests/unit/test_weakness_pubsub_tasks_and_extract.py``.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Protocol

from chora_ai_kernel_orchestrator.orchestrators.weakness_analyser_crew import (
    DiagnoseResult,
    ExtractResult,
)

logger = logging.getLogger(__name__)

#: The dispatch roles (ADR-254 D2). MUST match ``DispatchRoleDiagnose`` /
#: ``DispatchRoleExtract`` on the Go side (``internal/boot/dispatch.go``): a
#: mismatch subscribes to a topic nobody publishes to and every dispatch parks
#: forever while the pod reports healthy.
ROLE_COMPANION_DIAGNOSE = "companion_diagnose"
ROLE_COMPANION_EXTRACT = "companion_extract"

#: ``input_payload.task_kind`` on the diagnoser role (ADR-254 D6 addendum).
TASK_KIND_DIAGNOSE = "diagnose"
TASK_KIND_STUDY_AIDS = "study_aids"
TASK_KIND_PRACTICE_TEST = "practice_test"
OUTPUT_TASK_KINDS = (TASK_KIND_STUDY_AIDS, TASK_KIND_PRACTICE_TEST)

#: ``max_questions`` default for a practice test (the agent's own default too).
DEFAULT_PRACTICE_TEST_MAX_QUESTIONS = 8

#: The three keys the contract names for ``edges_json``; nothing else leaks
#: (no confidence, no strength, no tags).
_EDGE_KEYS = ("concept_key", "concept_label", "descriptor_json")

_THREAD_SEGMENTS = 2
_GS_SCHEME = "gs://"


class _ExecutorLike(Protocol):
    """The ADR-253 dispatch seam (``PubSubAgentExecutor``)."""

    async def execute(
        self,
        *,
        execution_id: str,
        tenant_id: str,
        agid: str,
        agent_role: str,
        input_payload: str,
        workflow_id: str = ...,
        prompt_template_id: str = ...,
    ) -> Any: ...


def _thread_id_from_langgraph() -> str:
    """Read the thread the run is parked on from LangGraph's own config."""
    from langgraph.config import get_config

    config = get_config() or {}
    thread_id = str((config.get("configurable") or {}).get("thread_id") or "")
    if not thread_id:
        raise ValueError(
            "companion_diagnosis dispatch: no thread_id on the graph config; a "
            "dispatch parked without one could never be resumed"
        )
    return thread_id


def upload_id_from_thread(thread_id: str) -> str:
    """Recover ``upload_id`` from the deterministic growth-edge thread key.

    Refuses anything that is not the documented 2-segment
    ``{tenant_id}:{upload_id}`` shape rather than guessing. Dispatching on a
    guessed upload id would write the outbox row against the wrong aggregate,
    which is worse than not dispatching at all.
    """
    parts = (thread_id or "").split(":")
    if len(parts) != _THREAD_SEGMENTS or not all(p.strip() for p in parts):
        raise ValueError(
            f"companion_diagnosis dispatch: thread {thread_id!r} is not the growth-edge "
            f"shape '{{tenant_id}}:{{upload_id}}' (checkpointer/factory.py:105). "
            f"Refusing to guess the upload id."
        )
    return parts[1].strip()


def edges_json_for_task(edges: list[dict[str, Any]]) -> str:
    """Project the published edges onto the contract's three keys, as a JSON
    array STRING (the agent parses ``edges_json`` from session state)."""
    projected = [{k: ("" if e.get(k) is None else e.get(k)) for k in _EDGE_KEYS} for e in edges if isinstance(e, dict)]
    return json.dumps(projected, separators=(",", ":"), ensure_ascii=False)


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _payload_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


class PubSubDiagnoserAdapter:
    """Crew ``Diagnoser`` + ``OutputTaskRunner`` ports over the diagnoser lane."""

    def __init__(
        self,
        *,
        executor: _ExecutorLike,
        model_id: str,
        agid: str = "",
        thread_id_provider: Any = _thread_id_from_langgraph,
    ) -> None:
        self._executor = executor
        self._model_id = model_id
        self._agid = agid
        self._thread_id = thread_id_provider

    async def diagnose(
        self,
        *,
        extracted_text: str,
        structured_clues_block: str,
        upload_kind: str,
        tenant_id: str,
        gcid: str,
        traceparent: str,
        tracestate: str,
    ) -> DiagnoseResult:
        thread_id = self._thread_id()
        upload_id = upload_id_from_thread(thread_id)

        # `extracted_text` and `clues_block` are the two names the diagnoser's
        # per-turn InstructionProvider reads, and it FAILS (permanent) on a
        # missing `extracted_text`, so these names are the wire contract.
        # gcid / traceparent / tracestate are read back out by the executor for
        # the ENVELOPE, and the Go merge skips them from the body for that reason.
        payload = {
            "task_kind": TASK_KIND_DIAGNOSE,
            "extracted_text": extracted_text,
            "clues_block": structured_clues_block,
            "upload_kind": upload_kind,
            "gcid": gcid,
            "traceparent": traceparent,
            "tracestate": tracestate,
        }
        input_payload = _payload_json(payload)

        response = await self._executor.execute(
            execution_id=f"{upload_id}:diagnose:{_digest(input_payload)}",
            tenant_id=tenant_id,
            agid=self._agid,
            agent_role=ROLE_COMPANION_DIAGNOSE,
            input_payload=input_payload,
            workflow_id=upload_id,
        )

        return DiagnoseResult(
            diagnosis_json=str(getattr(response, "output_payload", "") or ""),
            model_used=self._model_id,
            input_tokens=int(getattr(response, "input_tokens", 0) or 0),
            output_tokens=int(getattr(response, "output_tokens", 0) or 0),
        )

    async def run_task(
        self,
        *,
        task_kind: str,
        edges: list[dict[str, Any]],
        tenant_id: str,
        gcid: str,
        traceparent: str,
        tracestate: str,
        max_questions: int = DEFAULT_PRACTICE_TEST_MAX_QUESTIONS,
    ) -> str:
        """Dispatch one learner-output task on the diagnoser role.

        Returns the agent's ``output_payload`` TEXT (study_aids: the strict JSON
        object; practice_test: ``{title, questions[], rejected_reason?}``).
        The crew node screens and parses it; a FAILED completion surfaces as
        ``AgentDispatchError`` from the executor.
        """
        kind = (task_kind or "").strip()
        if kind not in OUTPUT_TASK_KINDS:
            raise ValueError(
                f"companion_diagnosis dispatch: unknown output task_kind {task_kind!r}; "
                f"expected one of {list(OUTPUT_TASK_KINDS)}"
            )
        thread_id = self._thread_id()
        upload_id = upload_id_from_thread(thread_id)

        payload: dict[str, Any] = {
            "task_kind": kind,
            "edges_json": edges_json_for_task(edges),
            "gcid": gcid,
            "traceparent": traceparent,
            "tracestate": tracestate,
        }
        if kind == TASK_KIND_PRACTICE_TEST:
            payload["max_questions"] = int(max_questions)
        input_payload = _payload_json(payload)

        response = await self._executor.execute(
            execution_id=f"{upload_id}:{kind}:{_digest(input_payload)}",
            tenant_id=tenant_id,
            agid=self._agid,
            agent_role=ROLE_COMPANION_DIAGNOSE,
            input_payload=input_payload,
            workflow_id=upload_id,
        )
        return str(getattr(response, "output_payload", "") or "")


class PubSubExtractorAdapter:
    """Crew ``Extractor`` port over the extractor lane (by reference)."""

    def __init__(
        self,
        *,
        executor: _ExecutorLike,
        agid: str = "",
        thread_id_provider: Any = _thread_id_from_langgraph,
    ) -> None:
        self._executor = executor
        self._agid = agid
        self._thread_id = thread_id_provider

    async def extract(
        self,
        *,
        source_blob_uri: str,
        source_mime_type: str,
        tenant_id: str,
        gcid: str,
        traceparent: str,
        tracestate: str,
    ) -> ExtractResult:
        uri = (source_blob_uri or "").strip()
        if not uri.startswith(_GS_SCHEME) or len(uri) <= len(_GS_SCHEME):
            # The extractor injects a FileData part by reference; anything that
            # is not a gs:// object is a permanent failure on the agent. Refuse
            # here, with nothing parked and nothing metered.
            raise ValueError(f"companion_extract: source_blob_uri must be a gs:// object, got {source_blob_uri!r}")
        thread_id = self._thread_id()
        upload_id = upload_id_from_thread(thread_id)

        payload = {
            "source_blob_uri": uri,
            "source_mime_type": source_mime_type,
            "gcid": gcid,
            "traceparent": traceparent,
            "tracestate": tracestate,
        }
        response = await self._executor.execute(
            # One extract per upload: stable across a node re-execution (the
            # rebuilt request collides with the queued row instead of
            # dispatching twice). The crew never loops through extract.
            execution_id=f"{upload_id}:extract",
            tenant_id=tenant_id,
            agid=self._agid,
            agent_role=ROLE_COMPANION_EXTRACT,
            input_payload=_payload_json(payload),
            workflow_id=upload_id,
        )
        return ExtractResult(
            text=str(getattr(response, "output_payload", "") or ""),
            model_used="",
            input_tokens=int(getattr(response, "input_tokens", 0) or 0),
            output_tokens=int(getattr(response, "output_tokens", 0) or 0),
        )


__all__ = [
    "DEFAULT_PRACTICE_TEST_MAX_QUESTIONS",
    "OUTPUT_TASK_KINDS",
    "ROLE_COMPANION_DIAGNOSE",
    "ROLE_COMPANION_EXTRACT",
    "TASK_KIND_DIAGNOSE",
    "TASK_KIND_PRACTICE_TEST",
    "TASK_KIND_STUDY_AIDS",
    "PubSubDiagnoserAdapter",
    "PubSubExtractorAdapter",
    "edges_json_for_task",
    "upload_id_from_thread",
]
