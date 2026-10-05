"""Crew runner for the graduated Growth-Edge analyser (ADR-205 WS-2 / CHO-1973).

The load-bearing seam between the Pub/Sub subscription and the FE resume route:

  * ``handle_uploaded(event)`` — one ``WeaknessDocUploaded`` → build the initial
    state → ``ainvoke`` the graph, which runs to the HITL ``interrupt()`` (or to
    a fail-loud/BLOCK terminal). On the interrupt it builds the canonical FE
    review PANEL (CHO-1973 Wave A) and returns it on the ``RunResult`` so the
    subscriber can emit ``weakness.review_pending.v1``. The PostgresSaver
    checkpoint persists the paused run.
  * ``resume(thread_id, decision)`` — the PANEL-SHAPE learner decision (CHO-1973
    Wave C) → ``ainvoke(Command(resume=...))`` → the graph completes
    (synthesise → publish → emit) on ``confirm`` or RE-PAUSES on ``reiterate``
    (returning a refreshed panel the FE re-enters review with).

``thread_id`` is DETERMINISTIC on ``{tenant}:{upload}`` ONLY (no run_id) — the
crew runs exactly one analysis per upload, so a resume after a pod restart finds
the checkpointed run from tenant + upload alone (D6 P1 pod-death survival). A
run_id is still minted internally for cost attribution; it just does not key the
checkpoint. The graph itself is idempotent on re-publish.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from langgraph.types import Command

from chora_ai_kernel_orchestrator.adapter.checkpointer.factory import (
    build_weakness_thread_id,
)
from chora_ai_kernel_orchestrator.adapter.pubsub.agent_dispatch import (
    DISPATCH_INTERRUPT_KEY,
)
from chora_ai_kernel_orchestrator.domain.state.state import new_request_id
from chora_ai_kernel_orchestrator.domain.weakness_analyser_crew.panel import (
    build_review_panel,
)

logger = logging.getLogger(__name__)

# One-shot "loud" gap marker (DARK path; logged once per process so the honest
# gap is visible in logs without flooding them).
_familiar_unresolved_logged = False


class WeaknessCrewError(ValueError):
    """Raised when an uploaded event is missing fields required to run."""


class FamiliarResolver(Protocol):
    """Resolves a learner's active Familiar (id / name / species) for the review
    panel. The real adapter is a chora-consumption gRPC client (the Familiar
    entity lives in chora_consumption — cross-DB reads are forbidden). When no
    resolver is wired the runner emits an EMPTY familiar (consumption enriches it
    on the read side, since it OWNS the entity AND receives the event)."""

    async def resolve(self, *, tenant_id: str, gcid: str) -> dict[str, Any] | None: ...


@dataclass(frozen=True)
class RunResult:
    """Outcome of one ``ainvoke`` (initial or resume)."""

    interrupted: bool
    thread_id: str
    run_id: str
    governance_status: str | None = None
    review_payload: dict[str, Any] | None = None
    edges: list[dict[str, Any]] = field(default_factory=list)
    published_row_id: str | None = None
    # WS-7 — the orchestrator-produced learner outputs (study aids / practice
    # test) surface to the FE here, NOT on the binary weakness.analyzed.v1 event
    # (which carries only the edges). Each entry: {type, content, metered}.
    generated_outputs: list[dict[str, Any]] = field(default_factory=list)
    # CHO-1973 Wave A/C — the canonical FE review panel (proposed_edges /
    # candidate_struggles / available_outputs / familiar). Present when the run is
    # interrupted (the subscriber emits it; the resume route returns it on
    # reiterate); None once the run completes.
    review_panel: dict[str, Any] | None = None
    # ADR-253: the run is suspended on an AGENT DISPATCH, not on a human.
    # Both arrive as __interrupt__, and before this existed the runner treated
    # every interrupt as the review one, tried to build a panel out of a
    # dispatch envelope, and raised. The graph subscriber turned that raise
    # into a NACK, the broker redelivered, and the whole graph re-ran from the
    # top, including a paid extract call, five times over.
    #
    # A caller seeing this must ACK: the park is committed with its dispatch
    # outbox row in one transaction (D3a) and the completion consumer resumes
    # the thread. Redelivery is not recovery here, it is duplicate work.
    awaiting_agent: bool = False


def initial_state_from_event(event: Mapping[str, Any], *, run_id: str) -> dict[str, Any]:
    """Project a decoded WeaknessDocUploaded into the graph's initial state."""
    return {
        "run_id": run_id,
        "tenant_id": str(event.get("tenant_id", "")).strip(),
        "learner_gcid": str(event.get("learner_gcid", "")).strip(),
        "upload_id": str(event.get("upload_id", "")).strip(),
        "source_blob_uri": str(event.get("source_blob_uri", "")).strip(),
        "source_mime_type": str(event.get("source_mime_type", "")),
        "upload_kind": str(event.get("upload_kind", "")).strip(),
        "structured_clues": dict(event.get("structured_clues") or {}),
        "requested_outputs": dict(event.get("requested_outputs") or {}),
        "reservation_id": str(event.get("reservation_id", "")).strip(),
        "context_hint": str(event.get("context_hint", "")),
        "traceparent": str(event.get("traceparent", "")),
        "tracestate": str(event.get("tracestate", "")),
    }


class WeaknessAnalyserCrewRunner:
    """Drives the compiled weakness-analyser graph (DI'd for testability).

    ``output_prices`` maps a metered output kind → mana price (resolved from env
    at the composition root; free kinds are always 0). ``familiar_resolver`` is
    the optional Familiar lookup port; absent ⇒ empty familiar + a one-shot loud
    log (consumption enriches on read).
    """

    def __init__(
        self,
        *,
        graph: Any,
        output_prices: Mapping[str, int] | None = None,
        familiar_resolver: FamiliarResolver | None = None,
        review_pending_publisher: Any | None = None,
    ) -> None:
        self._graph = graph
        self._output_prices = dict(output_prices or {})
        self._familiar_resolver = familiar_resolver
        # Used ONLY by handle_completion. On the Pub/Sub transport the run
        # reaches the HITL review inside a completion resume, not inside the
        # inbound start, so the panel has to be emitted from there.
        self._review_pending_publisher = review_pending_publisher

    async def handle_uploaded(self, event: Mapping[str, Any]) -> RunResult:
        run_id = str(event.get("run_id") or "").strip() or new_request_id()
        tenant_id = str(event.get("tenant_id", "")).strip()
        upload_id = str(event.get("upload_id", "")).strip()
        learner_gcid = str(event.get("learner_gcid", "")).strip()
        blob_uri = str(event.get("source_blob_uri", "")).strip()
        if not (tenant_id and upload_id and learner_gcid and blob_uri):
            raise WeaknessCrewError(
                "WeaknessDocUploaded missing required fields (tenant_id / upload_id / learner_gcid / source_blob_uri)"
            )
        thread_id = build_weakness_thread_id(tenant_id=tenant_id, upload_id=upload_id)
        state = initial_state_from_event(event, run_id=run_id)
        config = {"configurable": {"thread_id": thread_id}}
        logger.info("weakness_crew_runner.handle_uploaded", extra={"thread_id": thread_id, "upload_id": upload_id})
        terminal = await self._graph.ainvoke(state, config=config)
        return await self._result(terminal, thread_id=thread_id, fallback_run_id=run_id)

    async def handle_completion(self, completion: Mapping[str, Any]) -> RunResult:
        """Resume a run parked on an agent dispatch, from one completion (D2).

        The completion carries the thread it belongs to, so this consumer keeps
        no state of its own. Refuses a completion with no ``thread_id``: resuming
        a guessed thread would inject one learner's diagnosis into another
        learner's run.

        ⚠ The review_pending emit lives here for this transport, and that is a
        MOVE rather than a duplication. Under ADR-169 the inbound subscriber
        drove the graph all the way to the HITL interrupt and emitted the panel
        itself. On the bus it parks at the diagnose dispatch long before HITL, so
        the run now reaches the human gate inside THIS resume. Left only on the
        inbound path, the learner is never told their review is ready. The two
        cannot both fire on one run: on this transport the inbound path returns
        at the dispatch park, and on HTTP no completion is ever consumed.
        """
        thread_id = str(completion.get("thread_id") or "").strip()
        if not thread_id:
            raise ValueError(
                "weakness_crew_runner.handle_completion: completion carries no "
                "thread_id; refusing to resume a guessed thread"
            )
        logger.info(
            "weakness_crew_runner.completion",
            extra={"thread_id": thread_id, "status": completion.get("status")},
        )
        terminal = await self._graph.ainvoke(
            Command(resume=dict(completion)),
            config={"configurable": {"thread_id": thread_id}},
        )
        result = await self._result(terminal, thread_id=thread_id, fallback_run_id="")
        # A multi-hop crew parks again on its next dispatch. That is not a
        # review and must not be announced as one.
        review_publisher = getattr(self, "_review_pending_publisher", None)
        if result.review_panel and review_publisher is not None:
            await review_publisher.publish_review_pending(
                panel=result.review_panel,
                traceparent=str(completion.get("traceparent") or ""),
                tracestate=str(completion.get("tracestate") or ""),
            )
        return result

    async def resume(self, *, thread_id: str, decision: Mapping[str, Any]) -> RunResult:
        config = {"configurable": {"thread_id": thread_id}}
        logger.info("weakness_crew_runner.resume", extra={"thread_id": thread_id, "action": decision.get("action")})
        terminal = await self._graph.ainvoke(Command(resume=dict(decision)), config=config)
        return await self._result(terminal, thread_id=thread_id, fallback_run_id="")

    async def _result(self, terminal: Mapping[str, Any], *, thread_id: str, fallback_run_id: str) -> RunResult:
        interrupts = terminal.get("__interrupt__") or []
        raw_interrupt = interrupts[0].value if interrupts else None
        # Discriminate the two kinds of park by the wrapper the dispatch
        # executor stamps (``wrap_for_interrupt``), NOT by shape-sniffing the
        # payload: a review payload that happened to be missing its keys would
        # otherwise be mistaken for a dispatch and silently never reviewed.
        awaiting_agent = isinstance(raw_interrupt, Mapping) and (DISPATCH_INTERRUPT_KEY in raw_interrupt)
        review_payload = None if awaiting_agent else raw_interrupt
        # run_id rides the checkpointed state (set on the initial invoke); the
        # thread no longer carries it. Fall back to the just-minted id on the
        # initial invoke where the terminal may not echo it back.
        run_id = str(terminal.get("run_id") or fallback_run_id or "")
        body = terminal.get("analyzed_body") or {}
        # A review interrupt carries the candidate edges on its payload. A
        # dispatch park (ADR-254 D5: the learner outputs park AFTER the analyzed
        # publish) carries none, so the reviewed edges come from the analyzed
        # body; before this a confirm answered the FE with [].
        edges = (review_payload or {}).get("edges", []) if interrupts and not awaiting_agent else body.get("edges", [])
        review_panel = await self._build_panel(review_payload) if review_payload else None
        return RunResult(
            interrupted=bool(interrupts),
            thread_id=thread_id,
            run_id=run_id,
            governance_status=terminal.get("governance_status"),
            review_payload=review_payload,
            edges=list(edges),
            published_row_id=terminal.get("published_row_id"),
            generated_outputs=list(terminal.get("generated_outputs") or []),
            review_panel=review_panel,
            awaiting_agent=awaiting_agent,
        )

    async def _build_panel(self, review_payload: Mapping[str, Any]) -> dict[str, Any]:
        """Build the canonical FE review panel from the interrupt payload."""
        tenant_id = str(review_payload.get("tenant_id", ""))
        learner_gcid = str(review_payload.get("learner_gcid", ""))
        familiar = await self._resolve_familiar(tenant_id=tenant_id, gcid=learner_gcid)
        # candidate_struggles source = the diagnosed-but-below-promotion-threshold
        # concepts the graph now retains on the interrupt (CHO-1973 Q2). The panel
        # builder ranks/dedups/caps/cleans them into the bounded picker; an empty
        # list is valid (nothing sub-threshold was diagnosed) — never fabricated.
        return build_review_panel(
            candidate_edges=list(review_payload.get("edges") or []),
            upload_id=str(review_payload.get("upload_id", "")),
            tenant_id=tenant_id,
            learner_gcid=learner_gcid,
            requested_outputs=review_payload.get("requested_outputs") or {},
            output_prices=self._output_prices,
            familiar=familiar,
            candidate_struggles=review_payload.get("candidate_struggles") or None,
        )

    async def _resolve_familiar(self, *, tenant_id: str, gcid: str) -> dict[str, Any] | None:
        if self._familiar_resolver is None:
            global _familiar_unresolved_logged
            if not _familiar_unresolved_logged:
                logger.warning(
                    "weakness_crew_runner.familiar_resolver_unconfigured: emitting "
                    "empty familiar on review_pending (chora-consumption owns the "
                    "Familiar entity and enriches it on the read side)"
                )
                _familiar_unresolved_logged = True
            return None
        try:
            return await self._familiar_resolver.resolve(tenant_id=tenant_id, gcid=gcid)
        except Exception:  # noqa: BLE001 — Familiar is non-blocking; never sink the review
            logger.exception("weakness_crew_runner.familiar_resolve_failed")
            return None


__all__ = [
    "FamiliarResolver",
    "RunResult",
    "WeaknessAnalyserCrewRunner",
    "WeaknessCrewError",
    "initial_state_from_event",
]
