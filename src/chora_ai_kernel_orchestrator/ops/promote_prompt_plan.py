"""CHO-2368 P2 - operator CLI for the prompt-plan promotion gate.

The registry has NO authoring HTTP surface (that is CHO-2363's authoring UI).
This CLI is the drafts' only entry into the six-state gate, composing the SAME
domain pieces the in-pod wiring uses (prompt_promotion_audit_wiring.py):
PostgresPromptOverrideRepository + PromptActivationAuditEmitter +
HITLDecisionOutboxWriter/PromptHITLRequestEmitter + PromptPromotionService.

Run locally over the cloudsql port-forward::

    export CHORA_AI_KERNEL_PG_DSN='postgres://...@127.0.0.1:<port>/chora_ai_kernel?sslmode=disable'
    export CHORA_PUBSUB_PROJECT=chora-489812
    .venv/bin/python -m chora_ai_kernel_orchestrator.ops.promote_prompt_plan \\
        create-draft --agent qgen_question --operator <gcid>

Load-bearing rules (pinned by tests/unit/test_promote_prompt_plan_cli.py):

- the eval-passed edge is ALWAYS ``request_hitl_approval`` (never
  ``mark_eval_passed``, which flips the same edge silently and would strand a
  plan in pending_hitl with an empty O+ queue);
- this tool NEVER starts an outbox dispatcher: it only INSERTs outbox rows,
  which the live orchestrator pod's single dispatcher drains (single-drain
  invariant per the double-dispatch memory);
- ``archive`` drives the guarded ACTIVE -> ARCHIVED edge (the rollback lever;
  ARCHIVED is terminal, so re-promotion mints a FRESH plan);
- that fresh plan takes a FRESH label (CHO-2379): ``create-draft`` reads the
  agent's platform catalogue and auto-bumps to the next free patch, and
  ``request-hitl`` names the PLAN's own label rather than the embedded
  revision constant, so superseded attempts stay distinguishable.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from typing import Any

from ..domain.prompt_registry import revisions_110 as rev
from ..domain.prompt_registry.state_machine import PromptPlanState
from ..domain.prompt_registry.version_labels import next_version_label

_PLATFORM_SCOPE = "platform"

#: A platform-scope plan carries NO tenant, but the HITL gate must still land
#: in a tenant's O+ Human-Oversight queue (the outbox tenant_id column is
#: UUID NOT NULL and /api/hitl/pending filters by tenant). Platform gates are
#: therefore routed to the platform tenant's queue, which is where the
#: operator reviews them. Overridable for a tenant-scope promotion.
_PLATFORM_GATE_TENANT_ENV = "CHORA_PLATFORM_GATE_TENANT_ID"
_DEFAULT_PLATFORM_GATE_TENANT = "00000000-0000-7000-8000-000000000001"


def _say(message: str) -> None:
    """Operator-facing CLI output (stdout; not logging, not diagnostics)."""
    sys.stdout.write(message + "\n")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="promote_prompt_plan")
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("create-draft", help="create the next-patch draft plan + segments")
    c.add_argument("--agent", required=True, choices=sorted(rev.WAVE_AGENTS))
    c.add_argument("--operator", required=True, help="creator GCID (audit)")

    s = sub.add_parser("submit-for-eval", help="draft -> pending_eval")
    s.add_argument("--plan-id", required=True)

    h = sub.add_parser("request-hitl", help="pending_eval -> pending_hitl via the HITL gate")
    h.add_argument("--plan-id", required=True)
    h.add_argument("--agent", required=True, choices=sorted(rev.WAVE_AGENTS))
    h.add_argument("--eval-run-id", required=True)
    h.add_argument("--requester", required=True, help="requester GCID")

    st = sub.add_parser("status", help="print the plan row")
    st.add_argument("--plan-id", required=True)

    a = sub.add_parser("archive", help="guarded ACTIVE -> ARCHIVED (rollback lever)")
    a.add_argument("--plan-id", required=True)
    return p


async def run_command(argv: list[str], *, repo: Any, svc: Any) -> str | None:
    """Execute one subcommand against injected repo + service (unit-testable)."""
    args = _build_parser().parse_args(argv)

    if args.command == "create-draft":
        overrides = rev.overrides_for(args.agent)  # KeyError on non-wave agents
        notes = {s.segment_id: s.note for s in rev.revision_segments() if s.agent_id == args.agent}
        # CHO-2379: ARCHIVED is terminal, so every re-promotion mints a FRESH
        # plan and must carry a FRESH label. Reusing the revision constant left
        # superseded attempts unreachable behind the catalogue's newest-wins
        # single-version read (and collided the derived eval-run name).
        catalogue = await repo.list_agent_versions(agent_id=args.agent)
        version_label = next_version_label(
            (v.version_label for v in catalogue),
            base=rev.REVISION_VERSION_LABEL,
        )
        plan_id = await repo.create_draft_plan(
            plan_code=f"{args.agent}-{version_label}",
            scope=_PLATFORM_SCOPE,
            tenant_id=None,
            created_by=args.operator,
            agent_id=args.agent,
            version_label=version_label,
        )
        for segment_id, body in sorted(overrides.items()):
            await repo.add_segment(
                plan_id=plan_id,
                agent_id=args.agent,
                segment_id=segment_id,
                body=body,
                note=notes.get(segment_id, ""),
            )
        _say(
            f"draft created: plan_id={plan_id} agent={args.agent} version={version_label} segments={sorted(overrides)}"
        )
        return str(plan_id)

    if args.command == "submit-for-eval":
        await svc.submit_for_eval(args.plan_id)
        _say(f"submitted for eval: plan_id={args.plan_id} (status pending_eval)")
        return str(args.plan_id)

    if args.command == "request-hitl":
        # CHO-2379: the O+ gate card names the plan being approved, so read the
        # PLAN's own label. The embedded revision constant made every card read
        # 1.1.0 regardless of which attempt was on the gate.
        plan = await repo.get_plan(args.plan_id)
        if plan is None:
            raise ValueError(f"request-hitl: plan {args.plan_id!r} not found; refusing to gate an unknown plan")
        version_label = plan.version_label or rev.REVISION_VERSION_LABEL
        segment_ids = sorted(rev.overrides_for(args.agent))
        gate_tenant = (os.getenv(_PLATFORM_GATE_TENANT_ENV) or _DEFAULT_PLATFORM_GATE_TENANT).strip()
        await svc.request_hitl_approval(
            args.plan_id,
            requester_gcid=args.requester,
            tenant_id=gate_tenant,
            scope=_PLATFORM_SCOPE,
            eval_run_id=args.eval_run_id,
            agent_ids=[args.agent],
            segment_ids=segment_ids,
            summary=(
                f"prompt override plan {args.agent} {version_label} "
                f"passed eval {args.eval_run_id}; segments {', '.join(segment_ids)}; "
                "awaiting human sign-off"
            ),
        )
        _say(f"HITL requested: plan_id={args.plan_id} eval_run_id={args.eval_run_id}")
        return str(args.plan_id)

    if args.command == "status":
        plan = await repo.get_plan(args.plan_id)
        if plan is None:
            _say(f"plan {args.plan_id}: NOT FOUND")
            return None
        _say(
            f"plan {args.plan_id}: status={plan.status} agent={plan.agent_id} "
            f"version={plan.version_label} scope={plan.scope}"
        )
        return str(args.plan_id)

    if args.command == "archive":
        await repo.update_plan_status(
            plan_id=args.plan_id,
            expected_from=PromptPlanState.ACTIVE,
            to=PromptPlanState.ARCHIVED,
        )
        _say(f"archived: plan_id={args.plan_id} (rollback lever; terminal)")
        return str(args.plan_id)

    raise ValueError(f"unhandled command {args.command!r}")  # pragma: no cover


async def _amain(argv: list[str]) -> None:  # pragma: no cover - integration glue
    import psycopg

    from ..adapter.pg.prompt_override_repository import (
        PostgresPromptOverrideRepository,
    )
    from ..adapter.pubsub.hitl_decision_outbox_writer import HITLDecisionOutboxWriter
    from ..adapter.pubsub.prompt_activation_audit_emitter import (
        PromptActivationAuditEmitter,
    )
    from ..adapter.pubsub.prompt_hitl_request_emitter import PromptHITLRequestEmitter
    from ..adapter.secrets import resolve_dsn
    from ..domain.prompt_registry.transition_service import PromptPromotionService

    dsn = resolve_dsn()
    if not dsn:
        sys.exit("CHORA_AI_KERNEL_PG_DSN unset (point it at the port-forward)")
    project = (os.getenv("CHORA_PUBSUB_PROJECT") or "").strip()
    if not project:
        sys.exit("CHORA_PUBSUB_PROJECT unset")

    conn = await psycopg.AsyncConnection.connect(dsn, autocommit=True)
    try:
        repo = PostgresPromptOverrideRepository(conn=conn)
        svc = PromptPromotionService(
            repo=repo,
            audit_emitter=PromptActivationAuditEmitter(conn=conn, source_project=project),
            hitl_emitter=PromptHITLRequestEmitter(writer=HITLDecisionOutboxWriter(conn=conn, source_project=project)),
        )
        await run_command(argv, repo=repo, svc=svc)
    finally:
        await conn.close()


def main() -> None:  # pragma: no cover - console entry
    asyncio.run(_amain(sys.argv[1:]))


if __name__ == "__main__":  # pragma: no cover
    main()
