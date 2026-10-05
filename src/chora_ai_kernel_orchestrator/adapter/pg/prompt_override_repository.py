"""psycopg adapter implementing ``PromptOverrideRepository`` (ADR-197 M-B.1).

Resolves the active override segments for ``(tenant_id, agent_id)`` at each scope
in ONE indexed query (the precedence choice between scopes is the domain
resolver's job — this adapter just returns what's active per scope).

RLS-aware (mirrors the ``set_config('chora.tenant_id', $1, true)`` tx-local
pattern used by chora-notifications ``adapter/pg/runtime.go`` and the
``chora.tenant_id`` GUC convention in migration 0001 / 0007):

* The query runs inside an explicit transaction (``conn.transaction()``) so the
  ``set_config(..., is_local => true)`` GUC stays in effect for the SELECT
  regardless of the connection's autocommit mode (the shared orchestrator conn
  is autocommit=OFF on the qgen wiring / ON on the OE wiring — store.py §note).
* Only ``chora.tenant_id`` is set — NOT ``chora.role``. The app path is
  tenant-scoped and MUST NOT set role='admin' (that would bypass RLS).
* Platform-scope plans are world-readable by the 0007 policy, so they resolve
  even with no tenant set; tenant-scope rows resolve only for the matching
  tenant. The WHERE clause is NULLIF-safe so an empty tenant never casts
  ``''::uuid`` (the 22P02 RLS-cast bug; chora-identity mig 0019 / FU-4b).

Empty result → no override → ``[]`` (the resolver returns the embedded default).
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

import uuid_utils as _uuid_utils

from ...domain.prompt_registry.models import (
    VALID_SCOPES,
    CataloguePlanVersion,
    CatalogueSegmentView,
    PromptPlanRecord,
    ScopeOverride,
)
from ...domain.prompt_registry.state_machine import (
    PromptPlanState,
    StalePromptTransitionError,
)

logger = logging.getLogger(__name__)

# Bind chora.tenant_id for the current transaction (tx-local — the third arg).
# Parameterised so there is no SET LOCAL string interpolation.
_SET_TENANT_SQL = "SELECT set_config('chora.tenant_id', %s, TRUE)"

# Bind chora.role='admin' for the current transaction (tx-local). Used by the
# plan_id-only WRITE paths (add_segment / update_plan_status) which carry no
# tenant_id and so cannot use the tenant-scoped RLS rung. This is NOT an
# RLS-bypass surface — it satisfies the 0007 policy's OWN explicit admin
# branch (`OR current_setting('chora.role') = 'admin'`), authored precisely for
# O+ auditor / platform-ops plan management. RLS stays ENABLED + enforced; the
# bind reverts at transaction end (tx-local), so nothing leaks to the next op.
_SET_ROLE_ADMIN_SQL = "SELECT set_config('chora.role', 'admin', TRUE)"

# Read a single plan's scope/tenant/status (ADR-197 M-C.2 — the HITL-approval
# handler needs the scope + tenant to drive activation + the status for an
# idempotent replay no-op). Runs under the admin branch (plan_id-only, no tenant).
_GET_PLAN_SQL = """
    SELECT plan_id::text, scope, tenant_id::text, status, agent_id, version_label
      FROM prompt_override_plan
     WHERE plan_id = %(plan_id)s
"""

# --- Promotion write SQL (ADR-197 M-C.1) -------------------------------------

_INSERT_PLAN_SQL = """
    INSERT INTO prompt_override_plan
        (plan_id, plan_code, scope, tenant_id, status, created_by,
         agent_id, version_label)
    VALUES
        (%(plan_id)s, %(plan_code)s, %(scope)s, %(tenant_id)s, 'draft',
         %(created_by)s, %(agent_id)s, %(version_label)s)
"""

_INSERT_SEGMENT_SQL = """
    INSERT INTO prompt_override_segment
        (segment_row_id, plan_id, agent_id, segment_id, body, content_hash, note)
    VALUES
        (%(segment_row_id)s, %(plan_id)s, %(agent_id)s, %(segment_id)s,
         %(body)s, %(content_hash)s, %(note)s)
"""

# Guarded transition: only flips status when the row is still in expected_from
# (optimistic concurrency — a stale/illegal change matches 0 rows → fail-loud).
# COALESCE keeps the existing approved_by / eval_run_id when the caller passes
# NULL (e.g. submit_for_eval threads neither).
_UPDATE_STATUS_SQL = """
    UPDATE prompt_override_plan
       SET status      = %(to)s,
           approved_by = COALESCE(%(approved_by)s, approved_by),
           eval_run_id = COALESCE(%(eval_run_id)s, eval_run_id)
     WHERE plan_id = %(plan_id)s
       AND status  = %(expected_from)s
"""

# Archive whatever OVERRIDE plan is currently active for this scope/tenant AND
# agent (excluding the one being activated). IS NOT DISTINCT FROM handles the
# NULL tenant_id of a platform-scope plan and the NULL agent_id of a legacy
# agent-less plan. kind='override' is the 0009 carve-out: the always-active
# baseline catalogue rows must NEVER be archived by an activation, and the
# agent scoping keeps a per-agent 1.1.0 wave from archiving OTHER agents'
# active overrides (the pre-0009 failure mode). May match 0 rows (the
# first-ever activation) - legal, so not rowcount-guarded.
_ARCHIVE_PRIOR_ACTIVE_SQL = """
    UPDATE prompt_override_plan
       SET status = 'archived'
     WHERE status   = 'active'
       AND kind     = 'override'
       AND plan_id <> %(plan_id)s
       AND scope    = %(scope)s
       AND tenant_id IS NOT DISTINCT FROM %(tenant_id)s
       AND agent_id IS NOT DISTINCT FROM %(agent_id)s
"""

# Promote the plan to active. Guarded on pending_hitl (the only legal
# predecessor) — 0 rows = a stale/illegal activation → fail-loud. The 0007
# partial unique index makes the prior-archive + this-activate pair safe.
_ACTIVATE_SQL = """
    UPDATE prompt_override_plan
       SET status      = 'active',
           approved_by = %(approved_by)s,
           approved_at = now()
     WHERE plan_id = %(plan_id)s
       AND status  = 'pending_hitl'
"""

# Select the active override segments for the agent at each scope. RLS already
# filters tenant rows; the explicit NULLIF-safe tenant predicate is defence in
# depth (correct even if a caller ever sets chora.role='admin'). At most one
# active override plan exists per scope+agent (0009 partial unique indexes), so
# every row of a given scope shares one plan.
#
# kind='override' is the 0009 carve-out: baseline catalogue rows duplicate the
# EMBEDDED content and must never resolve as overrides (a baseline resolving
# would stamp prompt_source=platform_override on unchanged prompts and collide
# with a real 1.1.0 in the resolver's one-per-scope fail-loud check).
#
# The version token is the display version_label ('v1', '1.1.0') so
# prompt_conditions stamps + the O+ panel show human versions; legacy plans
# without a label fall back to the opaque plan_id.
_QUERY = """
    SELECT p.scope,
           COALESCE(p.version_label, p.plan_id::text) AS version,
           s.segment_id,
           s.body
      FROM prompt_override_plan p
      JOIN prompt_override_segment s ON s.plan_id = p.plan_id
     WHERE p.status = 'active'
       AND p.kind = 'override'
       AND s.agent_id = %(agent_id)s
       AND (
            p.scope = 'platform'
         OR (p.scope = 'tenant'
             AND p.tenant_id = NULLIF(%(tenant_id)s, '')::uuid)
       )
     ORDER BY p.scope, s.segment_id
"""

# --- Catalogue reads (CHO-2368) ----------------------------------------------
# The O+ read API lists PLATFORM plans only: the catalogue is the platform
# prompt surface (baselines + platform overrides). Platform rows pass the 0007
# RLS policies' scope branch with no GUC bound. activated_at is approved_at for
# gated overrides and effective_from for the seeded always-active baselines.

_LIST_VERSIONS_SQL = """
    SELECT p.version_label,
           p.kind,
           p.status,
           COALESCE(p.approved_at,
                    CASE WHEN p.kind = 'baseline' THEN p.effective_from END
           )::text AS activated_at,
           p.approved_by::text,
           p.eval_run_id,
           p.created_at::text,
           p.plan_code,
           p.plan_id::text
      FROM prompt_override_plan p
     WHERE p.scope = 'platform'
       AND p.agent_id = %(agent_id)s
       AND p.version_label IS NOT NULL
     ORDER BY p.created_at, p.version_label
"""

_GET_VERSION_PLAN_SQL = """
    SELECT p.version_label,
           p.kind,
           p.status,
           COALESCE(p.approved_at,
                    CASE WHEN p.kind = 'baseline' THEN p.effective_from END
           )::text AS activated_at,
           p.approved_by::text,
           p.eval_run_id,
           p.created_at::text,
           p.plan_code,
           p.plan_id::text
      FROM prompt_override_plan p
     WHERE p.scope = 'platform'
       AND p.agent_id = %(agent_id)s
       AND p.version_label = %(version_label)s
     ORDER BY p.created_at DESC
     LIMIT 1
"""

_GET_VERSION_SEGMENTS_SQL = """
    SELECT s.segment_id,
           s.body,
           s.locked,
           s.position,
           s.content_hash,
           s.note,
           s.version
      FROM prompt_override_segment s
     WHERE s.plan_id = %(plan_id)s::uuid
     ORDER BY s.position, s.segment_id
"""


class PostgresPromptOverrideRepository:
    """psycopg ``AsyncConnection``-backed ``PromptOverrideRepository``.

    Wraps a connection pointed at ``chora_ai_kernel`` (where the 0007 tables
    live). Construct one per request scope, or share a long-lived connection as
    the outbox store does — each call opens its own short read transaction.
    """

    def __init__(self, *, conn: Any) -> None:
        if conn is None:
            raise ValueError("PostgresPromptOverrideRepository requires a connection")
        self._conn = conn

    async def fetch_active_overrides(self, *, tenant_id: str, agent_id: str) -> list[ScopeOverride]:
        agent = (agent_id or "").strip()
        if not agent:
            raise ValueError("PostgresPromptOverrideRepository.fetch_active_overrides requires a non-empty agent_id")
        tenant = (tenant_id or "").strip()

        # transaction() wraps cursor() so the tx-local set_config GUC holds for
        # the SELECT regardless of the connection's autocommit mode.
        async with self._conn.transaction(), self._conn.cursor() as cur:
            await cur.execute(_SET_TENANT_SQL, (tenant,))
            await cur.execute(_QUERY, {"agent_id": agent, "tenant_id": tenant})
            rows = await cur.fetchall()

        # Group rows into one ScopeOverride per scope. All rows of a scope share
        # the same active plan (one active override plan per scope+agent), so
        # the plan's display version_label (plan_id fallback) is the stable
        # version token the stamps + O+ surfaces show.
        grouped: dict[str, dict[str, Any]] = {}
        for scope, version, segment_id, body in rows:
            entry = grouped.setdefault(scope, {"version": version, "segments": {}})
            entry["segments"][segment_id] = body

        return [
            ScopeOverride(
                scope=scope,
                version=entry["version"],
                segments=entry["segments"],
            )
            for scope, entry in grouped.items()
        ]

    async def get_plan(self, plan_id: str) -> PromptPlanRecord | None:
        """Read a plan's scope/tenant/status (ADR-197 M-C.2). Returns ``None``
        when the plan does not exist.

        Runs under the 0007 policy's admin branch (plan_id-only, no tenant) —
        same authorization rung the other plan_id-only write paths use. RLS stays
        ENABLED; the bind reverts at transaction end (tx-local).
        """
        if not (plan_id or "").strip():
            raise ValueError("get_plan requires a plan_id")
        async with self._conn.transaction(), self._conn.cursor() as cur:
            await cur.execute(_SET_ROLE_ADMIN_SQL)
            await cur.execute(_GET_PLAN_SQL, {"plan_id": plan_id})
            row = await cur.fetchone()
        if row is None:
            return None
        plan_id_v, scope, tenant_id, status, agent_id, version_label = row
        return PromptPlanRecord(
            plan_id=plan_id_v,
            scope=scope,
            tenant_id=tenant_id,
            status=status,
            agent_id=agent_id,
            version_label=version_label,
        )

    # ---- Promotion write paths (ADR-197 M-C.1) -----------------------------

    async def create_draft_plan(
        self,
        *,
        plan_code: str,
        scope: str,
        tenant_id: str | None,
        created_by: str,
        agent_id: str | None = None,
        version_label: str | None = None,
    ) -> str:
        """Insert a new ``draft`` plan; return its app-minted UUIDv7 plan_id.

        Tenant-scoped on the RLS tenant rung: a ``tenant`` plan's tenant_id must
        match ``chora.tenant_id`` (the 0007 policy's WITH CHECK); a ``platform``
        plan satisfies the policy's scope branch. So we bind the tenant GUC.
        """
        code = (plan_code or "").strip()
        if not code:
            raise ValueError("create_draft_plan requires a non-empty plan_code")
        if scope not in VALID_SCOPES:
            raise ValueError(f"create_draft_plan: unknown scope {scope!r} (expected one of {sorted(VALID_SCOPES)})")
        if scope == "platform":
            tenant_id = None  # platform plans never carry a tenant_id
        elif not (tenant_id or "").strip():
            raise ValueError("create_draft_plan: tenant scope requires a tenant_id")

        plan_id = str(_uuid_utils.uuid7())
        params = {
            "plan_id": plan_id,
            "plan_code": code,
            "scope": scope,
            "tenant_id": tenant_id,
            "created_by": created_by,
            "agent_id": (agent_id or None),
            "version_label": (version_label or None),
        }
        async with self._conn.transaction(), self._conn.cursor() as cur:
            await cur.execute(_SET_TENANT_SQL, ((tenant_id or ""),))
            await cur.execute(_INSERT_PLAN_SQL, params)
        logger.info(
            "prompt_override_plan.draft_created",
            extra={"plan_id": plan_id, "scope": scope, "plan_code": code},
        )
        return plan_id

    async def add_segment(
        self,
        *,
        plan_id: str,
        agent_id: str,
        segment_id: str,
        body: str,
        content_hash: str | None = None,
        note: str = "",
    ) -> str:
        """Append an override segment; return its UUIDv7 segment_row_id.

        ``content_hash`` defaults to ``sha256(body)`` hex (the audit / cache key
        per the 0007 schema). Runs under the admin authorization branch since
        the call carries no tenant_id (see ``_SET_ROLE_ADMIN_SQL``).
        """
        text = body or ""
        if not text.strip():
            raise ValueError("add_segment requires a non-empty body")
        if not (plan_id or "").strip():
            raise ValueError("add_segment requires a plan_id")
        digest = content_hash or hashlib.sha256(text.encode("utf-8")).hexdigest()

        segment_row_id = str(_uuid_utils.uuid7())
        params = {
            "segment_row_id": segment_row_id,
            "plan_id": plan_id,
            "agent_id": agent_id,
            "segment_id": segment_id,
            "body": text,
            "content_hash": digest,
            "note": note,
        }
        async with self._conn.transaction(), self._conn.cursor() as cur:
            await cur.execute(_SET_ROLE_ADMIN_SQL)
            await cur.execute(_INSERT_SEGMENT_SQL, params)
        logger.info(
            "prompt_override_segment.added",
            extra={
                "segment_row_id": segment_row_id,
                "plan_id": plan_id,
                "agent_id": agent_id,
                "segment_id": segment_id,
            },
        )
        return segment_row_id

    async def update_plan_status(
        self,
        *,
        plan_id: str,
        expected_from: PromptPlanState,
        to: PromptPlanState,
        approved_by: str | None = None,
        eval_run_id: str | None = None,
    ) -> None:
        """Guarded status transition. Fail-loud if no row is in ``expected_from``.

        The legal edge is validated by the caller (the transition service's
        ``assert_transition``); this DB guard is the concurrency / staleness
        backstop — 0 rows updated means the plan moved underneath us or never
        existed.
        """
        params = {
            "plan_id": plan_id,
            "to": str(to),
            "expected_from": str(expected_from),
            "approved_by": approved_by,
            "eval_run_id": eval_run_id,
        }
        async with self._conn.transaction(), self._conn.cursor() as cur:
            await cur.execute(_SET_ROLE_ADMIN_SQL)
            await cur.execute(_UPDATE_STATUS_SQL, params)
            updated = cur.rowcount
        if updated == 0:
            # StalePromptTransitionError subclasses RuntimeError — existing
            # callers/tests catching RuntimeError still observe it; the M-C.2
            # HITL-approval handler catches THIS specifically to treat a
            # re-delivery as an idempotent no-op.
            raise StalePromptTransitionError(
                f"update_plan_status: 0 rows updated for plan {plan_id} — "
                f"expected_from={expected_from} did not match the current "
                f"status (illegal or stale transition to {to})"
            )

    async def activate_plan(
        self,
        *,
        plan_id: str,
        scope: str,
        tenant_id: str | None,
        approved_by: str | None = None,
        agent_id: str | None = None,
    ) -> None:
        """Atomically archive the prior active override for the scope/tenant AND
        agent, then promote ``plan_id`` to ``active`` - both UPDATEs in ONE
        transaction.

        Relies on the 0009 partial unique index (one active override per
        scope+agent): archiving the incumbent before activating keeps the index
        satisfied. Baselines are never archived (kind='override' filter), and a
        NULL ``agent_id`` only ever archives legacy agent-less plans (IS NOT
        DISTINCT FROM). If a concurrent activation races, the index surfaces
        the conflict as an error (not silently swallowed). Fail-loud if
        ``plan_id`` is not in ``pending_hitl``.
        """
        if scope == "platform":
            tenant_id = None
        async with self._conn.transaction(), self._conn.cursor() as cur:
            await cur.execute(_SET_TENANT_SQL, ((tenant_id or ""),))
            await cur.execute(
                _ARCHIVE_PRIOR_ACTIVE_SQL,
                {
                    "plan_id": plan_id,
                    "scope": scope,
                    "tenant_id": tenant_id,
                    "agent_id": (agent_id or None),
                },
            )
            await cur.execute(_ACTIVATE_SQL, {"plan_id": plan_id, "approved_by": approved_by})
            activated = cur.rowcount
        if activated == 0:
            raise StalePromptTransitionError(
                f"activate_plan: 0 rows updated for plan {plan_id} — it is not "
                f"in pending_hitl (stale or illegal activation)"
            )
        logger.info(
            "prompt_override_plan.activated",
            extra={"plan_id": plan_id, "scope": scope, "approved_by": approved_by},
        )

    # ---- Catalogue reads (CHO-2368) ----------------------------------------

    @staticmethod
    def _version_from_row(row: tuple[Any, ...], *, agent_id: str) -> CataloguePlanVersion:
        (
            version_label,
            kind,
            status,
            activated_at,
            approved_by,
            eval_run_id,
            created_at,
            plan_code,
            plan_id,
        ) = row
        return CataloguePlanVersion(
            plan_id=plan_id,
            plan_code=plan_code,
            agent_id=agent_id,
            version_label=version_label,
            kind=kind,
            status=status,
            created_at=created_at,
            activated_at=activated_at,
            approved_by=approved_by,
            eval_run_id=eval_run_id,
        )

    async def list_agent_versions(self, *, agent_id: str) -> list[CataloguePlanVersion]:
        """All PLATFORM catalogue versions for one agent, oldest first.

        Covers both kinds (seeded baselines + gated overrides). Platform rows
        pass the 0007 RLS policies' scope branch, so only the defensive empty
        tenant GUC is bound (no admin rung on the read path).
        """
        agent = (agent_id or "").strip()
        if not agent:
            raise ValueError("list_agent_versions requires a non-empty agent_id")
        async with self._conn.transaction(), self._conn.cursor() as cur:
            await cur.execute(_SET_TENANT_SQL, ("",))
            await cur.execute(_LIST_VERSIONS_SQL, {"agent_id": agent})
            rows = await cur.fetchall()
        return [self._version_from_row(row, agent_id=agent) for row in rows]

    async def get_agent_version(
        self, *, agent_id: str, version_label: str
    ) -> tuple[CataloguePlanVersion, list[CatalogueSegmentView]] | None:
        """One catalogue version + its segments in composition order, or None."""
        agent = (agent_id or "").strip()
        label = (version_label or "").strip()
        if not agent or not label:
            raise ValueError("get_agent_version requires agent_id and version_label")
        async with self._conn.transaction(), self._conn.cursor() as cur:
            await cur.execute(_SET_TENANT_SQL, ("",))
            await cur.execute(_GET_VERSION_PLAN_SQL, {"agent_id": agent, "version_label": label})
            plan_row = await cur.fetchone()
            if plan_row is None:
                return None
            plan = self._version_from_row(plan_row, agent_id=agent)
            await cur.execute(_GET_VERSION_SEGMENTS_SQL, {"plan_id": plan.plan_id})
            seg_rows = await cur.fetchall()
        segments = [
            CatalogueSegmentView(
                segment_id=segment_id,
                body=body,
                locked=bool(locked),
                position=position,
                content_hash=content_hash,
                note=note,
                version=version,
            )
            for segment_id, body, locked, position, content_hash, note, version in seg_rows
        ]
        return plan, segments


__all__ = ["PostgresPromptOverrideRepository"]
