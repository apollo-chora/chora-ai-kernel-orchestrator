-- =============================================================================
-- chora-ai-kernel-orchestrator : 0059_kernel_rls_fail_closed.up.sql
--
-- APPLIED 2026-08-23T15:59:21Z. Verified by two instruments: this filename in
--     chora_runner_schema_migrations, and a direct pg_policies read showing
--     has_sweeper_arm = t and still_fail_open = f on BOTH tables.
--
--     ⚠ IT DID NOT APPLY VIA CI. (The CI path was UNPROVEN when this was
--     written; it has since been proven — build
--     813e0c46-fadf-4685-a51a-f79474c78de4 at 2026-08-23T16:12Z ran
--     detect-and-stage SUCCESS, its first pass in a month, then apply SUCCESS,
--     and put consumption 0114 and sharing 0052-0056 genuinely into their
--     databases, confirmed by direct read. That build's overall status is
--     FAILURE because a LATER step, `verify`, failed — read the STEP statuses,
--     never the build colour. 0059 itself was re-staged and re-applied by that
--     run as a tracker no-op, and the live discriminator was re-measured
--     afterwards: unset GUC still 0, sweeper=on still 41.) The only
--     migrations build, c89a6acc, FAILED at its `detect-and-stage` step (on an
--     unrelated un-allowlisted destructive file in chora-sharing, since cleared
--     by 344cdf78c). What actually applied this was a manual sequence: the
--     kernel migrations directory was hand-staged into GCS at 15:54:48Z and the
--     Cloud Run job `chora-dev-apply-familiar-migrations` was run at 15:57:16Z.
--     ⚠ Note the job name: the CI lane's `_APPLY_JOB` is
--     `chora-dev-apply-familiar-migrations`, NOT `chora-dev-migrations-apply`.
--     Two lanes, two jobs — resolve the job from the LANE's config, never from
--     whichever name reads more plausibly.
--
--     ⚠⚠ AND THE MECHANISM THAT MATTERS FOR ANYONE APPLYING A MIGRATION HERE:
--     THE APPLY JOB APPLIES WHAT IS STAGED IN GCS, NOT WHAT IS IN GIT. Staging
--     (a `gsutil rsync` of the whole migrations tree) happens in the build's
--     `detect-and-stage` step; the apply is a LATER step reading
--     GCS_PREFIX=migrations/. Firing the job without a successful stage applies
--     a STALE SNAPSHOT and still reports SUCCESS. A targeted stage is as
--     dangerous as a targeted apply: the manual run above staged only THIS
--     service's directory, which is why other services' migrations sat unstaged
--     afterwards while the database, git and the staging bucket all disagreed.
--
-- VERIFIED AFTER APPLY, as owner with FORCE RLS enabled in-tx (reproducing the
--     NOBYPASSRLS non-owner view), rolled back. The negative legs flipped, which
--     is the evidence — the positive legs pass under EITHER policy and prove
--     nothing on their own:
--         no GUC                    41 rows   -> 0
--         reaper predicate, no GUC   1        -> 0
--         mark_reaped, no GUC        UPDATE 1 -> UPDATE 0
--         sweeper=on                41 / 2 tenants (unchanged, correct)
--         settled_by moved to 'reaped' / 'reaper:request_expired' on a seeded,
--         rolled-back past-deadline row (production has 0 parked rows, so the
--         reaper could not be exercised on live data without seeding).
--
-- This file was held in migrations/staged/ (which the
--     runner's NON-RECURSIVE `services/<svc>/migrations/*.sql` glob cannot see)
--     until its ordering precondition was met. Both halves of that precondition
--     were verified before the `git mv`, and they are recorded here because the
--     next reader needs to know what made it safe, not that it once waited:
--
--     1. THE BINARY THAT SETS THE GUC IS SERVING. chora-ai-kernel-orchestrator
--        at sha256:d4f230e39a93a21, SPEC == RUNNING read BY CONTAINER NAME (not
--        off a set-image exit code), ready 1/1, ReplicaFailure absent. That
--        image carries `chora.kernel_sweeper` in three modules.
--        ⚠ A METHOD-LEVEL GREP FOR THE GUC IS A FALSE NEGATIVE: it is applied
--        as connection `session_settings` and re-applied by
--        ReconnectingAsyncConnection on every reconnect, so `fetch_expired`,
--        `mark_reaped` and `count_parked` contain no reference to it. The opt-in
--        is per CONNECTION, not per statement.
--     2. THE LOCK THAT WOULD HAVE BLOCKED THIS MIGRATION IS GONE. The boot
--        sweep's unreleased transaction held AccessShareLock on
--        ai_assist_inflight_jobs for the life of the pod, which is incompatible
--        with the ACCESS EXCLUSIVE that the DROP/CREATE POLICY below takes; a
--        dry run hung on exactly that and had to be killed. Fixed in the same
--        image; verified idle_in_transaction = 0 database-wide.
--
-- ADR           : ADR-184 shape (intra-service machinery on chora_ai_kernel),
--                 ADR-192 GUC pattern (a named opt-in that contributes nothing
--                 when unset). Amends the policies shipped by 0055 and 0056.
-- Domain        : AI Kernel (supporting/platform)
-- Database      : chora_ai_kernel
-- Date          : 2026-08-23
--
-- WHAT WAS WRONG
--   0055 and 0056 both shipped:
--     USING (NULLIF(current_setting('chora.tenant_id', TRUE), '') IS NULL
--            OR tenant_id = NULLIF(...)::uuid)
--   An UNSET GUC MATCHES EVERY TENANT. Consumption's companion_turns is the
--   opposite, fail-CLOSED at 0 rows, so the two domains were failing in
--   OPPOSITE directions and a forgotten set_config on the kernel path read
--   cross-tenant while looking exactly like working code.
--
--   Measured on the live database 2026-08-23 as chora_ai_kernel_app_rw
--   (NOBYPASSRLS, and NOT the table owner — owner is chora_ai_kernel_migrate,
--   relforcerowsecurity = false — so the policy genuinely applies to it and the
--   reading is not the owner-read false zero):
--     * GUC unset  -> 41 park rows visible, spanning 2 DISTINCT TENANTS
--     * GUC = a different valid tenant UUID -> 0 rows (the tenant arm is fine)
--     * GUC = '' -> 41 rows (the other fail-open entry)
--   So the tenant arm was always correct; only the DEFAULT was wrong.
--
-- WHY THIS IS NOT A FOURTH ADR-165/184/192 BYPASS
--   It TIGHTENS. Reach before = {ALL tenants by default, own tenant}. Reach
--   after = {NONE by default, own tenant, ALL by explicit opt-in}. The maximum
--   reach is UNCHANGED and the default goes from maximally open to closed, so
--   this REDUCES the surface. It takes an EXISTING, always-on, unsanctioned
--   cross-tenant surface and brings it under a shape the ADR chain already
--   ratified. A fourth *widening* surface would need its own ADR; closing a
--   hole does not.
--
--   Shape cited is ADR-184 (the static PERMISSIVE for intra-service machinery
--   on THIS database), NOT ADR-192: ADR-192 widens *by data*, bounded by the
--   franchise_satellite mapping, and the kernel sweeper needs UNBOUNDED
--   all-tenant reach, which that shape cannot express. This opt-in GUC is
--   strictly STRICTER than ADR-184's always-on policy.
--
--   ⚠ The 0055/0056 headers justified the fail-open arm as "the 0003_outbox
--   SWEEPER MODE precedent". That citation does not hold. 0003_outbox.sql keys
--   its fail-open arm on `app.current_tenant` — a DIFFERENT GUC, so a session
--   setting chora.tenant_id never touches it — and describes itself as POC
--   scaffolding with a compensating control at the app + envelope layer per
--   ADR-141. The other citation, 0007, licenses the NULLIF-safe CAST and its
--   own tenant arm is fail-CLOSED. A POC concession on one GUC had been
--   promoted into a named convention on the canonical one.
--
-- ORDERING — WHY THIS FILE IS STAGED AND NOT SHIPPED WITH ITS CODE
--   Migrations land BEFORE binaries and auto-apply on push. ALL FIFTEEN reader
--   paths over these two tables run with NO tenant GUC:
--     parks (10): queue_park INSERT, get, mark_completed, mark_reaped,
--       record_late_completion, fetch_expired (the reaper age scan),
--       fetch_outbox_dead_lettered, count_parked, thread_is_parked,
--       backfill_from_outbox
--     inflight (5): register INSERT, delete/delete_in_tx, mark_resumed, get,
--       sweep (the boot resume)
--   Against a binary that does NOT set chora.kernel_sweeper this policy gives
--   2 HARD ERRORS (the INSERTs, rejected by WITH CHECK) and 13 SILENT ZEROS.
--   A reaper that reaps nothing TESTS GREEN; a delete that removes nothing
--   leaves the job "unfinished" forever and the boot sweep resumes it forever.
--   Hence: code first (a no-op under the old fail-open policy), then this.
--
-- ⚠⚠ ROLLBACK IS NOT SYMMETRIC — STATED HERE RATHER THAN LEFT FOR AN INCIDENT.
--   Once this is applied, rolling the kernel binary BACK below the digest that
--   sets chora.kernel_sweeper reintroduces exactly the 2-errors-13-zeros state
--   above. The constraint is deliberately NOT scoped to tolerate that, because
--   a policy that tolerates the old binary is the fail-open policy we are
--   removing. THEREFORE: a rollback of the kernel binary past that digest MUST
--   run 0059_kernel_rls_fail_closed.down.sql FIRST. There is no version of this
--   change that is both safe against the old binary and actually closes the
--   hole; the ordering is the mitigation.
--
--   Idempotent: DROP POLICY IF EXISTS + CREATE, so a re-apply is safe. Grants
--   ride 9999_grant_app_roles.sql (run the FULL migrations job, never targeted:
--   a targeted run that skips 9999 leaves app_rw at 42501).
-- =============================================================================

BEGIN;

-- --- ai_assist_inflight_jobs (0055) -----------------------------------------
DROP POLICY IF EXISTS ai_assist_inflight_tenant_isolation ON ai_assist_inflight_jobs;

CREATE POLICY ai_assist_inflight_tenant_isolation ON ai_assist_inflight_jobs
    USING (
        -- Ordinary tenant-scoped session: own tenant only. An UNSET GUC makes
        -- this NULL (not TRUE), which is the fail-CLOSED default.
        tenant_id = NULLIF(current_setting('chora.tenant_id', TRUE), '')::uuid
        -- Intra-service sweeper, EXPLICIT opt-in. Unset contributes nothing.
        OR NULLIF(current_setting('chora.kernel_sweeper', TRUE), '') = 'on'
    )
    WITH CHECK (
        tenant_id = NULLIF(current_setting('chora.tenant_id', TRUE), '')::uuid
        OR NULLIF(current_setting('chora.kernel_sweeper', TRUE), '') = 'on'
    );

-- --- ai_kernel_agent_dispatch_parks (0056) ----------------------------------
DROP POLICY IF EXISTS ai_kernel_agent_dispatch_parks_tenant_isolation
    ON ai_kernel_agent_dispatch_parks;

CREATE POLICY ai_kernel_agent_dispatch_parks_tenant_isolation
    ON ai_kernel_agent_dispatch_parks
    USING (
        tenant_id = NULLIF(current_setting('chora.tenant_id', TRUE), '')::uuid
        OR NULLIF(current_setting('chora.kernel_sweeper', TRUE), '') = 'on'
    )
    WITH CHECK (
        tenant_id = NULLIF(current_setting('chora.tenant_id', TRUE), '')::uuid
        OR NULLIF(current_setting('chora.kernel_sweeper', TRUE), '') = 'on'
    );

COMMENT ON TABLE ai_kernel_agent_dispatch_parks IS
    'Park ledger (ADR-254 D5). RLS fail-CLOSED on an unset chora.tenant_id; '
    'intra-service sweepers opt in explicitly via chora.kernel_sweeper=on '
    '(ADR-184 shape, ADR-192 GUC pattern). Rolling the kernel binary back past '
    'the digest that sets that GUC requires 0059''s DOWN first.';

COMMENT ON TABLE ai_assist_inflight_jobs IS
    'In-flight assist registry (ADR-251 D4). RLS fail-CLOSED on an unset '
    'chora.tenant_id; intra-service sweepers opt in explicitly via '
    'chora.kernel_sweeper=on (ADR-184 shape, ADR-192 GUC pattern).';

COMMIT;
