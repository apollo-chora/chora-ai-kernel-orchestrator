"""The kernel sweeper opt-in GUC (G2) — one name, one place.

``ai_assist_inflight_jobs`` (0055) and ``ai_kernel_agent_dispatch_parks`` (0056)
shipped with a FAIL-OPEN tenant policy: an unset ``chora.tenant_id`` matched
EVERY tenant's rows. Measured on the live database 2026-08-23 as
``chora_ai_kernel_app_rw`` (NOBYPASSRLS, and NOT the table owner, so the policy
genuinely applied): the unset GUC saw 41 park rows spanning 2 DISTINCT TENANTS,
while a different valid tenant UUID correctly saw 0. The tenant arm was right;
only the DEFAULT was wrong, so a forgotten ``set_config`` read cross-tenant and
looked exactly like working code.

The migration that closes it makes the policy fail-CLOSED and adds an explicit
opt-in arm keyed on ``chora.kernel_sweeper``. This TIGHTENS: reach today is
{ALL tenants by default, own tenant}; after, it is {NONE by default, own
tenant, ALL by explicit opt-in}. Maximum reach is unchanged and the default
goes from maximally open to closed, so it is not a fourth ADR-165/184/192
bypass — it brings an EXISTING unsanctioned hole under an ALREADY RATIFIED
shape.

Shape cited is **ADR-184** (the static PERMISSIVE for intra-service machinery on
chora_ai_kernel, the same database), not ADR-192: ADR-192 widens *by data*,
bounded by the ``franchise_satellite`` mapping, and the kernel sweeper needs
UNBOUNDED all-tenant reach, which that shape cannot express. The opt-in GUC is
strictly stricter than ADR-184's always-on policy. The GUC pattern itself
(a named setting, unset contributing nothing) follows ADR-192.

⚠ The two migration headers justified the fail-open arm as "the 0003_outbox
SWEEPER MODE precedent". That citation does not hold: ``0003_outbox.sql`` keys
its fail-open arm on ``app.current_tenant`` — a DIFFERENT GUC — and describes
itself as POC scaffolding with a compensating control at the app + envelope
layer per ADR-141. 0007, the other citation, licenses the NULLIF-safe CAST and
its own tenant arm is fail-CLOSED. A POC concession on one GUC had been
promoted into a named convention on the canonical one.

Setting this GUC on a connection is bounded by which POLICIES reference it: no
other table's policy mentions ``chora.kernel_sweeper``, so a sweeper connection
gains nothing anywhere else.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

#: The opt-in GUC. Unset (the default everywhere else) contributes nothing.
KERNEL_SWEEPER_GUC = "chora.kernel_sweeper"

#: The only value the policy admits. Anything else reads as not-opted-in.
KERNEL_SWEEPER_ON = "on"

#: Pass as ``session_settings=`` to a connection that is sweeper-mode BY DESIGN
#: (the reaper scan, the boot resume sweep, the ledger backfill, the park
#: writer). ``ReconnectingAsyncConnection`` re-applies these on every
#: (re)connect — a transparent reconnect would otherwise drop the opt-in and the
#: sweeper would silently read 0 rows under the fail-closed policy.
SWEEPER_SESSION_SETTINGS: Mapping[str, str] = MappingProxyType({KERNEL_SWEEPER_GUC: KERNEL_SWEEPER_ON})

__all__ = [
    "KERNEL_SWEEPER_GUC",
    "KERNEL_SWEEPER_ON",
    "SWEEPER_SESSION_SETTINGS",
]
