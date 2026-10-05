"""ManaPort adapters for the Growth-Edge crew (ADR-205 D6).

``FreeTierMana`` is the BASE-loop adapter: auto-derived edges (graded-assessment
/ Ebbinghaus) and the closed-loop owner build run free, so there is no
reservation to settle or refund — both methods are safe no-ops. WS-4 introduces
``PaidMana`` (reserve-on-accept → refund-on-fail via PricePlanResolver, ADR-178)
for the upload-driven premium path; the crew graph's ManaPort is swapped at the
composition root with no graph change (it already refunds on every fail-loud /
BLOCK node).
"""

from __future__ import annotations

import logging
from typing import Protocol

logger = logging.getLogger(__name__)


class ManaReservationError(RuntimeError):
    """The paid-path mana reservation is missing or no longer active."""


class Wallet(Protocol):
    """The mana wallet (chora-identity/-payments gRPC in prod). The upload door
    RESERVES on accept; the crew verifies + releases (refund) here."""

    async def is_reservation_active(self, *, reservation_id: str, tenant_id: str, gcid: str) -> bool: ...
    async def release_reservation(self, *, reservation_id: str, tenant_id: str, gcid: str, reason: str) -> None: ...


class FreeTierMana:
    """ManaPort for the free base loop — no reservation lifecycle."""

    async def ensure_reserved(self, *, reservation_id: str, tenant_id: str, gcid: str) -> None:
        # Free tier carries no reservation_id; a stray one is logged (WS-4 paid
        # path owns real reservations) but never gates the run.
        if reservation_id:
            logger.info(
                "weakness_mana.free_tier.ignoring_reservation",
                extra={"reservation_id": reservation_id, "tenant_id": tenant_id},
            )

    async def refund(self, *, reservation_id: str, tenant_id: str, gcid: str, reason: str) -> None:
        if reservation_id:
            logger.info(
                "weakness_mana.free_tier.refund_noop", extra={"reservation_id": reservation_id, "reason": reason}
            )


class PaidMana:
    """ManaPort for the upload-driven PREMIUM path (ADR-205 D6 / ADR-178).

    The reservation is taken at the upload HTTP door (chora-consumption) on
    accept; the crew verifies it is still active before burning, and refunds it
    on any fail-loud / BLOCK node. Fail-loud: a missing or inactive reservation
    REFUSES the run (never silently free). Refund is idempotent + safe on an
    already-released reservation (the wallet release is the source of truth).
    """

    def __init__(self, *, wallet: Wallet) -> None:
        self._wallet = wallet

    async def ensure_reserved(self, *, reservation_id: str, tenant_id: str, gcid: str) -> None:
        if not (reservation_id or "").strip():
            raise ManaReservationError(
                "paid weakness analysis requires a mana reservation (taken at the upload door); none present"
            )
        active = await self._wallet.is_reservation_active(reservation_id=reservation_id, tenant_id=tenant_id, gcid=gcid)
        if not active:
            raise ManaReservationError(
                f"mana reservation {reservation_id} is not active (cancelled/expired/consumed); refusing run"
            )

    async def refund(self, *, reservation_id: str, tenant_id: str, gcid: str, reason: str) -> None:
        if not (reservation_id or "").strip():
            return
        await self._wallet.release_reservation(
            reservation_id=reservation_id, tenant_id=tenant_id, gcid=gcid, reason=reason
        )


__all__ = ["FreeTierMana", "ManaReservationError", "PaidMana", "Wallet"]
