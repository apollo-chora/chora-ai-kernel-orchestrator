"""WS-4 (ADR-205 / CHO-1956) — paid-path ManaPort: reserve→refund.

The upload door (chora-consumption) reserves mana on accept; the crew's
PaidMana VERIFIES the reservation is still active before burning, and the crew
graph REFUNDS it on any fail-loud / BLOCK node (ADR-205 D6). Fail-loud: a
missing/inactive reservation refuses the run (never silently free); a refund is
idempotent + safe to call on an already-released reservation.
"""

from __future__ import annotations

import asyncio

import pytest

from chora_ai_kernel_orchestrator.adapter.weakness.mana import ManaReservationError, PaidMana


class FakeWallet:
    def __init__(self, *, active: bool = True) -> None:
        self._active = active
        self.released: list[dict[str, str]] = []
        self.checked: list[str] = []

    async def is_reservation_active(self, *, reservation_id: str, tenant_id: str, gcid: str) -> bool:
        self.checked.append(reservation_id)
        return self._active

    async def release_reservation(self, *, reservation_id: str, tenant_id: str, gcid: str, reason: str) -> None:
        self.released.append({"reservation_id": reservation_id, "reason": reason})


def test_ensure_reserved_ok_for_active_reservation() -> None:
    w = FakeWallet(active=True)
    mana = PaidMana(wallet=w)
    asyncio.run(mana.ensure_reserved(reservation_id="rsv-1", tenant_id="t1", gcid="g1"))
    assert w.checked == ["rsv-1"]


def test_ensure_reserved_refuses_inactive_reservation() -> None:
    w = FakeWallet(active=False)
    mana = PaidMana(wallet=w)
    with pytest.raises(ManaReservationError):
        asyncio.run(mana.ensure_reserved(reservation_id="rsv-1", tenant_id="t1", gcid="g1"))


def test_ensure_reserved_refuses_missing_reservation() -> None:
    w = FakeWallet(active=True)
    mana = PaidMana(wallet=w)
    with pytest.raises(ManaReservationError):
        asyncio.run(mana.ensure_reserved(reservation_id="", tenant_id="t1", gcid="g1"))


def test_refund_releases_via_wallet() -> None:
    w = FakeWallet()
    mana = PaidMana(wallet=w)
    asyncio.run(mana.refund(reservation_id="rsv-1", tenant_id="t1", gcid="g1", reason="screen_input blocked"))
    assert w.released == [{"reservation_id": "rsv-1", "reason": "screen_input blocked"}]


def test_refund_empty_reservation_is_noop() -> None:
    w = FakeWallet()
    mana = PaidMana(wallet=w)
    asyncio.run(mana.refund(reservation_id="", tenant_id="t1", gcid="g1", reason="x"))
    assert w.released == []


def test_free_tier_mana_is_a_noop_reserve_and_refund() -> None:
    """The base-loop ManaPort: nothing to reserve, nothing to refund (the
    upload-door reservation is the price authority; WS-4 swaps in PaidMana)."""
    import asyncio

    from chora_ai_kernel_orchestrator.adapter.weakness.mana import FreeTierMana

    mana = FreeTierMana()
    assert asyncio.run(mana.ensure_reserved(reservation_id="r", tenant_id="t", gcid="g")) is None
    assert asyncio.run(mana.refund(reservation_id="r", tenant_id="t", gcid="g", reason="x")) is None
