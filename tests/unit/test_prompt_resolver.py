"""Unit tests for ``PromptResolver`` — the pure precedence/merge logic.

ADR-197 M-B.1. Resolver is exercised with an in-memory fake repository so the
precedence ladder is tested without a DB. Pins the PINNED contract M-B.2 + the
Go composers depend on:

* tenant override wins over platform override,
* platform override when no tenant override,
* embedded default when neither (segments={}, version=None, source="embedded"),
* WHOLE-ACTIVE-PLAN-WINS PER SCOPE (no cross-scope segment merge),
* fail-loud on malformed input / data-integrity violations.
"""

from __future__ import annotations

import pytest

from chora_ai_kernel_orchestrator.domain.prompt_registry import (
    PromptResolver,
    Resolved,
    ScopeOverride,
)


class _FakeRepo:
    """In-memory ``PromptOverrideRepository`` returning a fixed scope list."""

    def __init__(self, overrides: list[ScopeOverride]) -> None:
        self._overrides = overrides
        self.calls: list[tuple[str, str]] = []

    async def fetch_active_overrides(self, *, tenant_id: str, agent_id: str) -> list[ScopeOverride]:
        self.calls.append((tenant_id, agent_id))
        return list(self._overrides)


# ---------------------------------------------------------------------------
# Precedence
# ---------------------------------------------------------------------------


async def test_tenant_override_wins_over_platform() -> None:
    repo = _FakeRepo(
        [
            ScopeOverride(scope="platform", version="plan-platform", segments={"role": "PLAT"}),
            ScopeOverride(scope="tenant", version="plan-tenant", segments={"role": "TEN"}),
        ]
    )
    resolver = PromptResolver(repo)

    out = await resolver.resolve("tenant-1", "qgen-question")

    assert out == Resolved(segments={"role": "TEN"}, version="plan-tenant", source="tenant_override")


async def test_platform_override_when_no_tenant() -> None:
    repo = _FakeRepo(
        [
            ScopeOverride(scope="platform", version="plan-platform", segments={"task": "PLAT"}),
        ]
    )
    resolver = PromptResolver(repo)

    out = await resolver.resolve("tenant-1", "qgen-question")

    assert out == Resolved(
        segments={"task": "PLAT"},
        version="plan-platform",
        source="platform_override",
    )


async def test_embedded_when_no_override() -> None:
    repo = _FakeRepo([])
    resolver = PromptResolver(repo)

    out = await resolver.resolve("tenant-1", "qgen-question")

    assert out == Resolved(segments={}, version=None, source="embedded")
    assert out.segments == {}
    assert out.version is None
    assert out.source == "embedded"


async def test_whole_active_plan_wins_per_scope_no_cross_scope_merge() -> None:
    """Tenant overrides only [role]; platform overrides [role]+[task]. The tenant
    plan wins AS A WHOLE — platform's [task] is NOT merged in (the missing
    segment falls back to the embedded default in the Go composer)."""
    repo = _FakeRepo(
        [
            ScopeOverride(
                scope="platform",
                version="plan-platform",
                segments={"role": "PLAT_ROLE", "task": "PLAT_TASK"},
            ),
            ScopeOverride(
                scope="tenant",
                version="plan-tenant",
                segments={"role": "TEN_ROLE"},
            ),
        ]
    )
    resolver = PromptResolver(repo)

    out = await resolver.resolve("tenant-1", "qgen-question")

    assert out.source == "tenant_override"
    assert out.version == "plan-tenant"
    assert out.segments == {"role": "TEN_ROLE"}  # NO 'task' key from platform


async def test_empty_tenant_scope_falls_through_to_platform() -> None:
    """A tenant ScopeOverride with no segments for this agent does not win — the
    resolver falls through to the platform override."""
    repo = _FakeRepo(
        [
            ScopeOverride(scope="tenant", version="plan-tenant", segments={}),
            ScopeOverride(
                scope="platform",
                version="plan-platform",
                segments={"examples": "PLAT_EX"},
            ),
        ]
    )
    resolver = PromptResolver(repo)

    out = await resolver.resolve("tenant-1", "qgen-question")

    assert out == Resolved(
        segments={"examples": "PLAT_EX"},
        version="plan-platform",
        source="platform_override",
    )


async def test_empty_scopes_everywhere_yields_embedded() -> None:
    repo = _FakeRepo(
        [
            ScopeOverride(scope="tenant", version="t", segments={}),
            ScopeOverride(scope="platform", version="p", segments={}),
        ]
    )
    resolver = PromptResolver(repo)

    out = await resolver.resolve("tenant-1", "qgen-question")

    assert out == Resolved(segments={}, version=None, source="embedded")


async def test_resolve_returns_a_defensive_copy_of_segments() -> None:
    """Mutating the resolved segments must not corrupt the repo's cached map."""
    inner = {"role": "TEN"}
    repo = _FakeRepo([ScopeOverride(scope="tenant", version="plan-tenant", segments=inner)])
    resolver = PromptResolver(repo)

    out = await resolver.resolve("tenant-1", "qgen-question")
    out.segments["role"] = "MUTATED"

    assert inner == {"role": "TEN"}


async def test_resolver_passes_trimmed_keys_to_repo() -> None:
    repo = _FakeRepo([])
    resolver = PromptResolver(repo)

    await resolver.resolve("  tenant-1  ", "  qgen-question  ")

    assert repo.calls == [("tenant-1", "qgen-question")]


# ---------------------------------------------------------------------------
# Fail-loud
# ---------------------------------------------------------------------------


async def test_empty_agent_id_raises() -> None:
    resolver = PromptResolver(_FakeRepo([]))
    with pytest.raises(ValueError, match="non-empty agent_id"):
        await resolver.resolve("tenant-1", "   ")


async def test_unknown_scope_raises() -> None:
    repo = _FakeRepo([ScopeOverride(scope="galaxy", version="x", segments={"role": "X"})])
    resolver = PromptResolver(repo)
    with pytest.raises(ValueError, match="unknown override scope"):
        await resolver.resolve("tenant-1", "qgen-question")


async def test_two_active_overrides_same_scope_raises() -> None:
    repo = _FakeRepo(
        [
            ScopeOverride(scope="tenant", version="a", segments={"role": "A"}),
            ScopeOverride(scope="tenant", version="b", segments={"role": "B"}),
        ]
    )
    resolver = PromptResolver(repo)
    with pytest.raises(ValueError, match="more than one active"):
        await resolver.resolve("tenant-1", "qgen-question")


def test_none_repository_raises() -> None:
    with pytest.raises(ValueError, match="requires a PromptOverrideRepository"):
        PromptResolver(None)  # type: ignore[arg-type]
