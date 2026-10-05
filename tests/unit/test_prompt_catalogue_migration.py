"""Migration 0009 content pin (CHO-2368, ADR-197 catalogue carve-out).

Two jobs, mirroring the chora-consumption 0106 seedspec discipline:

1. The GENERATED BASELINE SEED block inside 0009 must byte-match (modulo
   whitespace runs) the output of ``baseline_seedspec.baseline_seed_sql()``,
   which slices the live repo prompt sources. Sources moving without a seed
   regen, or the seed edited by hand, both go RED.
2. The hand-authored DDL half must carry the load-bearing carve-out: the
   ``kind`` column + CHECK, the per-agent re-scoped active-override partial
   uniques (baselines excluded), the baseline uniqueness, and the segment
   ``locked`` + ``position`` columns. Without the carve-out a v1 baseline row
   would occupy the one-active-platform slot every override needs.
"""

from __future__ import annotations

import re
from pathlib import Path

from chora_ai_kernel_orchestrator.domain.prompt_registry import baseline_seedspec as spec

_SERVICE_ROOT = Path(__file__).resolve().parents[2]
_UP = _SERVICE_ROOT / "migrations" / "0009_prompt_catalogue_baselines.up.sql"
_DOWN = _SERVICE_ROOT / "migrations" / "0009_prompt_catalogue_baselines.down.sql"

_BEGIN = "-- BEGIN GENERATED BASELINE SEED"
_END = "-- END GENERATED BASELINE SEED"


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _up_sql() -> str:
    return _UP.read_text(encoding="utf-8")


def test_up_migration_exists() -> None:
    assert _UP.is_file(), f"missing {_UP}"
    assert _DOWN.is_file(), f"missing {_DOWN}"


def test_generated_seed_block_matches_seedspec() -> None:
    sql = _up_sql()
    assert _BEGIN in sql and _END in sql, "0009 must carry the generated seed block markers"
    after_marker_line = sql.split(_BEGIN, 1)[1].split("\n", 1)[1]
    block = after_marker_line.split(_END, 1)[0]
    assert _norm(block) == _norm(spec.baseline_seed_sql()), (
        "0009 generated seed block drifted from baseline_seedspec.baseline_seed_sql(); "
        "regenerate the block instead of hand-editing"
    )


def test_ddl_carries_the_kind_carve_out() -> None:
    sql = _up_sql().lower()
    assert "add column if not exists kind" in sql
    assert "kind in ('baseline','override')" in sql.replace('"', "'").replace(", ", ",")
    # Old global one-active uniques must be dropped and re-scoped per agent,
    # override-kind only.
    assert "drop index if exists uq_prompt_plan_active_platform" in sql
    assert "drop index if exists uq_prompt_plan_active_tenant" in sql
    for idx in ("uq_prompt_plan_active_platform", "uq_prompt_plan_active_tenant"):
        m = re.search(rf"create unique index if not exists {idx}[^;]+;", sql)
        assert m, f"missing re-created {idx}"
        body = m.group(0)
        assert "agent_id" in body
        assert "nulls not distinct" in body
        assert "kind = 'override'" in body
    m = re.search(r"create unique index if not exists uq_prompt_plan_baseline[^;]+;", sql)
    assert m, "missing baseline uniqueness index"
    assert "kind = 'baseline'" in m.group(0)


def test_ddl_adds_plan_and_segment_columns() -> None:
    sql = _up_sql().lower()
    assert "add column if not exists agent_id" in sql
    assert "add column if not exists version_label" in sql
    assert "add column if not exists locked" in sql
    assert "add column if not exists position" in sql


def test_up_never_drops_tables_and_seed_is_idempotent() -> None:
    sql = _up_sql().lower()
    assert "drop table" not in sql
    assert sql.count("on conflict") >= 1 + len(spec.baseline_segments())


def test_down_restores_the_0008_shape() -> None:
    sql = _DOWN.read_text(encoding="utf-8").lower()
    assert "drop index if exists uq_prompt_plan_baseline" in sql
    # Down restores the original global one-active-per-scope uniques.
    m = re.search(r"create unique index if not exists uq_prompt_plan_active_platform[^;]+;", sql)
    assert m and "agent_id" not in m.group(0)
    assert "delete from prompt_override_plan" in sql and "kind = 'baseline'" in sql
