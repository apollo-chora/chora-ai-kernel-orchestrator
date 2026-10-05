"""Migration 0010 content pin (CHO-2379, ADR-197 label uniqueness).

0010 does two things the catalogue depends on, and this module pins both
against a hand-edit:

1. a one-shot data relabel of the live duplicate ``(agent_id, version_label)``
   platform override rows. Oldest attempt keeps the base label, later attempts
   climb, and ``plan_code`` follows the new label so the dupe cannot survive on
   the other axis;
2. the partial unique index that makes the dupe impossible from here on.

The relabel must never delete or archive anything: an override plan is an
audited promotion attempt, and the label is display plus stamp identity only
(the resolver still picks by agent + status). Down therefore drops ONLY the
index. Relabels are not reversed - re-introducing the duplicates would put the
superseded plans back behind the catalogue's newest-wins read.

Live behaviour is verified against the applied database (the verification block
at the tail of the up file); this module guards the file content.
"""

from __future__ import annotations

import re
from pathlib import Path

_SERVICE_ROOT = Path(__file__).resolve().parents[2]
_UP = _SERVICE_ROOT / "migrations" / "0010_prompt_version_label_dedup.up.sql"
_DOWN = _SERVICE_ROOT / "migrations" / "0010_prompt_version_label_dedup.down.sql"

_INDEX = "uq_prompt_plan_override_version"


def _up_sql() -> str:
    return _UP.read_text(encoding="utf-8")


def _down_sql() -> str:
    return _DOWN.read_text(encoding="utf-8")


def test_both_migration_files_exist() -> None:
    assert _UP.is_file(), f"missing {_UP}"
    assert _DOWN.is_file(), f"missing {_DOWN}"


def test_up_is_one_transaction() -> None:
    sql = _up_sql().lower()
    # Relabel + index must land together: a half-applied 0010 would leave
    # duplicates with the guard absent, or the guard rejecting nothing.
    assert re.search(r"^begin;", sql, re.MULTILINE)
    assert re.search(r"^commit;", sql, re.MULTILINE)


def test_relabel_is_scoped_to_platform_overrides_with_semver_labels() -> None:
    sql = _up_sql().lower()
    assert "update prompt_override_plan" in sql
    assert "scope = 'platform'" in sql or "scope='platform'" in sql
    assert "kind = 'override'" in sql or "kind='override'" in sql
    # Baselines carry 'v1' and legacy rows carry NULL; both must be skipped, so
    # the selection is regex-gated before any ::int cast can see them.
    assert "version_label ~" in sql
    assert "::int" in sql


def test_relabel_orders_by_created_at_so_the_oldest_keeps_the_base_label() -> None:
    sql = _up_sql().lower()
    m = re.search(r"row_number\(\)\s+over\s*\(([^)]*)\)", sql)
    assert m, "the relabel must rank duplicates deterministically"
    over = m.group(1)
    assert "partition by" in over
    assert "order by" in over and "created_at" in over
    # plan_id breaks a created_at tie so two applies cannot disagree.
    assert "plan_id" in over


def test_plan_code_follows_the_new_label() -> None:
    sql = _up_sql().lower()
    assert "plan_code" in sql, "a stale plan_code re-introduces the dupe by the other axis"
    assert re.search(r"set\s+version_label\s*=", sql)


def test_up_creates_the_partial_unique_index_on_platform_overrides() -> None:
    sql = _up_sql().lower()
    m = re.search(rf"create unique index if not exists {_INDEX}[^;]+;", sql)
    assert m, f"missing {_INDEX}"
    body = m.group(0)
    assert "prompt_override_plan" in body
    assert "agent_id" in body and "version_label" in body
    # Partial: baselines keep their own 0009 index, and tenant-scope overrides
    # are deliberately out of scope.
    assert "kind = 'override'" in body.replace("kind='override'", "kind = 'override'")
    assert "scope = 'platform'" in body.replace("scope='platform'", "scope = 'platform'")


def test_up_never_destroys_data() -> None:
    sql = _up_sql().lower()
    assert "drop table" not in sql
    assert "delete from" not in sql
    assert "truncate" not in sql
    # 0009's baseline uniqueness must survive untouched.
    assert "drop index if exists uq_prompt_plan_baseline" not in sql


def test_down_drops_only_the_new_index() -> None:
    sql = _down_sql().lower()
    assert f"drop index if exists {_INDEX}" in sql
    assert "delete from" not in sql
    assert "drop table" not in sql
    # The relabel is deliberately NOT reversed.
    assert "update prompt_override_plan" not in sql
    for kept in ("uq_prompt_plan_baseline", "uq_prompt_plan_active_platform"):
        assert f"drop index if exists {kept}" not in sql
