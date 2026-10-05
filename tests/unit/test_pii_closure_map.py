"""RED: the chora_ai_kernel PII_Closure_Map declares the companion_chat sessions.

ADR-254 D6 puts the companion_chat agent's ADK sessions in the
``companion_chat_sessions`` schema of ``chora_ai_kernel`` (migration 0057).
Chat content is learner PII, so the domain's closure map must carry every
table of that schema keyed by (tenant_id, the ADK ``user_id`` = "{tenant_id}:{gcid}"),
in the shape every other domain's ``config/PII_Closure_Map.yaml`` uses and the
closure orchestrator's loader validates (domain, version, fields_to_tokenize,
retention_days_by_jurisdiction, on_creator_closure).
"""

from __future__ import annotations

from pathlib import Path

import yaml

MAP = Path(__file__).resolve().parents[2] / "config" / "PII_Closure_Map.yaml"

# Mirrors chora-closure-orchestrator adapter/pii_map/loader.py (the validator).
VALID_STRATEGIES = {"tombstone_string", "tombstone_email", "hash", "drop", "preserve", "encrypt", "tokenize"}
VALID_ON_CREATOR = {
    "tokenise_authorship_keep_atom",
    "soft_delete_authored_atoms",
    "transfer_authorship_to_tenant",
    "cascade_pseudonymise",
    "preserve_orphan",
}
# The four tables google.golang.org/adk session/database AutoMigrate creates
# (storage_session.go TableName(): sessions, events, app_states, user_states).
ADK_TABLES = {"sessions", "events", "app_states", "user_states"}


def _load() -> dict:
    assert MAP.is_file(), MAP
    data = yaml.safe_load(MAP.read_text())
    assert isinstance(data, dict)
    return data


def test_map_has_the_loader_shape() -> None:
    data = _load()
    assert data["domain"] == "chora_ai_kernel"
    assert str(data["version"])
    assert isinstance(data["fields_to_tokenize"], list) and data["fields_to_tokenize"]
    retention = data["retention_days_by_jurisdiction"]
    assert isinstance(retention, dict) and "default" in retention
    assert all(isinstance(v, int) for v in retention.values())
    on_creator = data["on_creator_closure"]
    assert on_creator["strategy"] in VALID_ON_CREATOR


def test_every_companion_chat_session_table_is_declared_and_keyed_by_gcid() -> None:
    data = _load()
    declared = {entry["table"] for entry in data["fields_to_tokenize"]}
    expected = {f"companion_chat_sessions.{t}" for t in ADK_TABLES}
    assert expected <= declared, expected - declared
    for entry in data["fields_to_tokenize"]:
        if not entry["table"].startswith("companion_chat_sessions."):
            continue
        table = entry["table"].split(".", 1)[1]
        if table != "app_states":
            # sessions, events, user_states: tenant_id (RLS, migration 0058) and the
            # ADK user_id "{tenant_id}:{gcid}" (owner ruling 2026-08-22)
            assert entry.get("keyed_by") == "(tenant_id, user_id = tenant_id:gcid)", entry["table"]
        for col in entry["columns"]:
            assert col["strategy"] in VALID_STRATEGIES, col
            assert isinstance(col.get("value", ""), str)


def test_chat_content_is_dropped_not_preserved() -> None:
    data = _load()
    by_table = {e["table"]: {c["column"]: c["strategy"] for c in e["columns"]} for e in data["fields_to_tokenize"]}
    events = by_table["companion_chat_sessions.events"]
    for column in ("content", "actions", "grounding_metadata", "custom_metadata", "citation_metadata", "error_message"):
        assert events.get(column) == "drop", column
    assert by_table["companion_chat_sessions.sessions"].get("state") == "drop"
    assert by_table["companion_chat_sessions.user_states"].get("state") == "drop"


def test_no_em_dash() -> None:
    assert "\u2014" not in MAP.read_text()
