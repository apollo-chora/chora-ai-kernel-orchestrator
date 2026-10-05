"""Tests for the process-wide logging bootstrap (CHO-2368).

Why this exists: the service configures NO logging handler, so every app
logger falls back to Python's ``lastResort`` handler - WARNING+ only. Every
INFO breadcrumb (lifespan ``*.started`` / ``*.skipped``, the audit consumer's
``acked`` / ``activate_skipped_status``, inbox dedupe hits) is INVISIBLE in
``kubectl logs``. That silence turned a commit-semantics defect into a
multi-session goose chase - the consumer processed and acked an approval with
zero observable trace. Fail-loud demands the success path be observable.

Contract:

* ``configure_logging()`` attaches ONE stderr StreamHandler at INFO
  (override via ``CHORA_LOG_LEVEL``), format carrying level + logger name.
* Idempotent - a second call must not stack handlers.
* Respectful - a logger that already has handlers (embedded runs, test
  harnesses) is left untouched.

The tests inject a fresh Logger: pytest's own logging plugin re-attaches
capture handlers to the global root around every test phase, so the bare
branch is unobservable on the real root under pytest.
"""

from __future__ import annotations

import logging

import pytest

from chora_ai_kernel_orchestrator.observability.logging_bootstrap import (
    configure_logging,
)


@pytest.fixture()
def bare_root() -> logging.Logger:
    """A fresh, handler-less logger standing in for the process root."""
    root = logging.Logger("test-bootstrap-root")
    assert not root.handlers
    return root


def test_configures_single_stream_handler_at_info(bare_root: logging.Logger, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CHORA_LOG_LEVEL", raising=False)
    configure_logging(bare_root)
    assert len(bare_root.handlers) == 1
    assert isinstance(bare_root.handlers[0], logging.StreamHandler)
    assert bare_root.level == logging.INFO


def test_idempotent_second_call_does_not_stack(bare_root: logging.Logger, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CHORA_LOG_LEVEL", raising=False)
    configure_logging(bare_root)
    configure_logging(bare_root)
    assert len(bare_root.handlers) == 1


def test_respects_chora_log_level_env(bare_root: logging.Logger, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHORA_LOG_LEVEL", "WARNING")
    configure_logging(bare_root)
    assert bare_root.level == logging.WARNING


def test_invalid_level_falls_back_to_info(bare_root: logging.Logger, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHORA_LOG_LEVEL", "chatty")
    configure_logging(bare_root)
    assert bare_root.level == logging.INFO


def test_leaves_preconfigured_root_untouched(bare_root: logging.Logger, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CHORA_LOG_LEVEL", raising=False)
    sentinel = logging.NullHandler()
    bare_root.addHandler(sentinel)
    configure_logging(bare_root)
    assert bare_root.handlers == [sentinel]


def test_default_targets_process_root() -> None:
    """No-arg call resolves the real root (and under pytest, whose capture
    handlers are present, must leave it untouched)."""
    root = logging.getLogger()
    before = root.handlers[:]
    configure_logging()
    assert root.handlers == before
