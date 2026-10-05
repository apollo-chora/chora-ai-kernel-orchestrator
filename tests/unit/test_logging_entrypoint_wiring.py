"""The logging bootstrap is CALLED, and an INFO record actually renders.

``test_logging_bootstrap.py`` proves ``configure_logging()`` is correct. It does
not prove anything CALLS it, and it does not prove a record survives the trip.
Both gaps matter here, because the defect this guards is precisely a helper that
exists, is green, and is unwired - the silence then reads as "the lane did not
run" rather than "the lane ran unobserved".

Two guards, deliberately different in kind:

* **Emission.** A module logger's ``.info()`` must reach the handler's stream
  carrying its logger name. Asserting the handler is attached at INFO (which the
  sibling module already does) leaves the actual record path untested - level,
  formatter and propagation all have to line up for ``graph.skipped`` to appear.
* **Wiring.** ``main`` must call ``configure_logging()`` at module scope. This is
  read statically rather than by importing ``main`` and inspecting the root
  logger: pytest attaches its own capture handlers to the real root, so
  ``configure_logging()`` takes its documented no-op branch under test and a
  root-handler assertion would pass whether or not ``main`` ever called it - a
  green that carries no information. The AST cannot be fooled that way, and it
  also catches the call being indented into a function that never runs.
"""

from __future__ import annotations

import ast
import io
import logging
from pathlib import Path

import pytest

from chora_ai_kernel_orchestrator.observability.logging_bootstrap import (
    configure_logging,
)

MAIN_SOURCE = Path(__file__).resolve().parents[2] / "src" / "chora_ai_kernel_orchestrator" / "main.py"

# Registered through getLogger, not constructed directly: children resolve their
# parent through the manager, so a directly-built Logger would leave
# getChild() parented to the REAL root and the emission test would assert
# nothing about this handler.
STANDIN_ROOT = "chora-test-standin-root"


@pytest.fixture()
def bare_root() -> logging.Logger:
    root = logging.getLogger(STANDIN_ROOT)
    root.handlers.clear()
    root.setLevel(logging.NOTSET)
    return root


def _stream_of(root: logging.Logger) -> io.StringIO:
    stream = io.StringIO()
    root.handlers[0].setStream(stream)  # type: ignore[attr-defined]
    return stream


def test_module_logger_info_reaches_the_stream(bare_root: logging.Logger, monkeypatch: pytest.MonkeyPatch) -> None:
    """An INFO record renders with its logger name - the `graph.skipped` case."""
    monkeypatch.delenv("CHORA_LOG_LEVEL", raising=False)
    configure_logging(bare_root)
    stream = _stream_of(bare_root)

    bare_root.getChild("adapter.langgraph").info("graph.skipped")

    rendered = stream.getvalue()
    assert "graph.skipped" in rendered
    assert "INFO" in rendered
    assert "adapter.langgraph" in rendered


def test_debug_is_withheld_at_the_default_level(bare_root: logging.Logger, monkeypatch: pytest.MonkeyPatch) -> None:
    """The positive control's negative half: INFO passing must mean something."""
    monkeypatch.delenv("CHORA_LOG_LEVEL", raising=False)
    configure_logging(bare_root)
    stream = _stream_of(bare_root)

    bare_root.getChild("adapter.langgraph").debug("graph.noisy")

    assert stream.getvalue() == ""


def test_main_calls_configure_logging_at_module_scope() -> None:
    """Delete or indent the call in `main` and this fails."""
    tree = ast.parse(MAIN_SOURCE.read_text(encoding="utf-8"))

    module_level_calls = {
        node.value.func.id
        for node in tree.body
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
    }

    assert "configure_logging" in module_level_calls, (
        "main.py must call configure_logging() at module scope, before any "
        "adapter import emits its first INFO breadcrumb"
    )
