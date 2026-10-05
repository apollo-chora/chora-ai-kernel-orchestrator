"""RED: ADR-253, a best-effort ``except Exception`` must not eat a dispatch park.

Under the HTTP transport of ADR-169, ``executor.execute`` returned a response and
a best-effort ``try/except Exception`` around it was correct: a failed hop should
not sink the whole run.

Under the Pub/Sub transport of ADR-253 that call no longer returns. It PARKS the
run via ``interrupt()``, and LangGraph signals a park by raising
``GraphInterrupt``, which subclasses ``GraphBubbleUp(Exception)``. So every
pre-existing best-effort guard around a dispatch silently became a park-eater:

  1. the interrupt never reaches LangGraph, so the graph never parks;
  2. the dispatch outbox row written in the same transaction as the park
     (ADR-253 D3a) rolls back with it, so the request is never published;
  3. the node returns its degraded value and the graph runs on to its terminal,
     settling the run as though the hop had merely failed.

That is exactly how the whole-assessment overall comment was lost on the live OE
lane on 2026-08-21: zero ``:summary`` dispatch rows had EVER been written, while
per-question rows were fine, because only ``assess_summary_node`` was wrapped.

⚠ Why an in-memory fake cannot see this: a fake executor that RETURNS a value
never exercises the guard at all. The park has to be raised, which is what
``_ParkingExecutor`` below does. This is the same blind spot that hid the
``get_next_version`` defect (green unit tests, broken on the wire).

The AST test is the durable half: it fails when a NEW dispatch call site is added
inside a bare ``except Exception``, which is how this defect class spreads.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest
from langgraph.errors import GraphBubbleUp, GraphInterrupt

from chora_ai_kernel_orchestrator.orchestrators import (
    oe_grading_crew,
    qgen_crew,
    single_agent_workflow,
    weakness_analyser_crew,
)
from chora_ai_kernel_orchestrator.orchestrators.oe_grading_crew import (
    assess_summary_node,
)
from chora_ai_kernel_orchestrator.orchestrators.weakness_analyser_crew import (
    diagnose_node,
)


class _ParkingExecutor:
    """Duck-typed executor that PARKS, exactly as the Pub/Sub executor does.

    ``adapter/pubsub/pubsub_agent_executor.py`` calls ``interrupt(...)``, whose
    raise is a ``GraphInterrupt``. Simulating the park by raising it here is the
    whole point: a fake that returns a payload proves nothing about the guard.
    """

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def execute(self, *, execution_id: str, **_: Any) -> Any:
        self.calls.append(execution_id)
        raise GraphInterrupt(({"__chora_agent_dispatch__": {"execution_id": execution_id}},))


@pytest.mark.asyncio
async def test_assess_summary_node_lets_the_park_propagate() -> None:
    """The park must escape ``assess_summary_node``, not be swallowed as an error.

    If this fails with no exception raised, the run settles with an EMPTY overall
    comment and the summary dispatch is never published, silently, and the learner
    and the instructor both see a blank OVERALL COMMENT band.
    """
    executor = _ParkingExecutor()
    state: dict[str, Any] = {
        "submission_id": "01a02062-e5b4-7870-8fca-53ce363cd542",
        "assessment_id": "01a02062-72e3-7da4-aeb5-c441ed92d9c8",
        "tenant_id": "11111111-1111-7111-8111-111111111111",
        "gcid": "00000000-0000-7000-8000-000000001999",
        "subject": "",
        "passing_threshold_percent": 70,
        "total_points_possible": 20,
        "mcq_points_earned": 0,
        "graded": [],
        "mcq_results": [],
        "oe_questions": [],
    }

    with pytest.raises(GraphBubbleUp):
        await assess_summary_node(state, executor=executor)

    assert executor.calls, "the node must actually reach the dispatch before parking"


# -----------------------------------------------------------------------------
# The durable half: no dispatch may sit inside a bare `except Exception` that
# does not re-raise the park first.
# -----------------------------------------------------------------------------

_GUARD_NAME = "reraise_if_dispatch_park"
_PARK_TYPES = {"GraphBubbleUp", "GraphInterrupt"}


# The dispatch seam is not ONE method name. OE and qgen both go through
# ``_ExecutorLike.execute``; the growth-edge lane's diagnoser port speaks its own
# verb (``Diagnoser.diagnose``). A guard keyed on the convention is blind to the
# non-adopter, which is exactly how this defect class reached a third crew
# unnoticed. Enumerate the seams by EFFECT ("a call that may PARK"), not by the
# name the first two callers happened to use.
_DISPATCH_METHODS = {"execute", "diagnose", "extract", "run_task"}


def _is_dispatch_call(node: ast.AST) -> bool:
    """True for any call that may park: ``<x>.execute(...)`` / ``<x>.diagnose(...)``."""
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in _DISPATCH_METHODS


def _handler_catches_bare_exception(handler: ast.ExceptHandler) -> bool:
    t = handler.type
    if t is None:
        return True  # bare `except:`
    names = [t] if not isinstance(t, ast.Tuple) else list(t.elts)
    return any(isinstance(n, ast.Name) and n.id in {"Exception", "BaseException"} for n in names)


def _handler_reraises_park(handler: ast.ExceptHandler) -> bool:
    """The handler is safe if it calls the shared guard anywhere in its body."""
    for node in ast.walk(handler):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == _GUARD_NAME:
            return True
    return False


def _try_has_park_handler_before(try_node: ast.Try, handler: ast.ExceptHandler) -> bool:
    """A dedicated ``except GraphBubbleUp: raise`` earlier in the same Try also works."""
    for h in try_node.handlers:
        if h is handler:
            return False
        t = h.type
        names = [t] if not isinstance(t, ast.Tuple) else list(getattr(t, "elts", []))
        if any(isinstance(n, ast.Name) and n.id in _PARK_TYPES for n in names if n is not None):
            return True
    return False


def _unguarded_dispatch_sites(module_path: Path) -> list[str]:
    tree = ast.parse(module_path.read_text())
    offenders: list[str] = []
    for try_node in [n for n in ast.walk(tree) if isinstance(n, ast.Try)]:
        dispatch_lines = [c.lineno for stmt in try_node.body for c in ast.walk(stmt) if _is_dispatch_call(c)]
        if not dispatch_lines:
            continue
        for handler in try_node.handlers:
            if not _handler_catches_bare_exception(handler):
                continue
            if _handler_reraises_park(handler):
                continue
            if _try_has_park_handler_before(try_node, handler):
                continue
            offenders.append(
                f"{module_path.name}:{dispatch_lines[0]} dispatch is inside a bare "
                f"`except` at line {handler.lineno} that never re-raises the park"
            )
    return offenders


@pytest.mark.parametrize(
    "module",
    [oe_grading_crew, qgen_crew, weakness_analyser_crew, single_agent_workflow],
    ids=lambda m: m.__name__,
)
def test_no_dispatch_call_sits_in_a_park_eating_except(module: Any) -> None:
    """Every ``executor.execute`` inside a bare ``except`` must re-raise the park.

    Add a new dispatch behind a best-effort guard and this fails, naming the line.
    """
    offenders = _unguarded_dispatch_sites(Path(module.__file__))
    assert offenders == [], "park-eating dispatch guard(s):\n  " + "\n  ".join(offenders)


# -----------------------------------------------------------------------------
# The growth-edge lane (ADR-253 D7 as amended). A DIFFERENT seam shape: OE and
# qgen dispatch through ``_ExecutorLike.execute``, but the diagnoser is a domain
# PORT (``Diagnoser.diagnose``) whose Pub/Sub implementation parks inside the
# port call. The guard above is extended to that verb; this is the behavioural
# half for it.
# -----------------------------------------------------------------------------


class _ParkingDiagnoser:
    """Duck-typed ``Diagnoser`` port that PARKS instead of returning.

    The Pub/Sub implementation reaches ``interrupt()`` underneath the port, so a
    park surfaces at the ``diagnose(...)`` call exactly like this. A fake that
    returns a ``DiagnoseResult`` cannot exercise the guard at all.
    """

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def diagnose(self, *, extracted_text: str, **_: Any) -> Any:
        self.calls.append(extracted_text)
        raise GraphInterrupt(({"__chora_agent_dispatch__": {"agent_role": "weakness_diagnose"}},))


@pytest.mark.asyncio
async def test_diagnose_node_lets_the_park_propagate() -> None:
    """The park must escape ``diagnose_node``, not become a governance failure.

    This node's guard is the WORST instance of the class found so far. The OE
    summary guard merely logged and returned an empty comment; this one returns
    ``governance_status=GOV_FAILED`` and routes the run to refund. Swallow the
    park here and EVERY diagnosis on the converted lane surfaces to the learner
    as a governance failure, on 100% of runs, while the dispatch outbox row
    rolls back with the park and nothing is ever published.
    """
    diagnoser = _ParkingDiagnoser()
    state: dict[str, Any] = {
        "tenant_id": "11111111-1111-7111-8111-111111111111",
        "learner_gcid": "00000000-0000-7000-8000-000000001999",
        "upload_id": "01a02062-e5b4-7870-8fca-53ce363cd542",
        "extracted_text": "the learner confused mitosis with meiosis",
        "structured_clues": {},
        "upload_kind": "marked_test",
        "traceparent": "",
        "tracestate": "",
    }

    with pytest.raises(GraphBubbleUp):
        await diagnose_node(state, diagnoser=diagnoser)

    assert diagnoser.calls, "the node must actually reach the dispatch before parking"


# -----------------------------------------------------------------------------
# The guard's OWN blind spot (fail-loud sweep, 2026-08-23).
#
# The AST guard above is real, but it is parametrised over FOUR hand-listed
# modules, and its seam predicate is a method-NAME list ({execute, diagnose,
# extract, run_task}). Both halves are the census mistake this file's own
# comment warns about, one level up: a list keyed on the modules that dispatched
# when it was written cannot see a module that starts dispatching later, and a
# name-keyed predicate cannot tell an agent dispatch from a psycopg
# ``cursor.execute``.
#
# The seam by EFFECT is the keyword: an agent dispatch is a call that passes
# ``agent_role=``. Nothing else in this service does, and every psycopg execute
# is positional. Keying on that lets the guard walk the WHOLE package instead of
# a list somebody has to remember to update.
# -----------------------------------------------------------------------------

_SRC_ROOT = Path(qgen_crew.__file__).parent.parent


def _is_agent_dispatch(node: ast.AST) -> bool:
    """A call that may PARK: it passes ``agent_role=``.

    Effect, not name. ``cur.execute(SQL, params)`` is positional and
    ``propagate.extract(carrier)`` takes no such keyword, so neither is mistaken
    for a dispatch; a new crew that dispatches is caught the day it is written.
    """
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and any(kw.arg == "agent_role" for kw in node.keywords)
    )


def _park_eaters_in_source(source: str, label: str) -> list[str]:
    """Every dispatch sitting in a bare ``except`` that never re-raises the park."""
    tree = ast.parse(source)
    offenders: list[str] = []
    for try_node in [n for n in ast.walk(tree) if isinstance(n, ast.Try)]:
        dispatch_lines = [c.lineno for stmt in try_node.body for c in ast.walk(stmt) if _is_agent_dispatch(c)]
        if not dispatch_lines:
            continue
        for handler in try_node.handlers:
            if not _handler_catches_bare_exception(handler):
                continue
            if _handler_reraises_park(handler):
                continue
            if _try_has_park_handler_before(try_node, handler):
                continue
            offenders.append(
                f"{label}:{dispatch_lines[0]} dispatch is inside a bare `except` "
                f"at line {handler.lineno} that never re-raises the park"
            )
    return offenders


def _modules_that_dispatch() -> dict[str, int]:
    """Every module in the package holding at least one real dispatch site."""
    out: dict[str, int] = {}
    for path in sorted(_SRC_ROOT.rglob("*.py")):
        n = sum(1 for node in ast.walk(ast.parse(path.read_text())) if _is_agent_dispatch(node))
        if n:
            out[str(path.relative_to(_SRC_ROOT))] = n
    return out


def test_the_module_list_guard_is_blind_to_modules_that_dispatch() -> None:
    """RED: the four-module parametrize does not cover everything that parks.

    Kept as a live assertion rather than deleted with the fix: it names the
    modules the OLD guard could never have checked, so if anyone narrows the
    package walk back to a list, this says what that costs.
    """
    listed = {
        "orchestrators/oe_grading_crew.py",
        "orchestrators/qgen_crew.py",
        "orchestrators/weakness_analyser_crew.py",
        "orchestrators/single_agent_workflow.py",
    }
    dispatching = set(_modules_that_dispatch())
    assert dispatching - listed == {
        "adapter/weakness/pubsub_dispatch.py",
        "orchestrators/image_regen_graph.py",
        "orchestrators/qgen_dispatch.py",
    }, (
        "the set of modules that dispatch but were never covered by the "
        f"four-module guard changed: {sorted(dispatching - listed)}"
    )


def test_the_park_eater_detector_can_actually_fail() -> None:
    """POSITIVE CONTROL. A zero from the package sweep below is only evidence if
    the detector can produce a non-zero. This feeds it a synthetic park-eater
    of each shape the real code uses and requires a hit for every one."""
    eater = """
async def node(state, *, executor):
    try:
        resp = await executor.execute(agent_role="qgen_generate", input_payload="x")
    except Exception as exc:
        return {"errors": [str(exc)]}
"""
    assert _park_eaters_in_source(eater, "synthetic") != [], "the detector missed a plain park-eater"

    bare = """
async def node(state, *, executor):
    try:
        resp = await executor.execute(agent_role="qgen_generate", input_payload="x")
    except:
        resp = None
"""
    assert _park_eaters_in_source(bare, "synthetic") != [], "the detector missed a bare `except:` park-eater"

    # ...and does NOT fire on the two shapes that are correctly guarded.
    guarded = """
async def node(state, *, executor):
    try:
        resp = await executor.execute(agent_role="qgen_generate", input_payload="x")
    except Exception as exc:
        reraise_if_dispatch_park(exc)
        return {"errors": [str(exc)]}
"""
    assert _park_eaters_in_source(guarded, "synthetic") == []

    dedicated = """
async def node(state, *, executor):
    try:
        resp = await executor.execute(agent_role="qgen_generate", input_payload="x")
    except GraphBubbleUp:
        raise
    except Exception as exc:
        return {"errors": [str(exc)]}
"""
    assert _park_eaters_in_source(dedicated, "synthetic") == []

    # A psycopg execute in a best-effort guard is NOT a dispatch and must not
    # be reported: that false positive is what forced the old module list.
    sql = """
async def probe(conn):
    try:
        await conn.execute("SELECT 1")
    except Exception:
        return False
"""
    assert _park_eaters_in_source(sql, "synthetic") == []


def test_no_dispatch_anywhere_in_the_package_sits_in_a_park_eating_except() -> None:
    """The whole-package sweep. No hand-maintained module list to fall behind."""
    offenders: list[str] = []
    for path in sorted(_SRC_ROOT.rglob("*.py")):
        offenders += _park_eaters_in_source(path.read_text(), str(path.relative_to(_SRC_ROOT)))
    assert offenders == [], "park-eating dispatch guard(s):\n  " + "\n  ".join(offenders)
