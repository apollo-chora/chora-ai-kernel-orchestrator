"""RED: nothing in this service may swallow a failure without saying so.

The standing engineering rule is fail-loud: surface failures explicitly, never
swallow an error, never fabricate a success. This file is the durable half of
the 2026-08-23 fail-loud sweep. The sweep itself fixed the sites that existed;
these guards are what stop the class coming back, because every instance of it
looks locally reasonable and none of them fail a test.

Three shapes, in order of how invisible they are:

1. ``contextlib.suppress(Exception)``. The most invisible of the three, and the
   one an exception-handling census misses entirely: it is not an
   ``ExceptHandler`` node, so a sweep that walks ``except`` clauses reports a
   clean tree while the suppressions sit right next to them. Shutdown paths are
   where these accumulate, and a loop that could not be stopped is exactly why
   the next boot finds a subscription still attached.

2. A handler whose whole body is ``pass`` (or ``...``). Nothing recorded, no
   value degraded on purpose, no comment: the failure simply did not happen as
   far as any operator can tell.

3. A broad handler that logs only at ``info``. INFO is not a failure channel
   here: this service has run with no logging handler configured, where every
   INFO line was invisible, and an INFO on an error path reads to a human
   scanning logs as normal operation rather than as something that went wrong.

Each guard names its own escape hatch. The point is not that a best-effort
catch is forbidden, it is that choosing one has to be deliberate and visible.
"""

from __future__ import annotations

import ast
from pathlib import Path

from chora_ai_kernel_orchestrator import main as _main_module

_SRC_ROOT = Path(_main_module.__file__).parent

_LOUD = {"exception", "error", "critical", "warning"}


def _iter_modules() -> list[tuple[str, ast.Module]]:
    out = []
    for path in sorted(_SRC_ROOT.rglob("*.py")):
        out.append((str(path.relative_to(_SRC_ROOT)), ast.parse(path.read_text())))
    return out


def _is_broad(handler: ast.ExceptHandler) -> bool:
    t = handler.type
    if t is None:
        return True
    names = [t] if not isinstance(t, ast.Tuple) else list(t.elts)
    return any(isinstance(n, ast.Name) and n.id in {"Exception", "BaseException"} for n in names)


def _log_levels(handler: ast.ExceptHandler) -> set[str]:
    levels = set()
    for node in ast.walk(handler):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in {"logger", "log", "_logger"}
        ):
            levels.add(node.func.attr)
    return levels


def _reraises(handler: ast.ExceptHandler) -> bool:
    return any(isinstance(n, ast.Raise) for stmt in handler.body for n in ast.walk(stmt))


# -----------------------------------------------------------------------------
# 1. contextlib.suppress(Exception)
# -----------------------------------------------------------------------------


def test_nothing_suppresses_a_broad_exception_silently() -> None:
    """``with suppress(Exception)`` records nothing at all.

    Escape hatch: catch it, log it, carry on. That keeps the "teardown must not
    abort" property while leaving a line behind. If a suppression is genuinely
    right, narrow it to the exact exception type, which is not matched here.
    """
    offenders: list[str] = []
    for label, tree in _iter_modules():
        for node in ast.walk(tree):
            if not isinstance(node, ast.With | ast.AsyncWith):
                continue
            for item in node.items:
                call = item.context_expr
                if not isinstance(call, ast.Call):
                    continue
                fname = call.func.attr if isinstance(call.func, ast.Attribute) else getattr(call.func, "id", "")
                if fname != "suppress":
                    continue
                if any(isinstance(a, ast.Name) and a.id in {"Exception", "BaseException"} for a in call.args):
                    offenders.append(f"{label}:{node.lineno} suppress(Exception)")
    assert offenders == [], "silent broad suppression(s); catch, log and carry on instead:\n  " + "\n  ".join(offenders)


# -----------------------------------------------------------------------------
# 2. a handler whose entire body is pass
# -----------------------------------------------------------------------------


def test_no_broad_handler_is_a_bare_pass() -> None:
    offenders: list[str] = []
    for label, tree in _iter_modules():
        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler) or not _is_broad(node):
                continue
            body = [s for s in node.body if not isinstance(s, ast.Expr) or not isinstance(s.value, ast.Constant)]
            if len(body) == 1 and isinstance(body[0], ast.Pass):
                offenders.append(f"{label}:{node.lineno}")
    assert offenders == [], "broad handler(s) whose whole body is `pass`:\n  " + "\n  ".join(offenders)


# -----------------------------------------------------------------------------
# 3. INFO is not a failure channel
# -----------------------------------------------------------------------------


def test_no_broad_handler_reports_a_failure_at_info_only() -> None:
    """A broad handler must re-raise, or log at warning or above.

    INFO-only is the shape that hid a real defect twice: this service has run
    with no logging handler configured, where INFO went nowhere at all, and an
    INFO on an error path reads as normal operation to anyone scanning logs.
    """
    offenders: list[str] = []
    for label, tree in _iter_modules():
        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler) or not _is_broad(node):
                continue
            if _reraises(node):
                continue
            levels = _log_levels(node)
            if levels and not (levels & _LOUD):
                offenders.append(f"{label}:{node.lineno} logs only {sorted(levels)}")
    assert offenders == [], "broad handler(s) reporting a failure below warning:\n  " + "\n  ".join(offenders)


# -----------------------------------------------------------------------------
# 4. every broad handler is accountable
# -----------------------------------------------------------------------------

_EXEMPT_MARKER = "fail-loud-exempt:"


def test_every_broad_handler_reraises_logs_or_is_explicitly_exempt() -> None:
    """A broad ``except`` must do one of three things, all of them visible.

    Re-raise, log at warning or above, or carry a ``# fail-loud-exempt: <why>``
    comment. The third exists because a blanket rule would be a lie: some broad
    catches are correct precisely BECAUSE the failure is the return value. A
    connection probe answering False means "reconnect", and logging every miss
    on that hot path would bury the failures that matter.

    The point of the marker is that the exemption becomes deliberate, greppable
    and reviewable, instead of being indistinguishable from the ones nobody
    thought about. If you find yourself adding one, say what makes the silence
    correct, not that it is convenient.
    """
    offenders: list[str] = []
    for path in sorted(_SRC_ROOT.rglob("*.py")):
        label = str(path.relative_to(_SRC_ROOT))
        lines = path.read_text().split("\n")
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler) or not _is_broad(node):
                continue
            if _reraises(node) or (_log_levels(node) & _LOUD):
                continue
            # The marker may sit on the `except` line or anywhere in the body.
            window = "\n".join(lines[node.lineno - 1 : node.end_lineno])
            if _EXEMPT_MARKER in window:
                continue
            offenders.append(f"{label}:{node.lineno}")
    assert offenders == [], (
        "broad handler(s) that neither re-raise, nor log at warning+, nor carry "
        f"a `# {_EXEMPT_MARKER} <why>` comment:\n  " + "\n  ".join(offenders)
    )


def test_every_exemption_states_a_reason() -> None:
    """``# fail-loud-exempt:`` with nothing after it is not a reason."""
    bad: list[str] = []
    for path in sorted(_SRC_ROOT.rglob("*.py")):
        for i, line in enumerate(path.read_text().split("\n"), start=1):
            if _EXEMPT_MARKER not in line:
                continue
            reason = line.split(_EXEMPT_MARKER, 1)[1].strip()
            if len(reason) < 20:
                bad.append(f"{path.relative_to(_SRC_ROOT)}:{i} reason too thin: {reason!r}")
    assert bad == [], "exemption(s) without a real reason:\n  " + "\n  ".join(bad)


# -----------------------------------------------------------------------------
# 5. a lane that dispatches must ride the transactional saver
#
# `require_transactional_saver` enforces this at RUNTIME, per lane, and that is
# the load-bearing check. This is the census half: a FIFTH lane added later
# could simply never call the guard, and nothing would notice until the ADR's
# single-transaction claim was quietly false again for that one lane. The same
# blind spot the park-eater guard had when it was keyed on four hand-listed
# modules.
#
# The seam by EFFECT is registration: a lane exists when something calls
# `runtime.register_lane(...)`. A lane registered with NO roles never
# dispatches, so it never parks and needs no saver; `prompt_promotion_audit` is
# exactly that, which is why the rule is scoped to lanes carrying roles.
# -----------------------------------------------------------------------------


def _register_lane_calls(tree: ast.Module) -> list[ast.Call]:
    return [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "register_lane"
    ]


def _registers_a_dispatching_lane(tree: ast.Module) -> bool:
    """True when a register_lane call passes a non-empty `roles`."""
    for call in _register_lane_calls(tree):
        for kw in call.keywords:
            if kw.arg != "roles":
                continue
            # An empty literal tuple/list is the no-dispatch lane; anything else
            # (a name, a constant collection with members) is a dispatching one.
            if isinstance(kw.value, ast.Tuple | ast.List) and not kw.value.elts:
                continue
            return True
    return False


def _calls_the_saver_guard(tree: ast.Module) -> bool:
    return any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "require_transactional_saver"
        for n in ast.walk(tree)
    )


def test_every_dispatching_lane_passes_its_saver_through_the_guard() -> None:
    offenders: list[str] = []
    for label, tree in _iter_modules():
        if not _registers_a_dispatching_lane(tree):
            continue
        if not _calls_the_saver_guard(tree):
            offenders.append(label)
    assert offenders == [], (
        "lane(s) registered with roles but never passing their checkpointer "
        "through require_transactional_saver; ADR-253 D3a needs the park and "
        "the dispatch outbox row in ONE transaction:\n  " + "\n  ".join(offenders)
    )


def test_the_dispatching_lane_detector_can_actually_fail() -> None:
    """POSITIVE CONTROL for the guard above, and for its exemption."""
    dispatching = ast.parse("runtime.register_lane('x', crew='x', roles=('a','b'), runner=r)")
    assert _registers_a_dispatching_lane(dispatching)
    assert not _calls_the_saver_guard(dispatching), "unguarded module must be flagged"

    no_roles = ast.parse("runtime.register_lane('x', crew='x', roles=(), runner=None)")
    assert not _registers_a_dispatching_lane(no_roles), "a lane with no roles never dispatches and must stay exempt"

    guarded = ast.parse(
        "s = require_transactional_saver(x)\nruntime.register_lane('x', crew='x', roles=ROLES, runner=r)"
    )
    assert _registers_a_dispatching_lane(guarded) and _calls_the_saver_guard(guarded)
