"""companion_diagnosis crew (ex weakness analyser): the PURE analysis core.

The crew consumes ``chora.consumption.weakness_doc.uploaded.v1`` and, since
ADR-254 D5, dispatches every model call over the bus (``companion_extract``,
``companion_diagnose`` by ``task_kind``); it publishes
``chora.consumption.weakness.{review_pending, analyzed, outputs_generated}.v1``.

This package holds what stays pure: the structured-clues rendering, the
structured-output parsing with a confidence threshold (``analysis``), and the
bounded HITL review panel (``panel``). The adapters live under ``adapter/``;
the graph and its runner live in ``orchestrators/``.
"""
