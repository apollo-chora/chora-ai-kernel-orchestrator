"""Tests for the W6 managed 4-criterion qgen-loop Eval harness
(``eval/qgen_loop_eval.py``).

Strict-TDD coverage of:

  - DETERMINISTIC criteria 2 (infinite-loop risk), 3 (token compounding),
    4 (total latency) — asserted to compute EXACTLY on synthetic
    enriched-pipeline_trace fixtures.
  - Criterion 1 (managed Vertex Gen-AI Eval autorater) — asserted to build
    the correct per-attempt transition tuples + call a MOCKED
    ``eval_task_runner`` with them (NO live Vertex call).
  - The fail-loud contract — the live DB / EvalTask / metric-logging paths
    raise a clear ``QGenEvalDependencyError`` when their dep is absent.

Fixtures cover the four required runs:
  (a) a 2-attempt reject→accept run (deliberately-FLAWED candidate_1: an MCQ
      with TWO correct options → critic correctly rejects → actor fixes →
      accept on attempt 2);
  (b) a 1-attempt happy path;
  (c) a max-attempts-no-accept LIVELOCK run (4 attempts, never accepted);
  (d) a run with an empty-``suggested_revisions`` rejection (non-actionable
      feedback = livelock signal).

The harness lives OUTSIDE ``src/`` (in ``eval/``), so this test bootstraps
that directory onto ``sys.path`` and imports the module top-level. No source
under ``src/`` is touched.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

# --- Bootstrap the eval/ dir (sibling of src/) onto sys.path --------------
# tests/unit/test_qgen_loop_eval.py → parents[2] == the service root, where
# eval/ lives next to src/.
_EVAL_DIR = Path(__file__).resolve().parents[2] / "eval"
if not _EVAL_DIR.exists():
    # The eval/ harness is a separate, cloud-coupled component excluded from
    # this cloud-neutral repo. This test covers that harness, so it skips
    # when the harness is absent rather than failing on the import.
    pytest.skip(
        "eval/ harness not present (excluded from this repo)",
        allow_module_level=True,
    )
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

import qgen_loop_eval as qle  # noqa: E402

# ===========================================================================
# Synthetic enriched-pipeline_trace fixtures (match qgen_crew.py _append_trace)
# ===========================================================================

# Deliberately-FLAWED candidate_1: an MCQ with TWO correct options. This is
# the seed that proves the reject→fix path is MEASURED (Criterion 1).
_FLAWED_MCQ_TWO_CORRECT = (
    '{"stem":"What is 2+2?","mcq_payload":{"options":['
    '{"marker":"A","text":"4","is_correct":true},'
    '{"marker":"B","text":"four","is_correct":true},'  # second correct → flaw
    '{"marker":"C","text":"5","is_correct":false},'
    '{"marker":"D","text":"3","is_correct":false}]}}'
)
# The FIXED candidate the actor produces on attempt 2 (single correct option).
_FIXED_MCQ_ONE_CORRECT = (
    '{"stem":"What is 2+2?","mcq_payload":{"options":['
    '{"marker":"A","text":"4","is_correct":true},'
    '{"marker":"B","text":"22","is_correct":false},'
    '{"marker":"C","text":"5","is_correct":false},'
    '{"marker":"D","text":"3","is_correct":false}]}}'
)
_GOOD_MCQ = (
    '{"stem":"Capital of France?","mcq_payload":{"options":['
    '{"marker":"A","text":"Paris","is_correct":true},'
    '{"marker":"B","text":"Lyon","is_correct":false},'
    '{"marker":"C","text":"Nice","is_correct":false},'
    '{"marker":"D","text":"Lille","is_correct":false}]}}'
)


def _gen_row(
    attempt: int,
    candidate_json: str,
    *,
    status: str = "COMPLETED",
    input_tokens: int = 0,
    output_tokens: int = 0,
    started_at: str = "",
    completed_at: str = "",
    cost_micros: int | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "name": "generate",
        "status": status,
        "attempt": attempt,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "candidate_json": candidate_json,
        "notes": "candidate produced",
    }
    if started_at:
        row["started_at"] = started_at
    if completed_at:
        row["completed_at"] = completed_at
    if cost_micros is not None:
        row["cost_micros"] = cost_micros
    return row


def _crit_row(
    attempt: int,
    *,
    accepted: bool,
    notes: str,
    suggested_revisions: list[str],
    input_tokens: int = 0,
    started_at: str = "",
    completed_at: str = "",
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "name": "critique",
        "status": "ACCEPTED" if accepted else "REJECTED",
        "attempt": attempt,
        "input_tokens": input_tokens,
        "accepted": accepted,
        "suggested_revisions": suggested_revisions,
        "notes": notes,
    }
    if started_at:
        row["started_at"] = started_at
    if completed_at:
        row["completed_at"] = completed_at
    return row


@pytest.fixture
def trace_a_reject_then_accept() -> dict[str, Any]:
    """(a) 2-attempt reject→accept. candidate_1 is FLAWED (two correct MCQ
    options); critic rejects WITH actionable revisions; actor fixes;
    candidate_2 accepted."""
    return {
        "job_id": "job-a",
        "created_at": "2026-06-01T10:00:00+00:00",
        "completed_at": "2026-06-01T10:00:30+00:00",
        "pipeline_trace": [
            {"name": "validate_input", "status": "ACCEPTED"},
            {"name": "guardrail_pre", "status": "ACCEPTED"},
            _gen_row(
                1,
                _FLAWED_MCQ_TWO_CORRECT,
                input_tokens=400,
                output_tokens=120,
                started_at="2026-06-01T10:00:01+00:00",
                completed_at="2026-06-01T10:00:09+00:00",
                cost_micros=1500,
            ),
            {"name": "guardrail_post", "status": "ACCEPTED", "attempt": 1},
            _crit_row(
                1,
                accepted=False,
                notes="Two options (A and B) are both correct for '2+2'. Exactly one option must be correct.",
                suggested_revisions=["Make only option A correct", "Fix option B"],
                input_tokens=300,
                started_at="2026-06-01T10:00:09+00:00",
                completed_at="2026-06-01T10:00:14+00:00",
            ),
            {"name": "quality_gate", "status": "RETRY", "attempt": 1},
            _gen_row(
                2,
                _FIXED_MCQ_ONE_CORRECT,
                input_tokens=500,
                output_tokens=130,
                started_at="2026-06-01T10:00:15+00:00",
                completed_at="2026-06-01T10:00:23+00:00",
                cost_micros=1800,
            ),
            {"name": "guardrail_post", "status": "ACCEPTED", "attempt": 2},
            _crit_row(
                2,
                accepted=True,
                notes="Single correct option; distractors plausible.",
                suggested_revisions=[],
                input_tokens=320,
                started_at="2026-06-01T10:00:23+00:00",
                completed_at="2026-06-01T10:00:28+00:00",
            ),
            {"name": "quality_gate", "status": "ACCEPTED", "attempt": 2},
            {"name": "publish_completed", "status": "COMPLETED", "attempt": 2},
        ],
    }


@pytest.fixture
def trace_b_happy_path() -> dict[str, Any]:
    """(b) 1-attempt happy path — accepted on the first pass."""
    return {
        "job_id": "job-b",
        "created_at": "2026-06-01T11:00:00+00:00",
        "completed_at": "2026-06-01T11:00:12+00:00",
        "pipeline_trace": [
            {"name": "validate_input", "status": "ACCEPTED"},
            {"name": "guardrail_pre", "status": "ACCEPTED"},
            _gen_row(
                1,
                _GOOD_MCQ,
                input_tokens=420,
                output_tokens=110,
                started_at="2026-06-01T11:00:01+00:00",
                completed_at="2026-06-01T11:00:08+00:00",
                cost_micros=1400,
            ),
            {"name": "guardrail_post", "status": "ACCEPTED", "attempt": 1},
            _crit_row(
                1,
                accepted=True,
                notes="Clear stem, one correct answer, plausible distractors.",
                suggested_revisions=[],
                input_tokens=280,
                started_at="2026-06-01T11:00:08+00:00",
                completed_at="2026-06-01T11:00:11+00:00",
            ),
            {"name": "quality_gate", "status": "ACCEPTED", "attempt": 1},
            {"name": "publish_completed", "status": "COMPLETED", "attempt": 1},
        ],
    }


@pytest.fixture
def trace_c_livelock_max_attempts() -> dict[str, Any]:
    """(c) max-attempts-no-accept LIVELOCK — 4 attempts, critic rejects every
    time, never accepted (terminal quality_warning). Revisions ARE present
    (actionable) so this isolates the max-without-accept signal from the
    non-actionable signal in (d)."""
    rows: list[dict[str, Any]] = [
        {"name": "validate_input", "status": "ACCEPTED"},
        {"name": "guardrail_pre", "status": "ACCEPTED"},
    ]
    for attempt in range(1, 5):  # 4 attempts (max_attempts)
        rows.append(
            _gen_row(
                attempt,
                _FLAWED_MCQ_TWO_CORRECT,
                input_tokens=400,
                output_tokens=100,
                cost_micros=1500,
            )
        )
        rows.append({"name": "guardrail_post", "status": "ACCEPTED", "attempt": attempt})
        rows.append(
            _crit_row(
                attempt,
                accepted=False,
                notes=f"Still two correct options (attempt {attempt}).",
                suggested_revisions=["Make only one option correct"],
                input_tokens=300,
            )
        )
        gate = "RETRY" if attempt < 4 else "QUALITY_WARNING"
        rows.append({"name": "quality_gate", "status": gate, "attempt": attempt})
    rows.append({"name": "publish_completed", "status": "COMPLETED", "attempt": 4})
    return {
        "job_id": "job-c",
        "created_at": "2026-06-01T12:00:00+00:00",
        "completed_at": "2026-06-01T12:01:20+00:00",  # 80s E2E
        "pipeline_trace": rows,
    }


@pytest.fixture
def trace_d_empty_revisions() -> dict[str, Any]:
    """(d) rejection with EMPTY suggested_revisions = non-actionable feedback.
    2 attempts: attempt 1 rejected with NO revisions (the actor has no signal
    to fix), attempt 2 accepted. Proves Criterion 2's non-actionable count."""
    return {
        "job_id": "job-d",
        "created_at": "2026-06-01T13:00:00+00:00",
        "completed_at": "2026-06-01T13:00:25+00:00",
        "pipeline_trace": [
            {"name": "validate_input", "status": "ACCEPTED"},
            {"name": "guardrail_pre", "status": "ACCEPTED"},
            _gen_row(1, _FLAWED_MCQ_TWO_CORRECT, input_tokens=400, output_tokens=100),
            {"name": "guardrail_post", "status": "ACCEPTED", "attempt": 1},
            _crit_row(
                1,
                accepted=False,
                notes="Not good enough.",  # vague, no concrete revisions
                suggested_revisions=[],  # EMPTY → non-actionable
                input_tokens=300,
            ),
            {"name": "quality_gate", "status": "RETRY", "attempt": 1},
            _gen_row(2, _FIXED_MCQ_ONE_CORRECT, input_tokens=450, output_tokens=110),
            {"name": "guardrail_post", "status": "ACCEPTED", "attempt": 2},
            _crit_row(
                2,
                accepted=True,
                notes="Good now.",
                suggested_revisions=[],
                input_tokens=310,
            ),
            {"name": "quality_gate", "status": "ACCEPTED", "attempt": 2},
            {"name": "publish_completed", "status": "COMPLETED", "attempt": 2},
        ],
    }


@pytest.fixture
def all_traces(
    trace_a_reject_then_accept: dict[str, Any],
    trace_b_happy_path: dict[str, Any],
    trace_c_livelock_max_attempts: dict[str, Any],
    trace_d_empty_revisions: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        trace_a_reject_then_accept,
        trace_b_happy_path,
        trace_c_livelock_max_attempts,
        trace_d_empty_revisions,
    ]


# ===========================================================================
# Trajectory reconstruction
# ===========================================================================


class TestReconstructTrajectory:
    def test_reject_then_accept_builds_two_transitions(self, trace_a_reject_then_accept: dict[str, Any]) -> None:
        traj = qle.reconstruct_trajectory(
            trace_a_reject_then_accept["pipeline_trace"],
            job_id="job-a",
            created_at=trace_a_reject_then_accept["created_at"],
            completed_at=trace_a_reject_then_accept["completed_at"],
        )
        assert traj.attempt_count == 2
        assert traj.accepted is True
        assert traj.quality_warning is False
        assert len(traj.transitions) == 2

        t0, t1 = traj.transitions
        # Transition 0: flawed candidate, rejected, with actionable revisions,
        # and next_candidate == the fixed candidate (the actor APPLIED it).
        assert t0.attempt_index == 0
        assert t0.candidate_json == _FLAWED_MCQ_TWO_CORRECT
        assert t0.accepted is False
        assert "both correct" in t0.critique_notes or "correct" in t0.critique_notes
        assert t0.suggested_revisions == ["Make only option A correct", "Fix option B"]
        assert t0.next_candidate_json == _FIXED_MCQ_ONE_CORRECT
        # Transition 1: the fixed candidate, accepted, no next candidate.
        assert t1.attempt_index == 1
        assert t1.candidate_json == _FIXED_MCQ_ONE_CORRECT
        assert t1.accepted is True
        assert t1.next_candidate_json == ""

    def test_happy_path_single_transition(self, trace_b_happy_path: dict[str, Any]) -> None:
        traj = qle.reconstruct_trajectory(trace_b_happy_path["pipeline_trace"], job_id="job-b")
        assert traj.attempt_count == 1
        assert traj.accepted is True
        assert len(traj.transitions) == 1
        assert traj.transitions[0].accepted is True

    def test_livelock_four_transitions_never_accepted(self, trace_c_livelock_max_attempts: dict[str, Any]) -> None:
        traj = qle.reconstruct_trajectory(trace_c_livelock_max_attempts["pipeline_trace"], job_id="job-c")
        assert traj.attempt_count == 4
        assert traj.accepted is False
        assert traj.quality_warning is True
        assert all(not t.accepted for t in traj.transitions)

    def test_non_list_trace_fails_loud(self) -> None:
        with pytest.raises(TypeError, match="must be a list"):
            qle.reconstruct_trajectory({"not": "a list"})  # type: ignore[arg-type]


# ===========================================================================
# Criterion 2 — Infinite-loop risk (DETERMINISTIC)
# ===========================================================================


class TestCriterion2InfiniteLoop:
    def test_metrics_compute_exactly(self, all_traces: list[dict[str, Any]]) -> None:
        records = qle.job_records_from_traces(all_traces)
        trajectories = [
            qle.reconstruct_trajectory(
                r.pipeline_trace,
                job_id=r.job_id,
                created_at=r.created_at,
                completed_at=r.completed_at,
            )
            for r in records
        ]
        m = qle.compute_infinite_loop_metrics(trajectories, max_attempts=4)

        assert m.total_runs == 4
        # Accepted runs: a (attempt 2), b (attempt 1), d (attempt 2). c never
        # accepts. Distribution = {1: 1 (b), 2: 2 (a, d)}.
        assert m.attempts_to_accept_distribution == {1: 1, 2: 2}
        # mean attempts-to-accept = (2 + 1 + 2) / 3 = 1.666...
        assert m.mean_attempts_to_accept == pytest.approx(5 / 3)
        # Only c hit max (4 attempts) without accepting.
        assert m.runs_hit_max_without_accept == 1
        assert m.pct_hit_max_without_accept == pytest.approx(25.0)
        # Total rejections across all transitions:
        #   a: 1 (attempt1) ; b: 0 ; c: 4 ; d: 1  → 6
        assert m.total_rejections == 6
        # Empty-revision rejections: only d's attempt-1 rejection → 1.
        assert m.rejections_with_empty_revisions == 1
        assert m.pct_rejections_non_actionable == pytest.approx(100 / 6)

    def test_livelock_only_flags_max_without_accept(self, trace_c_livelock_max_attempts: dict[str, Any]) -> None:
        records = qle.job_records_from_traces([trace_c_livelock_max_attempts])
        trajectories = [qle.reconstruct_trajectory(r.pipeline_trace, job_id=r.job_id) for r in records]
        m = qle.compute_infinite_loop_metrics(trajectories, max_attempts=4)
        assert m.runs_hit_max_without_accept == 1
        assert m.pct_hit_max_without_accept == pytest.approx(100.0)
        # c's revisions are actionable → 0 non-actionable.
        assert m.rejections_with_empty_revisions == 0


# ===========================================================================
# Criterion 3 — Token compounding (DETERMINISTIC)
# ===========================================================================


class TestCriterion3TokenCompounding:
    def test_per_question_and_aggregate(self, all_traces: list[dict[str, Any]]) -> None:
        records = qle.job_records_from_traces(all_traces)
        trajectories = [qle.reconstruct_trajectory(r.pipeline_trace, job_id=r.job_id) for r in records]
        m = qle.compute_token_compounding_metrics(trajectories)

        by_id = {p.job_id: p for p in m.per_question}
        # job-a: attempt1 gen(400+120)+crit(300) = 820 ; attempt2 gen(500+130)
        #        +crit(320) = 950 → input=400+300+500+320=1520,
        #        output=120+130=250, total=1770.
        a = by_id["job-a"]
        assert a.total_input_tokens == 1520
        assert a.total_output_tokens == 250
        assert a.total_tokens == 1770
        assert a.total_cost_micros == 1500 + 1800
        assert a.accepted is True
        assert a.attempt_count == 2

        # job-b: gen(420+110)+crit(280) → input=700, output=110, total=810.
        b = by_id["job-b"]
        assert b.total_tokens == 810
        assert b.total_cost_micros == 1400

        # job-c: 4×[gen(400+100)+crit(300)] = 4×800 = 3200 ;
        #        input=4×700=2800, output=4×100=400.
        c = by_id["job-c"]
        assert c.total_input_tokens == 2800
        assert c.total_output_tokens == 400
        assert c.total_tokens == 3200
        assert c.total_cost_micros == 4 * 1500

        # Aggregate across ACCEPTED questions only (a, b, d).
        # d: attempt1 gen(400+100)+crit(300)=800 ; attempt2 gen(450+110)
        #    +crit(310)=870 → total 1670.
        d = by_id["job-d"]
        assert d.total_tokens == 1670
        assert m.accepted_total_tokens == 1770 + 810 + 1670  # a+b+d
        assert m.accepted_mean_tokens_per_question == pytest.approx((1770 + 810 + 1670) / 3)

        # All-runs aggregate + marginal-per-attempt growth.
        all_total = 1770 + 810 + 3200 + 1670
        all_attempts = 2 + 1 + 4 + 2  # = 9
        assert m.all_runs_total_tokens == all_total
        assert m.mean_tokens_per_attempt == pytest.approx(all_total / all_attempts)
        assert m.all_runs_total_cost_micros == (1500 + 1800) + 1400 + (4 * 1500) + 0
        assert m.mean_cost_micros_per_attempt == pytest.approx(m.all_runs_total_cost_micros / all_attempts)


# ===========================================================================
# Criterion 4 — Total latency (DETERMINISTIC)
# ===========================================================================


class TestCriterion4Latency:
    def test_e2e_percentiles_and_marginal(self, all_traces: list[dict[str, Any]]) -> None:
        records = qle.job_records_from_traces(all_traces)
        trajectories = [
            qle.reconstruct_trajectory(
                r.pipeline_trace,
                job_id=r.job_id,
                created_at=r.created_at,
                completed_at=r.completed_at,
            )
            for r in records
        ]
        m = qle.compute_latency_metrics(trajectories, records)

        # E2E latencies: a=30s, b=12s, c=80s, d=25s.
        assert sorted(m.e2e_latencies_s) == [12.0, 25.0, 30.0, 80.0]
        # p50 (linear interp over [12,25,30,80]): rank=0.5*3=1.5 →
        # 25 + 0.5*(30-25) = 27.5.
        assert m.p50_s == pytest.approx(27.5)
        # p95: rank=0.95*3=2.85 → 30 + 0.85*(80-30) = 72.5.
        assert m.p95_s == pytest.approx(72.5)
        assert m.mean_s == pytest.approx((30 + 12 + 80 + 25) / 4)
        # Marginal latency per attempt: a + b have per-row timestamps.
        #  a: gen1 8s + crit1 5s + gen2 8s + crit2 5s = 26s / 2 attempts = 13s.
        #  b: gen1 7s + crit1 3s = 10s / 1 attempt = 10s.
        #  c, d: rows lack timestamps → fall back to E2E/attempts:
        #    c: 80/4 = 20s ; d: 25/2 = 12.5s.
        # mean of [13, 10, 20, 12.5] = 13.875.
        assert m.mean_marginal_latency_per_attempt_s == pytest.approx((13.0 + 10.0 + 20.0 + 12.5) / 4)

    def test_latency_falls_back_to_trace_span_without_job_timestamps(self) -> None:
        # No created_at/completed_at → fall back to trace first-start→last-end.
        trace = {
            "job_id": "job-x",
            "pipeline_trace": [
                _gen_row(
                    1,
                    _GOOD_MCQ,
                    started_at="2026-06-01T09:00:00+00:00",
                    completed_at="2026-06-01T09:00:06+00:00",
                ),
                _crit_row(
                    1,
                    accepted=True,
                    notes="ok",
                    suggested_revisions=[],
                    started_at="2026-06-01T09:00:06+00:00",
                    completed_at="2026-06-01T09:00:10+00:00",
                ),
            ],
        }
        records = qle.job_records_from_traces([trace])
        traj = qle.reconstruct_trajectory(records[0].pipeline_trace, job_id="job-x")
        # E2E from trace span = 09:00:00 → 09:00:10 = 10s.
        assert traj.e2e_latency_s == pytest.approx(10.0)


# ===========================================================================
# Criterion 1 — Trajectory (managed autorater) via MOCKED runner
# ===========================================================================


class _MockEvalTaskRunner:
    """Records the rows + kwargs it was called with; returns a canned summary
    so no live Vertex EvalTask is constructed."""

    def __init__(self) -> None:
        self.rows: list[qle.TrajectoryEvalRow] = []
        self.kwargs: dict[str, Any] = {}
        self.calls = 0

    def __call__(
        self,
        rows: list[qle.TrajectoryEvalRow],
        *,
        experiment_name: str,
        location: str,
        project: str,
    ) -> dict[str, Any]:
        self.calls += 1
        self.rows = rows
        self.kwargs = {
            "experiment_name": experiment_name,
            "location": location,
            "project": project,
        }
        return {
            f"{qle.TRAJECTORY_METRIC_NAME}/mean": 0.875,
            "row_count": len(rows),
        }


class TestCriterion1TrajectoryMocked:
    def test_builds_correct_transition_rows(self, all_traces: list[dict[str, Any]]) -> None:
        trajectories = [
            qle.reconstruct_trajectory(r.pipeline_trace, job_id=r.job_id)
            for r in qle.job_records_from_traces(all_traces)
        ]
        rows = qle.build_trajectory_eval_rows(trajectories)
        # One row per transition: a(2) + b(1) + c(4) + d(2) = 9.
        assert len(rows) == 9

        # The flawed-candidate transition (job-a, attempt 0) must carry the
        # reject verdict + the next (fixed) candidate so the rubric can judge
        # both "reject-was-correct" AND "critique-applied".
        a0 = next(r for r in rows if r.job_id == "job-a" and r.attempt_index == 0)
        assert a0.critic_verdict == "REJECTED"
        assert a0.candidate == _FLAWED_MCQ_TWO_CORRECT
        assert a0.next_candidate == _FIXED_MCQ_ONE_CORRECT
        assert "Make only option A correct" in a0.suggested_revisions

    def test_run_eval_calls_mocked_runner_with_transitions(self, all_traces: list[dict[str, Any]]) -> None:
        runner = _MockEvalTaskRunner()
        rec_logger = qle.RecordingMetricsLogger()
        report = qle.run_eval(
            traces=all_traces,
            eval_task_runner=runner,
            metrics_logger=rec_logger,
        )
        # Criterion 1 ran via the MOCK (no Vertex).
        assert runner.calls == 1
        assert len(runner.rows) == 9
        assert runner.kwargs["experiment_name"] == qle.DEFAULT_EXPERIMENT_NAME
        assert runner.kwargs["location"] == qle.DEFAULT_LOCATION
        # The mocked summary flowed into the report + logged metrics.
        assert report.trajectory_summary["row_count"] == 9
        assert report.trajectory_summary[f"{qle.TRAJECTORY_METRIC_NAME}/mean"] == 0.875

        # The metrics logger captured all 4 criteria's summary scalars.
        assert rec_logger.calls == 1
        lm = rec_logger.metrics
        assert lm["c2_total_runs"] == 4.0
        assert lm["c2_pct_hit_max_without_accept"] == pytest.approx(25.0)
        assert lm["c3_all_runs_total_tokens"] == float(report.token_compounding.all_runs_total_tokens)
        assert lm["c4_e2e_p50_s"] == pytest.approx(27.5)
        # Criterion 1 summary flattened with the c1_ prefix.
        assert lm[f"c1_{qle.TRAJECTORY_METRIC_NAME}/mean"] == 0.875
        # Distribution logged as a param.
        assert "c2_attempts_to_accept_distribution" in rec_logger.params

    def test_run_eval_skips_trajectory_when_disabled(self, all_traces: list[dict[str, Any]]) -> None:
        runner = _MockEvalTaskRunner()
        cfg = qle.QGenLoopEvalConfig(run_trajectory_eval=False)
        report = qle.run_eval(
            traces=all_traces,
            config=cfg,
            eval_task_runner=runner,
            metrics_logger=qle.RecordingMetricsLogger(),
        )
        assert runner.calls == 0
        assert report.trajectory_summary == {}
        # Deterministic criteria still computed.
        assert report.infinite_loop.total_runs == 4


# ===========================================================================
# Fail-loud contract (live paths raise without their dep)
# ===========================================================================


class TestFailLoud:
    def test_run_eval_requires_exactly_one_source(self) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            qle.run_eval(metrics_logger=qle.RecordingMetricsLogger())
        with pytest.raises(ValueError, match="exactly one"):
            qle.run_eval(
                records=[],
                traces=[],
                metrics_logger=qle.RecordingMetricsLogger(),
            )

    def test_load_from_db_without_dsn_fails_loud(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(qle.ENV_CREATION_DSN, raising=False)
        monkeypatch.delenv(qle.ENV_CREATION_DSN_SECRET_ID, raising=False)
        with pytest.raises(qle.QGenEvalDependencyError, match="no chora_creation DSN"):
            qle.load_completed_jobs_from_db()

    def test_resolve_creation_dsn_prefers_direct(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(qle.ENV_CREATION_DSN, "postgresql://direct/creation")
        monkeypatch.setenv(qle.ENV_CREATION_DSN_SECRET_ID, "ignored")
        assert qle.resolve_creation_dsn() == "postgresql://direct/creation"

    def test_resolve_creation_dsn_empty_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(qle.ENV_CREATION_DSN, raising=False)
        monkeypatch.delenv(qle.ENV_CREATION_DSN_SECRET_ID, raising=False)
        assert qle.resolve_creation_dsn() == ""

    def test_default_eval_task_runner_empty_rows_fails_loud(self) -> None:
        # Even with vertexai installed, zero rows must fail loud (no empty
        # managed-eval run).
        with pytest.raises(qle.QGenEvalDependencyError, match="no transition rows"):
            qle.default_eval_task_runner(
                [],
                experiment_name="x",
                location="us-central1",
                project="p",
            )

    def test_default_metrics_logger_without_aiplatform_fails_loud(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Simulate google-cloud-aiplatform being absent by blocking the import.
        import builtins

        real_import = builtins.__import__

        def _blocked(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "google.cloud" and args and "aiplatform" in (args[2] or ()):
                raise ImportError("simulated: aiplatform absent")
            if name.startswith("google.cloud.aiplatform"):
                raise ImportError("simulated: aiplatform absent")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _blocked)
        with pytest.raises(qle.QGenEvalDependencyError, match="google-cloud-aiplatform"):
            qle.default_metrics_logger(
                experiment_name="x",
                location="us-central1",
                project="p",
                metrics={"a": 1.0},
                params={},
            )
