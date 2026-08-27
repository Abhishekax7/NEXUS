import pytest

from app.evaluation.benchmark import (
    BenchmarkEngine,
    BenchmarkError,
    RegressionSeverity,
)
from app.evaluation.models import (
    EvaluationDimension,
    EvaluationStatus,
    MetricScore,
    WorkflowEvaluation,
)


ALL_DIMENSIONS = list(
    EvaluationDimension
)


def build_evaluation(
    run_id: str,
    score: float = 90.0,
    overrides=None,
):
    overrides = overrides or {}

    metrics = []

    for dimension in ALL_DIMENSIONS:
        metric_score = overrides.get(
            dimension,
            score,
        )

        if metric_score >= 80:
            status = EvaluationStatus.PASS

        elif metric_score >= 60:
            status = EvaluationStatus.WARN

        else:
            status = EvaluationStatus.FAIL

        metrics.append(
            MetricScore(
                dimension=dimension,
                score=metric_score,
                status=status,
                reason=(
                    f"{dimension.value} "
                    "benchmark score."
                ),
                evidence=[
                    "Deterministic test evidence."
                ],
            )
        )

    overall_score = (
        sum(
            metric.score
            for metric in metrics
        )
        / len(metrics)
    )

    return WorkflowEvaluation(
        run_id=run_id,
        overall_score=overall_score,
        status=(
            EvaluationStatus.PASS
            if overall_score >= 80
            else (
                EvaluationStatus.WARN
                if overall_score >= 60
                else EvaluationStatus.FAIL
            )
        ),
        metrics=metrics,
        agent_evaluations=[],
        strengths=[],
        weaknesses=[],
        recommendations=[],
        regression_risk=0.0,
    )


def delta_for(
    report,
    dimension,
):
    return next(
        item
        for item in report.dimension_deltas
        if item.dimension == dimension
    )


def test_identical_quality_has_no_regression():
    baseline = build_evaluation(
        "baseline-run",
        score=90,
    )

    candidate = build_evaluation(
        "candidate-run",
        score=90,
    )

    report = BenchmarkEngine().compare(
        baseline,
        candidate,
    )

    assert (
        report.regression_detected
        is False
    )

    assert (
        report.severity
        == RegressionSeverity.NONE
    )

    assert report.overall_delta == 0.0

    assert (
        len(
            report.unchanged_dimensions
        )
        == len(ALL_DIMENSIONS)
    )


def test_single_dimension_improvement_detected():
    baseline = build_evaluation(
        "baseline-run",
        score=80,
    )

    candidate = build_evaluation(
        "candidate-run",
        score=80,
        overrides={
            EvaluationDimension.TOOL_USE:
                95,
        },
    )

    report = BenchmarkEngine().compare(
        baseline,
        candidate,
    )

    assert (
        EvaluationDimension.TOOL_USE
        in report.improved_dimensions
    )

    assert (
        report.regression_detected
        is False
    )

    delta = delta_for(
        report,
        EvaluationDimension.TOOL_USE,
    )

    assert delta.delta == 15.0
    assert delta.improved is True
    assert delta.regressed is False


def test_single_dimension_regression_detected():
    baseline = build_evaluation(
        "baseline-run",
        score=90,
    )

    candidate = build_evaluation(
        "candidate-run",
        score=90,
        overrides={
            EvaluationDimension.SECURITY:
                70,
        },
    )

    report = BenchmarkEngine().compare(
        baseline,
        candidate,
    )

    assert (
        report.regression_detected
        is True
    )

    assert (
        EvaluationDimension.SECURITY
        in report.regressed_dimensions
    )

    delta = delta_for(
        report,
        EvaluationDimension.SECURITY,
    )

    assert delta.delta == -20.0
    assert delta.regressed is True


def test_small_drop_within_tolerance_is_unchanged():
    baseline = build_evaluation(
        "baseline-run",
        score=90,
    )

    candidate = build_evaluation(
        "candidate-run",
        score=90,
        overrides={
            EvaluationDimension.GROUNDING:
                86,
        },
    )

    report = BenchmarkEngine(
        regression_tolerance=5.0
    ).compare(
        baseline,
        candidate,
    )

    assert (
        EvaluationDimension.GROUNDING
        not in report.regressed_dimensions
    )

    assert (
        EvaluationDimension.GROUNDING
        in report.unchanged_dimensions
    )


def test_drop_beyond_tolerance_is_regression():
    baseline = build_evaluation(
        "baseline-run",
        score=90,
    )

    candidate = build_evaluation(
        "candidate-run",
        score=90,
        overrides={
            EvaluationDimension.GROUNDING:
                84,
        },
    )

    report = BenchmarkEngine(
        regression_tolerance=5.0
    ).compare(
        baseline,
        candidate,
    )

    assert (
        EvaluationDimension.GROUNDING
        in report.regressed_dimensions
    )


def test_exact_regression_tolerance_is_not_regression():
    baseline = build_evaluation(
        "baseline-run",
        score=90,
    )

    candidate = build_evaluation(
        "candidate-run",
        score=90,
        overrides={
            EvaluationDimension.GROUNDING:
                85,
        },
    )

    report = BenchmarkEngine(
        regression_tolerance=5.0
    ).compare(
        baseline,
        candidate,
    )

    assert (
        EvaluationDimension.GROUNDING
        in report.unchanged_dimensions
    )


def test_exact_improvement_threshold_is_unchanged():
    baseline = build_evaluation(
        "baseline-run",
        score=80,
    )

    candidate = build_evaluation(
        "candidate-run",
        score=80,
        overrides={
            EvaluationDimension.TOOL_USE:
                85,
        },
    )

    report = BenchmarkEngine(
        improvement_threshold=5.0
    ).compare(
        baseline,
        candidate,
    )

    assert (
        EvaluationDimension.TOOL_USE
        in report.unchanged_dimensions
    )


def test_improvement_beyond_threshold_detected():
    baseline = build_evaluation(
        "baseline-run",
        score=80,
    )

    candidate = build_evaluation(
        "candidate-run",
        score=80,
        overrides={
            EvaluationDimension.TOOL_USE:
                86,
        },
    )

    report = BenchmarkEngine(
        improvement_threshold=5.0
    ).compare(
        baseline,
        candidate,
    )

    assert (
        EvaluationDimension.TOOL_USE
        in report.improved_dimensions
    )


def test_low_severity_regression():
    baseline = build_evaluation(
        "baseline-run",
        score=90,
    )

    candidate = build_evaluation(
        "candidate-run",
        score=90,
        overrides={
            EvaluationDimension.TOOL_USE:
                82,
        },
    )

    report = BenchmarkEngine().compare(
        baseline,
        candidate,
    )

    assert (
        report.severity
        == RegressionSeverity.LOW
    )


def test_medium_severity_from_worst_dimension():
    baseline = build_evaluation(
        "baseline-run",
        score=90,
    )

    candidate = build_evaluation(
        "candidate-run",
        score=90,
        overrides={
            EvaluationDimension.SECURITY:
                65,
        },
    )

    report = BenchmarkEngine().compare(
        baseline,
        candidate,
    )

    assert (
        report.severity
        == RegressionSeverity.MEDIUM
    )


def test_high_severity_from_large_dimension_drop():
    baseline = build_evaluation(
        "baseline-run",
        score=90,
    )

    candidate = build_evaluation(
        "candidate-run",
        score=90,
        overrides={
            EvaluationDimension.SECURITY:
                50,
        },
    )

    report = BenchmarkEngine().compare(
        baseline,
        candidate,
    )

    assert (
        report.severity
        == RegressionSeverity.HIGH
    )


def test_medium_severity_from_multiple_regressions():
    baseline = build_evaluation(
        "baseline-run",
        score=90,
    )

    candidate = build_evaluation(
        "candidate-run",
        score=90,
        overrides={
            EvaluationDimension.GROUNDING:
                82,
            EvaluationDimension.TOOL_USE:
                82,
            EvaluationDimension.TEST_QUALITY:
                82,
        },
    )

    report = BenchmarkEngine().compare(
        baseline,
        candidate,
    )

    assert len(
        report.regressed_dimensions
    ) == 3

    assert (
        report.severity
        == RegressionSeverity.MEDIUM
    )


def test_high_severity_from_many_regressions():
    baseline = build_evaluation(
        "baseline-run",
        score=90,
    )

    candidate = build_evaluation(
        "candidate-run",
        score=90,
        overrides={
            EvaluationDimension.GROUNDING:
                82,
            EvaluationDimension.TOOL_USE:
                82,
            EvaluationDimension.TEST_QUALITY:
                82,
            EvaluationDimension.SECURITY:
                82,
            EvaluationDimension.ARTIFACT_QUALITY:
                82,
        },
    )

    report = BenchmarkEngine().compare(
        baseline,
        candidate,
    )

    assert len(
        report.regressed_dimensions
    ) == 5

    assert (
        report.severity
        == RegressionSeverity.HIGH
    )


def test_report_contains_all_dimension_deltas():
    baseline = build_evaluation(
        "baseline-run",
        score=90,
    )

    candidate = build_evaluation(
        "candidate-run",
        score=90,
    )

    report = BenchmarkEngine().compare(
        baseline,
        candidate,
    )

    assert (
        len(
            report.dimension_deltas
        )
        == len(ALL_DIMENSIONS)
    )

    dimensions = {
        item.dimension
        for item
        in report.dimension_deltas
    }

    assert dimensions == set(
        ALL_DIMENSIONS
    )


def test_report_preserves_run_ids():
    baseline = build_evaluation(
        "baseline-123",
        score=90,
    )

    candidate = build_evaluation(
        "candidate-456",
        score=92,
    )

    report = BenchmarkEngine().compare(
        baseline,
        candidate,
    )

    assert (
        report.baseline_run_id
        == "baseline-123"
    )

    assert (
        report.candidate_run_id
        == "candidate-456"
    )


def test_overall_delta_is_calculated():
    baseline = build_evaluation(
        "baseline-run",
        score=80,
    )

    candidate = build_evaluation(
        "candidate-run",
        score=90,
    )

    report = BenchmarkEngine().compare(
        baseline,
        candidate,
    )

    assert report.overall_delta == 10.0


def test_improvement_summary_mentions_no_regression():
    baseline = build_evaluation(
        "baseline-run",
        score=80,
    )

    candidate = build_evaluation(
        "candidate-run",
        score=90,
    )

    report = BenchmarkEngine().compare(
        baseline,
        candidate,
    )

    assert (
        "No regression detected"
        in report.summary
    )


def test_regression_summary_mentions_regression():
    baseline = build_evaluation(
        "baseline-run",
        score=90,
    )

    candidate = build_evaluation(
        "candidate-run",
        score=70,
    )

    report = BenchmarkEngine().compare(
        baseline,
        candidate,
    )

    assert (
        "Regression detected"
        in report.summary
    )


def test_same_run_cannot_be_compared():
    baseline = build_evaluation(
        "same-run",
        score=90,
    )

    candidate = build_evaluation(
        "same-run",
        score=80,
    )

    with pytest.raises(
        BenchmarkError,
        match="different runs",
    ):
        BenchmarkEngine().compare(
            baseline,
            candidate,
        )


def test_mismatched_dimensions_are_rejected():
    baseline = build_evaluation(
        "baseline-run",
        score=90,
    )

    candidate = build_evaluation(
        "candidate-run",
        score=90,
    )

    candidate.metrics = [
        metric
        for metric
        in candidate.metrics
        if (
            metric.dimension
            != EvaluationDimension.SECURITY
        )
    ]

    with pytest.raises(
        BenchmarkError,
        match="dimensions do not match",
    ):
        BenchmarkEngine().compare(
            baseline,
            candidate,
        )


def test_negative_regression_tolerance_rejected():
    with pytest.raises(
        ValueError,
        match="regression_tolerance",
    ):
        BenchmarkEngine(
            regression_tolerance=-1
        )


def test_negative_improvement_threshold_rejected():
    with pytest.raises(
        ValueError,
        match="improvement_threshold",
    ):
        BenchmarkEngine(
            improvement_threshold=-1
        )


def test_custom_regression_tolerance_changes_behavior():
    baseline = build_evaluation(
        "baseline-run",
        score=90,
    )

    candidate = build_evaluation(
        "candidate-run",
        score=90,
        overrides={
            EvaluationDimension.SECURITY:
                82,
        },
    )

    strict_report = BenchmarkEngine(
        regression_tolerance=5
    ).compare(
        baseline,
        candidate,
    )

    relaxed_report = BenchmarkEngine(
        regression_tolerance=10
    ).compare(
        baseline,
        candidate,
    )

    assert (
        strict_report.regression_detected
        is True
    )

    assert (
        relaxed_report.regression_detected
        is False
    )


def test_dimension_delta_preserves_scores():
    baseline = build_evaluation(
        "baseline-run",
        score=90,
    )

    candidate = build_evaluation(
        "candidate-run",
        score=90,
        overrides={
            EvaluationDimension.SECURITY:
                72,
        },
    )

    report = BenchmarkEngine().compare(
        baseline,
        candidate,
    )

    delta = delta_for(
        report,
        EvaluationDimension.SECURITY,
    )

    assert (
        delta.baseline_score
        == 90
    )

    assert (
        delta.candidate_score
        == 72
    )

    assert delta.delta == -18
