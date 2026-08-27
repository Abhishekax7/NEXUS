from enum import Enum

from pydantic import (
    BaseModel,
    Field,
)

from app.evaluation.models import (
    EvaluationDimension,
    WorkflowEvaluation,
)


class RegressionSeverity(
    str,
    Enum,
):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class DimensionDelta(BaseModel):
    dimension: EvaluationDimension

    baseline_score: float = Field(
        ge=0.0,
        le=100.0,
    )

    candidate_score: float = Field(
        ge=0.0,
        le=100.0,
    )

    delta: float

    regressed: bool

    improved: bool


class BenchmarkReport(BaseModel):
    baseline_run_id: str = Field(
        min_length=1
    )

    candidate_run_id: str = Field(
        min_length=1
    )

    baseline_score: float = Field(
        ge=0.0,
        le=100.0,
    )

    candidate_score: float = Field(
        ge=0.0,
        le=100.0,
    )

    overall_delta: float

    improved_dimensions: list[
        EvaluationDimension
    ]

    regressed_dimensions: list[
        EvaluationDimension
    ]

    unchanged_dimensions: list[
        EvaluationDimension
    ]

    dimension_deltas: list[
        DimensionDelta
    ]

    regression_detected: bool

    severity: RegressionSeverity

    summary: str


class BenchmarkError(Exception):
    """
    Raised when two evaluations cannot
    be compared safely.
    """


class BenchmarkEngine:
    """
    Compare two deterministic NEXUS
    workflow evaluations.

    The baseline represents known behavior.
    The candidate represents the new run.
    """

    def __init__(
        self,
        regression_tolerance: float = 5.0,
        improvement_threshold: float = 5.0,
    ):
        if regression_tolerance < 0:
            raise ValueError(
                "regression_tolerance "
                "cannot be negative."
            )

        if improvement_threshold < 0:
            raise ValueError(
                "improvement_threshold "
                "cannot be negative."
            )

        self.regression_tolerance = (
            float(
                regression_tolerance
            )
        )

        self.improvement_threshold = (
            float(
                improvement_threshold
            )
        )

    def _metric_map(
        self,
        evaluation: WorkflowEvaluation,
    ):
        return {
            metric.dimension: metric
            for metric
            in evaluation.metrics
        }

    def _severity(
        self,
        overall_delta: float,
        regressed_dimensions: int,
        worst_delta: float,
    ) -> RegressionSeverity:
        if regressed_dimensions == 0:
            return RegressionSeverity.NONE

        if (
            overall_delta <= -20.0
            or worst_delta <= -30.0
            or regressed_dimensions >= 5
        ):
            return RegressionSeverity.HIGH

        if (
            overall_delta <= -10.0
            or worst_delta <= -20.0
            or regressed_dimensions >= 3
        ):
            return RegressionSeverity.MEDIUM

        return RegressionSeverity.LOW

    def compare(
        self,
        baseline: WorkflowEvaluation,
        candidate: WorkflowEvaluation,
    ) -> BenchmarkReport:
        if (
            baseline.run_id
            == candidate.run_id
        ):
            raise BenchmarkError(
                "Baseline and candidate must "
                "represent different runs."
            )

        baseline_metrics = (
            self._metric_map(
                baseline
            )
        )

        candidate_metrics = (
            self._metric_map(
                candidate
            )
        )

        baseline_dimensions = set(
            baseline_metrics
        )

        candidate_dimensions = set(
            candidate_metrics
        )

        if (
            baseline_dimensions
            != candidate_dimensions
        ):
            missing_candidate = (
                baseline_dimensions
                - candidate_dimensions
            )

            missing_baseline = (
                candidate_dimensions
                - baseline_dimensions
            )

            raise BenchmarkError(
                "Evaluation dimensions do not "
                "match. "
                f"Missing from candidate: "
                f"{sorted(
                    item.value
                    for item
                    in missing_candidate
                )}; "
                f"missing from baseline: "
                f"{sorted(
                    item.value
                    for item
                    in missing_baseline
                )}."
            )

        dimension_deltas = []

        improved_dimensions = []
        regressed_dimensions = []
        unchanged_dimensions = []

        for dimension in sorted(
            baseline_dimensions,
            key=lambda item: item.value,
        ):
            baseline_score = (
                baseline_metrics[
                    dimension
                ].score
            )

            candidate_score = (
                candidate_metrics[
                    dimension
                ].score
            )

            delta = (
                candidate_score
                - baseline_score
            )

            regressed = (
                delta
                < -self.regression_tolerance
            )

            improved = (
                delta
                > self.improvement_threshold
            )

            if regressed:
                regressed_dimensions.append(
                    dimension
                )

            elif improved:
                improved_dimensions.append(
                    dimension
                )

            else:
                unchanged_dimensions.append(
                    dimension
                )

            dimension_deltas.append(
                DimensionDelta(
                    dimension=dimension,
                    baseline_score=(
                        baseline_score
                    ),
                    candidate_score=(
                        candidate_score
                    ),
                    delta=round(
                        delta,
                        2,
                    ),
                    regressed=regressed,
                    improved=improved,
                )
            )

        overall_delta = (
            candidate.overall_score
            - baseline.overall_score
        )

        worst_delta = min(
            (
                item.delta
                for item
                in dimension_deltas
            ),
            default=0.0,
        )

        regression_detected = bool(
            regressed_dimensions
        )

        severity = self._severity(
            overall_delta=overall_delta,
            regressed_dimensions=len(
                regressed_dimensions
            ),
            worst_delta=worst_delta,
        )

        if regression_detected:
            summary = (
                f"Regression detected across "
                f"{len(regressed_dimensions)} "
                "evaluation dimension(s). "
                f"Overall score changed by "
                f"{overall_delta:+.2f} points."
            )

        elif improved_dimensions:
            summary = (
                "No regression detected. "
                f"{len(improved_dimensions)} "
                "evaluation dimension(s) "
                "improved. "
                f"Overall score changed by "
                f"{overall_delta:+.2f} points."
            )

        else:
            summary = (
                "No meaningful regression or "
                "improvement detected. "
                f"Overall score changed by "
                f"{overall_delta:+.2f} points."
            )

        return BenchmarkReport(
            baseline_run_id=(
                baseline.run_id
            ),
            candidate_run_id=(
                candidate.run_id
            ),
            baseline_score=(
                baseline.overall_score
            ),
            candidate_score=(
                candidate.overall_score
            ),
            overall_delta=round(
                overall_delta,
                2,
            ),
            improved_dimensions=(
                improved_dimensions
            ),
            regressed_dimensions=(
                regressed_dimensions
            ),
            unchanged_dimensions=(
                unchanged_dimensions
            ),
            dimension_deltas=(
                dimension_deltas
            ),
            regression_detected=(
                regression_detected
            ),
            severity=severity,
            summary=summary,
        )
