from dataclasses import dataclass
from typing import Optional

from app.core.state import NexusState
from app.evaluation.benchmark import (
    BenchmarkEngine,
    BenchmarkReport,
)
from app.evaluation.engine import (
    EvaluationEngine,
)
from app.evaluation.history import (
    EvaluationHistoryStore,
)
from app.evaluation.models import (
    WorkflowEvaluation,
)


@dataclass
class EvaluationServiceResult:
    """
    Combined result of evaluating a NEXUS run.

    evaluation:
        Deterministic score for the current run.

    benchmark:
        Regression comparison against the
        active baseline, when available.

    baseline_run_id:
        Active baseline used for comparison.

    baseline_created:
        True when this run was automatically
        promoted as the first baseline.
    """

    evaluation: WorkflowEvaluation

    benchmark: Optional[
        BenchmarkReport
    ]

    baseline_run_id: Optional[
        str
    ]

    baseline_created: bool


class EvaluationService:
    """
    Production evaluation pipeline for NEXUS.

    Responsibilities:

    - evaluate a completed NexusState
    - persist the evaluation
    - optionally create the first baseline
    - compare future runs against the baseline
    - expose one structured result
    """

    def __init__(
        self,
        evaluation_engine: EvaluationEngine,
        history_store: EvaluationHistoryStore,
        benchmark_engine: BenchmarkEngine,
        auto_create_baseline: bool = True,
    ):
        self.evaluation_engine = (
            evaluation_engine
        )

        self.history_store = (
            history_store
        )

        self.benchmark_engine = (
            benchmark_engine
        )

        self.auto_create_baseline = (
            auto_create_baseline
        )

    def evaluate_run(
        self,
        state: NexusState,
    ) -> EvaluationServiceResult:
        evaluation = (
            self.evaluation_engine.evaluate(
                state
            )
        )

        current_baseline = (
            self.history_store
            .get_baseline()
        )

        self.history_store.save(
            evaluation
        )

        if current_baseline is None:
            baseline_created = False
            baseline_run_id = None

            if self.auto_create_baseline:
                self.history_store.set_baseline(
                    evaluation.run_id
                )

                baseline_created = True

                baseline_run_id = (
                    evaluation.run_id
                )

            return EvaluationServiceResult(
                evaluation=evaluation,
                benchmark=None,
                baseline_run_id=(
                    baseline_run_id
                ),
                baseline_created=(
                    baseline_created
                ),
            )

        if (
            current_baseline.run_id
            == evaluation.run_id
        ):
            return EvaluationServiceResult(
                evaluation=evaluation,
                benchmark=None,
                baseline_run_id=(
                    current_baseline.run_id
                ),
                baseline_created=False,
            )

        benchmark = (
            self.benchmark_engine.compare(
                baseline=current_baseline,
                candidate=evaluation,
            )
        )

        return EvaluationServiceResult(
            evaluation=evaluation,
            benchmark=benchmark,
            baseline_run_id=(
                current_baseline.run_id
            ),
            baseline_created=False,
        )

    def set_baseline(
        self,
        run_id: str,
    ) -> None:
        self.history_store.set_baseline(
            run_id
        )

    def clear_baseline(
        self,
    ) -> None:
        self.history_store.clear_baseline()

    def get_baseline(
        self,
    ) -> Optional[
        WorkflowEvaluation
    ]:
        return (
            self.history_store
            .get_baseline()
        )

    def get_evaluation(
        self,
        run_id: str,
    ) -> Optional[
        WorkflowEvaluation
    ]:
        return self.history_store.get(
            run_id
        )

    def recent_evaluations(
        self,
        limit: int = 10,
    ) -> list[
        WorkflowEvaluation
    ]:
        return (
            self.history_store
            .list_recent(
                limit=limit
            )
        )
