from app.core.models import (
    AgentRole,
    AgentTask,
    Artifact,
    ArtifactType,
    TaskStatus,
)
from app.core.state import NexusState

from app.evaluation.benchmark import (
    BenchmarkEngine,
    RegressionSeverity,
)
from app.evaluation.engine import (
    EvaluationEngine,
)
from app.evaluation.history import (
    EvaluationHistoryStore,
)
from app.evaluation.service import (
    EvaluationService,
    EvaluationServiceResult,
)


def build_state(
    *,
    test_passed: bool = True,
    security_risk: float = 10.0,
    completed: bool = True,
    failed: bool = False,
):
    state = NexusState(
        user_request=(
            "Build a secure FastAPI application."
        )
    )

    task = AgentTask(
        title="Implement application",
        description=(
            "Build and validate the application."
        ),
        assigned_agent=AgentRole.CODER,
        status=(
            TaskStatus.COMPLETED
            if completed
            else TaskStatus.PENDING
        ),
    )

    state.add_task(
        task
    )

    code = Artifact(
        type=ArtifactType.CODE,
        name="generated_code",
        content={
            "files": [
                {
                    "path": "app.py",
                    "content": "print('ok')",
                }
            ]
        },
        created_by=AgentRole.CODER,
        metadata={
            "grounded_in_requirements": True,
        },
    )

    test_result = Artifact(
        type=ArtifactType.TEST_RESULT,
        name="test_result",
        content={
            "passed": test_passed,
            "summary": (
                "Tests passed."
                if test_passed
                else "Tests failed."
            ),
        },
        created_by=AgentRole.TESTER,
    )

    security = Artifact(
        type=ArtifactType.SECURITY_REPORT,
        name="security_report",
        content={
            "passed": (
                security_risk < 50
            ),
            "risk_score": security_risk,
            "summary": (
                "Security review completed."
            ),
        },
        created_by=AgentRole.SECURITY,
    )

    critic = Artifact(
        type=ArtifactType.EVALUATION,
        name="critic_evaluation",
        content={
            "verdict": (
                "accept"
                if test_passed
                and security_risk < 50
                else "revise"
            ),
            "quality_score": (
                95
                if test_passed
                and security_risk < 50
                else 60
            ),
        },
        created_by=AgentRole.CRITIC,
    )

    for artifact in [
        code,
        test_result,
        security,
        critic,
    ]:
        state.add_artifact(
            artifact
        )

    state.completed = completed
    state.failed = failed

    if failed:
        state.errors.append(
            "Workflow failed."
        )

    return state


def build_service(
    tmp_path,
    *,
    auto_create_baseline=True,
):
    history = EvaluationHistoryStore(
        db_path=str(
            tmp_path
            / "evaluations.db"
        )
    )

    return EvaluationService(
        evaluation_engine=(
            EvaluationEngine()
        ),
        history_store=history,
        benchmark_engine=(
            BenchmarkEngine()
        ),
        auto_create_baseline=(
            auto_create_baseline
        ),
    )


def test_evaluate_run_returns_service_result(
    tmp_path,
):
    service = build_service(
        tmp_path
    )

    result = service.evaluate_run(
        build_state()
    )

    assert isinstance(
        result,
        EvaluationServiceResult,
    )

    assert (
        result.evaluation
        is not None
    )


def test_first_run_is_persisted(
    tmp_path,
):
    service = build_service(
        tmp_path
    )

    state = build_state()

    result = service.evaluate_run(
        state
    )

    stored = service.get_evaluation(
        state.run_id
    )

    assert stored is not None

    assert (
        stored.run_id
        == result.evaluation.run_id
    )


def test_first_run_becomes_baseline_by_default(
    tmp_path,
):
    service = build_service(
        tmp_path
    )

    state = build_state()

    result = service.evaluate_run(
        state
    )

    assert (
        result.baseline_created
        is True
    )

    assert (
        result.baseline_run_id
        == state.run_id
    )

    assert result.benchmark is None

    baseline = service.get_baseline()

    assert baseline is not None

    assert (
        baseline.run_id
        == state.run_id
    )


def test_first_run_does_not_become_baseline_when_disabled(
    tmp_path,
):
    service = build_service(
        tmp_path,
        auto_create_baseline=False,
    )

    state = build_state()

    result = service.evaluate_run(
        state
    )

    assert (
        result.baseline_created
        is False
    )

    assert (
        result.baseline_run_id
        is None
    )

    assert result.benchmark is None

    assert (
        service.get_baseline()
        is None
    )


def test_second_run_is_compared_with_baseline(
    tmp_path,
):
    service = build_service(
        tmp_path
    )

    first = build_state()

    service.evaluate_run(
        first
    )

    second = build_state()

    result = service.evaluate_run(
        second
    )

    assert (
        result.benchmark
        is not None
    )

    assert (
        result.baseline_run_id
        == first.run_id
    )

    assert (
        result.benchmark.baseline_run_id
        == first.run_id
    )

    assert (
        result.benchmark.candidate_run_id
        == second.run_id
    )


def test_equal_quality_second_run_has_no_regression(
    tmp_path,
):
    service = build_service(
        tmp_path
    )

    service.evaluate_run(
        build_state()
    )

    result = service.evaluate_run(
        build_state()
    )

    assert result.benchmark is not None

    assert (
        result.benchmark.regression_detected
        is False
    )

    assert (
        result.benchmark.severity
        == RegressionSeverity.NONE
    )


def test_worse_candidate_detects_regression(
    tmp_path,
):
    service = build_service(
        tmp_path
    )

    baseline = build_state(
        test_passed=True,
        security_risk=5.0,
    )

    service.evaluate_run(
        baseline
    )

    candidate = build_state(
        test_passed=False,
        security_risk=90.0,
        completed=False,
        failed=True,
    )

    result = service.evaluate_run(
        candidate
    )

    assert result.benchmark is not None

    assert (
        result.benchmark.regression_detected
        is True
    )

    assert (
        len(
            result.benchmark
            .regressed_dimensions
        )
        > 0
    )


def test_candidate_is_also_persisted(
    tmp_path,
):
    service = build_service(
        tmp_path
    )

    baseline = build_state()
    candidate = build_state()

    service.evaluate_run(
        baseline
    )

    service.evaluate_run(
        candidate
    )

    stored = service.get_evaluation(
        candidate.run_id
    )

    assert stored is not None

    assert (
        stored.run_id
        == candidate.run_id
    )


def test_baseline_remains_original_after_candidate(
    tmp_path,
):
    service = build_service(
        tmp_path
    )

    baseline_state = build_state()

    service.evaluate_run(
        baseline_state
    )

    service.evaluate_run(
        build_state()
    )

    baseline = service.get_baseline()

    assert baseline is not None

    assert (
        baseline.run_id
        == baseline_state.run_id
    )


def test_manual_baseline_can_be_changed(
    tmp_path,
):
    service = build_service(
        tmp_path
    )

    first = build_state()
    second = build_state()

    service.evaluate_run(
        first
    )

    service.evaluate_run(
        second
    )

    service.set_baseline(
        second.run_id
    )

    baseline = service.get_baseline()

    assert baseline is not None

    assert (
        baseline.run_id
        == second.run_id
    )


def test_manual_baseline_is_used_for_future_run(
    tmp_path,
):
    service = build_service(
        tmp_path
    )

    first = build_state()
    second = build_state()

    service.evaluate_run(
        first
    )

    service.evaluate_run(
        second
    )

    service.set_baseline(
        second.run_id
    )

    third = build_state()

    result = service.evaluate_run(
        third
    )

    assert result.benchmark is not None

    assert (
        result.baseline_run_id
        == second.run_id
    )

    assert (
        result.benchmark.baseline_run_id
        == second.run_id
    )


def test_baseline_can_be_cleared(
    tmp_path,
):
    service = build_service(
        tmp_path
    )

    service.evaluate_run(
        build_state()
    )

    service.clear_baseline()

    assert (
        service.get_baseline()
        is None
    )


def test_new_baseline_created_after_clear(
    tmp_path,
):
    service = build_service(
        tmp_path
    )

    first = build_state()

    service.evaluate_run(
        first
    )

    service.clear_baseline()

    second = build_state()

    result = service.evaluate_run(
        second
    )

    assert (
        result.baseline_created
        is True
    )

    assert (
        result.baseline_run_id
        == second.run_id
    )


def test_recent_evaluations_are_exposed(
    tmp_path,
):
    service = build_service(
        tmp_path
    )

    for _ in range(3):
        service.evaluate_run(
            build_state()
        )

    recent = (
        service.recent_evaluations(
            limit=2
        )
    )

    assert len(recent) == 2


def test_same_baseline_run_is_not_compared_to_itself(
    tmp_path,
):
    service = build_service(
        tmp_path
    )

    state = build_state()

    first_result = (
        service.evaluate_run(
            state
        )
    )

    assert (
        first_result.baseline_created
        is True
    )

    second_result = (
        service.evaluate_run(
            state
        )
    )

    assert (
        second_result.benchmark
        is None
    )

    assert (
        second_result.baseline_run_id
        == state.run_id
    )


def test_history_survives_new_service_instance(
    tmp_path,
):
    db_path = (
        tmp_path
        / "evaluations.db"
    )

    first_service = EvaluationService(
        evaluation_engine=(
            EvaluationEngine()
        ),
        history_store=(
            EvaluationHistoryStore(
                db_path=str(
                    db_path
                )
            )
        ),
        benchmark_engine=(
            BenchmarkEngine()
        ),
    )

    baseline_state = build_state()

    first_service.evaluate_run(
        baseline_state
    )

    second_service = EvaluationService(
        evaluation_engine=(
            EvaluationEngine()
        ),
        history_store=(
            EvaluationHistoryStore(
                db_path=str(
                    db_path
                )
            )
        ),
        benchmark_engine=(
            BenchmarkEngine()
        ),
    )

    baseline = (
        second_service.get_baseline()
    )

    assert baseline is not None

    assert (
        baseline.run_id
        == baseline_state.run_id
    )


def test_new_service_uses_persisted_baseline(
    tmp_path,
):
    db_path = (
        tmp_path
        / "evaluations.db"
    )

    first_service = EvaluationService(
        evaluation_engine=(
            EvaluationEngine()
        ),
        history_store=(
            EvaluationHistoryStore(
                db_path=str(
                    db_path
                )
            )
        ),
        benchmark_engine=(
            BenchmarkEngine()
        ),
    )

    baseline_state = build_state()

    first_service.evaluate_run(
        baseline_state
    )

    second_service = EvaluationService(
        evaluation_engine=(
            EvaluationEngine()
        ),
        history_store=(
            EvaluationHistoryStore(
                db_path=str(
                    db_path
                )
            )
        ),
        benchmark_engine=(
            BenchmarkEngine()
        ),
    )

    candidate = build_state()

    result = (
        second_service.evaluate_run(
            candidate
        )
    )

    assert result.benchmark is not None

    assert (
        result.benchmark.baseline_run_id
        == baseline_state.run_id
    )


def test_service_preserves_evaluation_score(
    tmp_path,
):
    service = build_service(
        tmp_path
    )

    state = build_state()

    direct = (
        EvaluationEngine().evaluate(
            state
        )
    )

    result = service.evaluate_run(
        state
    )

    assert (
        result.evaluation.overall_score
        == direct.overall_score
    )


def test_service_result_reports_no_new_baseline_for_candidate(
    tmp_path,
):
    service = build_service(
        tmp_path
    )

    service.evaluate_run(
        build_state()
    )

    result = service.evaluate_run(
        build_state()
    )

    assert (
        result.baseline_created
        is False
    )
