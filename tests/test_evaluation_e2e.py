from app.agents.base import BaseAgent
from app.agents.registry import AgentRegistry

from app.core.engine import NexusEngine
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
)
from app.evaluation.engine import (
    EvaluationEngine,
)
from app.evaluation.history import (
    EvaluationHistoryStore,
)
from app.evaluation.service import (
    EvaluationService,
)


class DeterministicAgent(BaseAgent):
    def __init__(
        self,
        role,
        artifact_type,
        artifact_name,
        content,
    ):
        self.role = role
        self.artifact_type = (
            artifact_type
        )
        self.artifact_name = (
            artifact_name
        )
        self.content = content
        self.calls = 0

    def execute(
        self,
        task,
        state,
    ):
        self.calls += 1

        return Artifact(
            type=self.artifact_type,
            name=self.artifact_name,
            created_by=self.role,
            content=dict(
                self.content
            ),
        )


def build_service(
    db_path,
):
    return EvaluationService(
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
        auto_create_baseline=True,
    )


def build_registry(
    *,
    test_passed=True,
    security_risk=10,
    critic_score=95,
):
    registry = AgentRegistry()

    requirements_agent = (
        DeterministicAgent(
            role=AgentRole.REQUIREMENTS,
            artifact_type=(
                ArtifactType.REQUIREMENTS
            ),
            artifact_name="requirements",
            content={
                "objective": (
                    "Build a secure API."
                ),
            },
        )
    )

    tester_agent = (
        DeterministicAgent(
            role=AgentRole.TESTER,
            artifact_type=(
                ArtifactType.TEST_RESULT
            ),
            artifact_name="tests",
            content={
                "passed": test_passed,
                "summary": (
                    "Tests passed."
                    if test_passed
                    else "Tests failed."
                ),
            },
        )
    )

    security_agent = (
        DeterministicAgent(
            role=AgentRole.SECURITY,
            artifact_type=(
                ArtifactType.SECURITY_REPORT
            ),
            artifact_name="security",
            content={
                "passed": (
                    security_risk < 50
                ),
                "risk_score": security_risk,
                "summary": (
                    "Security review complete."
                ),
            },
        )
    )

    critic_agent = (
        DeterministicAgent(
            role=AgentRole.CRITIC,
            artifact_type=(
                ArtifactType.EVALUATION
            ),
            artifact_name="critic",
            content={
                "verdict": (
                    "accept"
                    if (
                        test_passed
                        and security_risk < 50
                    )
                    else "revise"
                ),
                "quality_score": (
                    critic_score
                ),
            },
        )
    )

    registry.register(
        AgentRole.REQUIREMENTS,
        requirements_agent,
    )

    registry.register(
        AgentRole.TESTER,
        tester_agent,
    )

    registry.register(
        AgentRole.SECURITY,
        security_agent,
    )

    registry.register(
        AgentRole.CRITIC,
        critic_agent,
    )

    return registry


def build_state():
    state = NexusState(
        user_request=(
            "Build and validate a "
            "secure API."
        )
    )

    requirements = AgentTask(
        title="Requirements",
        description="Analyze request.",
        assigned_agent=(
            AgentRole.REQUIREMENTS
        ),
    )

    tests = AgentTask(
        title="Tests",
        description="Run tests.",
        assigned_agent=(
            AgentRole.TESTER
        ),
        dependencies=[
            requirements.id
        ],
    )

    security = AgentTask(
        title="Security",
        description="Review security.",
        assigned_agent=(
            AgentRole.SECURITY
        ),
        dependencies=[
            requirements.id
        ],
    )

    critic = AgentTask(
        title="Critic",
        description="Evaluate workflow.",
        assigned_agent=(
            AgentRole.CRITIC
        ),
        dependencies=[
            tests.id,
            security.id,
        ],
    )

    for task in [
        requirements,
        tests,
        security,
        critic,
    ]:
        state.add_task(
            task
        )

    return state


def test_first_completed_workflow_becomes_baseline(
    tmp_path,
):
    evaluation_db = (
        tmp_path
        / "evaluations.db"
    )

    engine = NexusEngine(
        registry=build_registry(),
        evaluation_service=(
            build_service(
                evaluation_db
            )
        ),
    )

    state = build_state()

    result = engine.run(
        state
    )

    assert result.completed is True
    assert result.failed is False

    assert (
        engine.last_evaluation_result
        is not None
    )

    evaluation_result = (
        engine.last_evaluation_result
    )

    assert (
        evaluation_result
        .baseline_created
        is True
    )

    assert (
        evaluation_result
        .baseline_run_id
        == state.run_id
    )

    assert (
        evaluation_result.benchmark
        is None
    )

    assert (
        "evaluation"
        in result.metadata
    )

    assert (
        result.metadata[
            "evaluation_baseline_created"
        ]
        is True
    )


def test_second_workflow_is_benchmarked_against_first(
    tmp_path,
):
    evaluation_db = (
        tmp_path
        / "evaluations.db"
    )

    first_engine = NexusEngine(
        registry=build_registry(),
        evaluation_service=(
            build_service(
                evaluation_db
            )
        ),
    )

    first_state = build_state()

    first_engine.run(
        first_state
    )

    second_engine = NexusEngine(
        registry=build_registry(),
        evaluation_service=(
            build_service(
                evaluation_db
            )
        ),
    )

    second_state = build_state()

    result = second_engine.run(
        second_state
    )

    assert (
        second_engine.last_evaluation_result
        is not None
    )

    service_result = (
        second_engine.last_evaluation_result
    )

    assert (
        service_result.benchmark
        is not None
    )

    assert (
        service_result.baseline_run_id
        == first_state.run_id
    )

    assert (
        service_result
        .benchmark
        .candidate_run_id
        == second_state.run_id
    )

    assert (
        result.metadata[
            "evaluation_benchmark"
        ]
        is not None
    )


def test_worse_second_workflow_detects_regression(
    tmp_path,
):
    evaluation_db = (
        tmp_path
        / "evaluations.db"
    )

    baseline_engine = NexusEngine(
        registry=build_registry(
            test_passed=True,
            security_risk=5,
            critic_score=96,
        ),
        evaluation_service=(
            build_service(
                evaluation_db
            )
        ),
    )

    baseline_state = build_state()

    baseline_engine.run(
        baseline_state
    )

    candidate_engine = NexusEngine(
        registry=build_registry(
            test_passed=False,
            security_risk=90,
            critic_score=55,
        ),
        evaluation_service=(
            build_service(
                evaluation_db
            )
        ),
        repair_loop=None,
    )

    candidate_state = build_state()

    result = candidate_engine.run(
        candidate_state
    )

    service_result = (
        candidate_engine
        .last_evaluation_result
    )

    assert service_result is not None

    assert (
        service_result.benchmark
        is not None
    )

    assert (
        service_result
        .benchmark
        .regression_detected
        is True
    )

    assert (
        len(
            service_result
            .benchmark
            .regressed_dimensions
        )
        > 0
    )

    assert (
        result.metadata[
            "evaluation_benchmark"
        ][
            "regression_detected"
        ]
        is True
    )


def test_evaluation_history_contains_both_runs(
    tmp_path,
):
    evaluation_db = (
        tmp_path
        / "evaluations.db"
    )

    first_engine = NexusEngine(
        registry=build_registry(),
        evaluation_service=(
            build_service(
                evaluation_db
            )
        ),
    )

    first_engine.run(
        build_state()
    )

    second_engine = NexusEngine(
        registry=build_registry(),
        evaluation_service=(
            build_service(
                evaluation_db
            )
        ),
    )

    second_engine.run(
        build_state()
    )

    history = (
        second_engine
        .evaluation_service
        .history_store
    )

    assert (
        history.count()
        == 2
    )


def test_engine_without_evaluation_preserves_old_behavior():
    engine = NexusEngine(
        registry=build_registry(),
        evaluation_service=None,
    )

    state = build_state()

    result = engine.run(
        state
    )

    assert result.completed is True

    assert (
        engine.last_evaluation_result
        is None
    )

    assert (
        "evaluation"
        not in result.metadata
    )


def test_evaluation_metadata_contains_score(
    tmp_path,
):
    engine = NexusEngine(
        registry=build_registry(),
        evaluation_service=(
            build_service(
                tmp_path
                / "evaluations.db"
            )
        ),
    )

    result = engine.run(
        build_state()
    )

    evaluation_metadata = (
        result.metadata[
            "evaluation"
        ]
    )

    assert (
        "overall_score"
        in evaluation_metadata
    )

    assert (
        0.0
        <= evaluation_metadata[
            "overall_score"
        ]
        <= 100.0
    )


def test_baseline_persists_across_engine_instances(
    tmp_path,
):
    evaluation_db = (
        tmp_path
        / "evaluations.db"
    )

    first_engine = NexusEngine(
        registry=build_registry(),
        evaluation_service=(
            build_service(
                evaluation_db
            )
        ),
    )

    first_state = build_state()

    first_engine.run(
        first_state
    )

    later_service = (
        build_service(
            evaluation_db
        )
    )

    baseline = (
        later_service.get_baseline()
    )

    assert baseline is not None

    assert (
        baseline.run_id
        == first_state.run_id
    )
