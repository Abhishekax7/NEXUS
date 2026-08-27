import pytest

from app.core.models import (
    AgentRole,
    AgentTask,
    Artifact,
    ArtifactType,
    TaskStatus,
)
from app.core.state import NexusState
from app.evaluation.engine import (
    EvaluationEngine,
)
from app.evaluation.models import (
    EvaluationDimension,
    EvaluationStatus,
)


def build_complete_state():
    state = NexusState(
        user_request=(
            "Build a secure FastAPI application."
        )
    )

    requirements_task = AgentTask(
        title="Analyze requirements",
        description="Analyze the request.",
        assigned_agent=AgentRole.REQUIREMENTS,
        status=TaskStatus.COMPLETED,
    )

    architect_task = AgentTask(
        title="Design architecture",
        description="Design the system.",
        assigned_agent=AgentRole.ARCHITECT,
        status=TaskStatus.COMPLETED,
        dependencies=[
            requirements_task.id
        ],
    )

    coder_task = AgentTask(
        title="Implement application",
        description="Generate code.",
        assigned_agent=AgentRole.CODER,
        status=TaskStatus.COMPLETED,
        dependencies=[
            architect_task.id
        ],
    )

    tester_task = AgentTask(
        title="Test application",
        description="Run tests.",
        assigned_agent=AgentRole.TESTER,
        status=TaskStatus.COMPLETED,
        dependencies=[
            coder_task.id
        ],
    )

    security_task = AgentTask(
        title="Security review",
        description="Review security.",
        assigned_agent=AgentRole.SECURITY,
        status=TaskStatus.COMPLETED,
        dependencies=[
            coder_task.id
        ],
    )

    critic_task = AgentTask(
        title="Final quality gate",
        description="Evaluate workflow.",
        assigned_agent=AgentRole.CRITIC,
        status=TaskStatus.COMPLETED,
        dependencies=[
            tester_task.id,
            security_task.id,
        ],
    )

    for task in [
        requirements_task,
        architect_task,
        coder_task,
        tester_task,
        security_task,
        critic_task,
    ]:
        state.add_task(
            task
        )

    requirements = Artifact(
        type=ArtifactType.REQUIREMENTS,
        name="requirements",
        content={
            "objective": (
                "Build a secure FastAPI application."
            ),
        },
        created_by=AgentRole.REQUIREMENTS,
    )

    architecture = Artifact(
        type=ArtifactType.ARCHITECTURE,
        name="architecture",
        content={
            "architecture_style": (
                "Modular architecture"
            ),
        },
        created_by=AgentRole.ARCHITECT,
        metadata={
            "grounded_in_requirements": True,
            "grounded_in_research": True,
        },
    )

    code = Artifact(
        type=ArtifactType.CODE,
        name="code",
        content={
            "files": [
                {
                    "path": "app.py",
                    "content": "print('ok')",
                }
            ],
        },
        created_by=AgentRole.CODER,
    )

    tests = Artifact(
        type=ArtifactType.TEST_RESULT,
        name="tests",
        content={
            "passed": True,
            "summary": "All tests passed.",
        },
        created_by=AgentRole.TESTER,
    )

    security = Artifact(
        type=ArtifactType.SECURITY_REPORT,
        name="security",
        content={
            "passed": True,
            "risk_score": 10,
            "summary": (
                "No significant security "
                "issues found."
            ),
        },
        created_by=AgentRole.SECURITY,
    )

    evaluation = Artifact(
        type=ArtifactType.EVALUATION,
        name="critic_evaluation",
        content={
            "verdict": "accept",
            "quality_score": 94,
        },
        created_by=AgentRole.CRITIC,
    )

    research = Artifact(
        type=ArtifactType.RESEARCH,
        name="research",
        content={
            "findings": [
                "FastAPI is suitable."
            ],
        },
        created_by=AgentRole.RESEARCH,
        metadata={
            "dynamic_tools_enabled": True,
            "dynamic_tool_used": True,
            "dynamic_tool_name": "search_memory",
            "dynamic_tool_success": True,
        },
    )

    for artifact in [
        requirements,
        architecture,
        code,
        tests,
        security,
        evaluation,
        research,
    ]:
        state.add_artifact(
            artifact
        )

    state.completed = True
    state.failed = False
    state.errors = []

    return state


def metric_by_dimension(
    report,
    dimension,
):
    return next(
        metric
        for metric in report.metrics
        if metric.dimension == dimension
    )


def test_evaluation_engine_returns_workflow_report():
    state = build_complete_state()

    engine = EvaluationEngine()

    report = engine.evaluate(
        state
    )

    assert (
        report.run_id
        == state.run_id
    )

    assert (
        0.0
        <= report.overall_score
        <= 100.0
    )

    assert len(
        report.metrics
    ) == 10


def test_complete_workflow_scores_highly():
    state = build_complete_state()

    report = EvaluationEngine().evaluate(
        state
    )

    assert (
        report.overall_score
        >= 80.0
    )

    assert (
        report.status
        == EvaluationStatus.PASS
    )


def test_all_completed_tasks_score_full_completion():
    state = build_complete_state()

    report = EvaluationEngine().evaluate(
        state
    )

    metric = metric_by_dimension(
        report,
        EvaluationDimension.TASK_COMPLETION,
    )

    assert metric.score == 100.0

    assert (
        metric.status
        == EvaluationStatus.PASS
    )


def test_partial_task_completion_reduces_score():
    state = build_complete_state()

    coder_task = next(
        task
        for task in state.tasks.values()
        if (
            task.assigned_agent
            == AgentRole.CODER
        )
    )

    coder_task.status = (
        TaskStatus.PENDING
    )

    report = EvaluationEngine().evaluate(
        state
    )

    metric = metric_by_dimension(
        report,
        EvaluationDimension.TASK_COMPLETION,
    )

    assert metric.score < 100.0


def test_empty_artifact_reduces_artifact_quality():
    state = build_complete_state()

    empty = Artifact(
        type=ArtifactType.CODE,
        name="empty_code",
        content={},
        created_by=AgentRole.CODER,
    )

    state.add_artifact(
        empty
    )

    report = EvaluationEngine().evaluate(
        state
    )

    metric = metric_by_dimension(
        report,
        EvaluationDimension.ARTIFACT_QUALITY,
    )

    assert metric.score < 100.0


def test_grounded_artifact_scores_full_grounding():
    state = build_complete_state()

    report = EvaluationEngine().evaluate(
        state
    )

    metric = metric_by_dimension(
        report,
        EvaluationDimension.GROUNDING,
    )

    assert metric.score == 100.0


def test_failed_grounding_reduces_score():
    state = build_complete_state()

    architecture = next(
        artifact
        for artifact in state.artifacts.values()
        if (
            artifact.type
            == ArtifactType.ARCHITECTURE
        )
    )

    architecture.metadata[
        "grounded_in_research"
    ] = False

    report = EvaluationEngine().evaluate(
        state
    )

    metric = metric_by_dimension(
        report,
        EvaluationDimension.GROUNDING,
    )

    assert metric.score == 0.0


def test_missing_grounding_metadata_uses_neutral_score():
    state = build_complete_state()

    for artifact in state.artifacts.values():
        artifact.metadata = {
            key: value
            for key, value
            in artifact.metadata.items()
            if not key.startswith(
                "grounded_in_"
            )
        }

    report = EvaluationEngine().evaluate(
        state
    )

    metric = metric_by_dimension(
        report,
        EvaluationDimension.GROUNDING,
    )

    assert metric.score == 70.0

    assert (
        metric.status
        == EvaluationStatus.WARN
    )


def test_passing_tests_score_full_test_quality():
    state = build_complete_state()

    report = EvaluationEngine().evaluate(
        state
    )

    metric = metric_by_dimension(
        report,
        EvaluationDimension.TEST_QUALITY,
    )

    assert metric.score == 100.0


def test_failing_tests_score_low():
    state = build_complete_state()

    test_artifact = next(
        artifact
        for artifact in state.artifacts.values()
        if (
            artifact.type
            == ArtifactType.TEST_RESULT
        )
    )

    test_artifact.content[
        "passed"
    ] = False

    test_artifact.content[
        "summary"
    ] = "Tests failed."

    report = EvaluationEngine().evaluate(
        state
    )

    metric = metric_by_dimension(
        report,
        EvaluationDimension.TEST_QUALITY,
    )

    assert metric.score == 20.0

    assert (
        metric.status
        == EvaluationStatus.FAIL
    )


def test_missing_test_artifact_scores_zero():
    state = build_complete_state()

    test_ids = [
        artifact_id
        for artifact_id, artifact
        in state.artifacts.items()
        if (
            artifact.type
            == ArtifactType.TEST_RESULT
        )
    ]

    for artifact_id in test_ids:
        del state.artifacts[
            artifact_id
        ]

    report = EvaluationEngine().evaluate(
        state
    )

    metric = metric_by_dimension(
        report,
        EvaluationDimension.TEST_QUALITY,
    )

    assert metric.score == 0.0


def test_security_score_uses_inverse_risk():
    state = build_complete_state()

    report = EvaluationEngine().evaluate(
        state
    )

    metric = metric_by_dimension(
        report,
        EvaluationDimension.SECURITY,
    )

    assert metric.score == 90.0

    assert (
        metric.status
        == EvaluationStatus.PASS
    )


def test_high_security_risk_reduces_score():
    state = build_complete_state()

    security = next(
        artifact
        for artifact in state.artifacts.values()
        if (
            artifact.type
            == ArtifactType.SECURITY_REPORT
        )
    )

    security.content[
        "risk_score"
    ] = 80

    security.content[
        "passed"
    ] = False

    report = EvaluationEngine().evaluate(
        state
    )

    metric = metric_by_dimension(
        report,
        EvaluationDimension.SECURITY,
    )

    assert metric.score == 20.0

    assert (
        metric.status
        == EvaluationStatus.FAIL
    )


def test_no_repairs_scores_full_efficiency():
    state = build_complete_state()

    report = EvaluationEngine().evaluate(
        state
    )

    metric = metric_by_dimension(
        report,
        EvaluationDimension.REPAIR_EFFICIENCY,
    )

    assert metric.score == 100.0


def test_debug_artifacts_reduce_repair_efficiency():
    state = build_complete_state()

    debug = Artifact(
        type=ArtifactType.DEBUG_REPORT,
        name="repair",
        content={
            "root_cause": "Bug",
        },
        created_by=AgentRole.DEBUGGER,
    )

    state.add_artifact(
        debug
    )

    report = EvaluationEngine().evaluate(
        state
    )

    metric = metric_by_dimension(
        report,
        EvaluationDimension.REPAIR_EFFICIENCY,
    )

    assert metric.score == 80.0


def test_no_replans_scores_full_efficiency():
    state = build_complete_state()

    report = EvaluationEngine().evaluate(
        state
    )

    metric = metric_by_dimension(
        report,
        EvaluationDimension.REPLANNING_EFFICIENCY,
    )

    assert metric.score == 100.0


def test_replans_reduce_replanning_efficiency():
    state = build_complete_state()

    state.metadata[
        "replan_count"
    ] = 2

    report = EvaluationEngine().evaluate(
        state
    )

    metric = metric_by_dimension(
        report,
        EvaluationDimension.REPLANNING_EFFICIENCY,
    )

    assert metric.score == 70.0


def test_successful_dynamic_tool_use_scores_full():
    state = build_complete_state()

    report = EvaluationEngine().evaluate(
        state
    )

    metric = metric_by_dimension(
        report,
        EvaluationDimension.TOOL_USE,
    )

    assert metric.score == 100.0


def test_failed_dynamic_tool_use_reduces_score():
    state = build_complete_state()

    research = next(
        artifact
        for artifact in state.artifacts.values()
        if (
            artifact.type
            == ArtifactType.RESEARCH
        )
    )

    research.metadata[
        "dynamic_tool_success"
    ] = False

    report = EvaluationEngine().evaluate(
        state
    )

    metric = metric_by_dimension(
        report,
        EvaluationDimension.TOOL_USE,
    )

    assert metric.score == 0.0


def test_no_tool_metadata_uses_neutral_score():
    state = build_complete_state()

    for artifact in state.artifacts.values():
        artifact.metadata.pop(
            "dynamic_tools_enabled",
            None,
        )

    report = EvaluationEngine().evaluate(
        state
    )

    metric = metric_by_dimension(
        report,
        EvaluationDimension.TOOL_USE,
    )

    assert metric.score == 75.0

    assert (
        metric.status
        == EvaluationStatus.WARN
    )


def test_critic_quality_uses_quality_score():
    state = build_complete_state()

    report = EvaluationEngine().evaluate(
        state
    )

    metric = metric_by_dimension(
        report,
        EvaluationDimension.CRITIC_QUALITY,
    )

    assert metric.score == 94.0


def test_missing_critic_evaluation_scores_zero():
    state = build_complete_state()

    evaluation_ids = [
        artifact_id
        for artifact_id, artifact
        in state.artifacts.items()
        if (
            artifact.type
            == ArtifactType.EVALUATION
        )
    ]

    for artifact_id in evaluation_ids:
        del state.artifacts[
            artifact_id
        ]

    report = EvaluationEngine().evaluate(
        state
    )

    metric = metric_by_dimension(
        report,
        EvaluationDimension.CRITIC_QUALITY,
    )

    assert metric.score == 0.0


def test_completed_reliable_workflow_scores_full():
    state = build_complete_state()

    report = EvaluationEngine().evaluate(
        state
    )

    metric = metric_by_dimension(
        report,
        EvaluationDimension.WORKFLOW_RELIABILITY,
    )

    assert metric.score == 100.0


def test_failed_workflow_scores_low_reliability():
    state = build_complete_state()

    state.completed = False
    state.failed = True
    state.errors.append(
        "Workflow failed."
    )

    report = EvaluationEngine().evaluate(
        state
    )

    metric = metric_by_dimension(
        report,
        EvaluationDimension.WORKFLOW_RELIABILITY,
    )

    assert metric.score == 20.0

    assert (
        metric.status
        == EvaluationStatus.FAIL
    )


def test_agent_evaluations_are_generated():
    state = build_complete_state()

    report = EvaluationEngine().evaluate(
        state
    )

    roles = {
        evaluation.agent_role
        for evaluation
        in report.agent_evaluations
    }

    assert (
        AgentRole.REQUIREMENTS.value
        in roles
    )

    assert (
        AgentRole.ARCHITECT.value
        in roles
    )

    assert (
        AgentRole.CODER.value
        in roles
    )

    assert (
        AgentRole.TESTER.value
        in roles
    )


def test_completed_agent_scores_highly():
    state = build_complete_state()

    report = EvaluationEngine().evaluate(
        state
    )

    coder = next(
        evaluation
        for evaluation
        in report.agent_evaluations
        if (
            evaluation.agent_role
            == AgentRole.CODER.value
        )
    )

    assert coder.score == 100.0

    assert len(
        coder.strengths
    ) > 0


def test_failed_metric_creates_recommendation():
    state = build_complete_state()

    security = next(
        artifact
        for artifact in state.artifacts.values()
        if (
            artifact.type
            == ArtifactType.SECURITY_REPORT
        )
    )

    security.content[
        "risk_score"
    ] = 95

    security.content[
        "passed"
    ] = False

    report = EvaluationEngine().evaluate(
        state
    )

    assert any(
        "security"
        in recommendation.lower()
        for recommendation
        in report.recommendations
    )


def test_regression_risk_increases_with_failures():
    good_state = build_complete_state()

    good_report = (
        EvaluationEngine().evaluate(
            good_state
        )
    )

    bad_state = build_complete_state()

    bad_state.completed = False
    bad_state.failed = True
    bad_state.errors.append(
        "failure"
    )

    test_artifact = next(
        artifact
        for artifact
        in bad_state.artifacts.values()
        if (
            artifact.type
            == ArtifactType.TEST_RESULT
        )
    )

    test_artifact.content[
        "passed"
    ] = False

    security = next(
        artifact
        for artifact
        in bad_state.artifacts.values()
        if (
            artifact.type
            == ArtifactType.SECURITY_REPORT
        )
    )

    security.content[
        "risk_score"
    ] = 90

    security.content[
        "passed"
    ] = False

    bad_report = (
        EvaluationEngine().evaluate(
            bad_state
        )
    )

    assert (
        bad_report.regression_risk
        > good_report.regression_risk
    )


def test_thresholds_are_configurable():
    engine = EvaluationEngine(
        pass_threshold=90.0,
        warn_threshold=70.0,
    )

    assert (
        engine.pass_threshold
        == 90.0
    )

    assert (
        engine.warn_threshold
        == 70.0
    )


def test_invalid_thresholds_are_rejected():
    with pytest.raises(
        ValueError,
        match="Thresholds",
    ):
        EvaluationEngine(
            pass_threshold=60.0,
            warn_threshold=80.0,
        )


def test_empty_state_can_still_be_evaluated():
    state = NexusState(
        user_request="Empty workflow"
    )

    report = EvaluationEngine().evaluate(
        state
    )

    assert (
        report.overall_score
        < 80.0
    )

    assert (
        report.status
        != EvaluationStatus.PASS
    )
