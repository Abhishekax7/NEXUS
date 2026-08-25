import pytest

from app.agents.base import BaseAgent
from app.agents.orchestrator import OrchestratorAgent
from app.agents.placeholders import (
    ArchitectAgent,
    CoderAgent,
    CriticAgent,
    ResearchAgent,
    SecurityAgent,
    TesterAgent,
)
from app.agents.registry import AgentRegistry
from app.core.engine import (
    NexusEngine,
    WorkflowRepairFailed,
    WorkflowStalled,
)
from app.core.models import (
    AgentRole,
    AgentTask,
    Artifact,
    ArtifactType,
    TaskStatus,
)
from app.core.repair_loop import RepairLoopResult
from app.core.state import NexusState


class TestRequirementsAgent(BaseAgent):
    role = AgentRole.REQUIREMENTS

    def execute(
        self,
        task: AgentTask,
        state: NexusState,
    ) -> Artifact:
        return Artifact(
            type=ArtifactType.REQUIREMENTS,
            name="requirements_test_output",
            content={
                "objective": "Deterministic test objective",
            },
            created_by=self.role,
        )


class FailingTesterAgent(BaseAgent):
    role = AgentRole.TESTER

    def execute(
        self,
        task: AgentTask,
        state: NexusState,
    ) -> Artifact:
        return Artifact(
            type=ArtifactType.TEST_RESULT,
            name="failed_test_report",
            content={
                "passed": False,
                "total_commands": 1,
                "passed_commands": 0,
                "failed_commands": 1,
                "results": [],
                "failed_command_names": [
                    "pytest -v"
                ],
                "summary": "Tests failed.",
            },
            created_by=self.role,
        )


class PassingRepairLoop:
    def __init__(self):
        self.calls = 0

    def run(
        self,
        state: NexusState,
    ) -> RepairLoopResult:
        self.calls += 1

        final_artifact = Artifact(
            type=ArtifactType.TEST_RESULT,
            name="repaired_test_report",
            content={
                "passed": True,
                "total_commands": 1,
                "passed_commands": 1,
                "failed_commands": 0,
                "results": [],
                "failed_command_names": [],
                "summary": "Tests passed after repair.",
            },
            created_by=AgentRole.TESTER,
        )

        return RepairLoopResult(
            passed=True,
            attempts=1,
            final_test_artifact=final_artifact,
            debug_artifacts=[],
        )


class FailingRepairLoop:
    def __init__(self):
        self.calls = 0

    def run(
        self,
        state: NexusState,
    ) -> RepairLoopResult:
        self.calls += 1

        final_artifact = Artifact(
            type=ArtifactType.TEST_RESULT,
            name="failed_repair_test_report",
            content={
                "passed": False,
                "total_commands": 1,
                "passed_commands": 0,
                "failed_commands": 1,
                "results": [],
                "failed_command_names": [
                    "pytest -v"
                ],
                "summary": "Tests still fail after repair.",
            },
            created_by=AgentRole.TESTER,
        )

        return RepairLoopResult(
            passed=False,
            attempts=2,
            final_test_artifact=final_artifact,
            debug_artifacts=[],
        )


def build_registry():
    registry = AgentRegistry()

    registry.register(
        AgentRole.REQUIREMENTS,
        TestRequirementsAgent,
    )

    registry.register(
        AgentRole.RESEARCH,
        ResearchAgent,
    )

    registry.register(
        AgentRole.ARCHITECT,
        ArchitectAgent,
    )

    registry.register(
        AgentRole.CODER,
        CoderAgent,
    )

    registry.register(
        AgentRole.TESTER,
        TesterAgent,
    )

    registry.register(
        AgentRole.SECURITY,
        SecurityAgent,
    )

    registry.register(
        AgentRole.CRITIC,
        CriticAgent,
    )

    return registry


def build_repair_registry():
    registry = build_registry()

    registry.register(
        AgentRole.TESTER,
        FailingTesterAgent,
    )

    return registry


def test_engine_completes_full_workflow():
    orchestrator = OrchestratorAgent()

    state = orchestrator.create_initial_plan(
        "Build a RAG application"
    )

    engine = NexusEngine(
        build_registry()
    )

    result = engine.run(state)

    assert result.completed is True
    assert result.failed is False


def test_all_tasks_are_completed():
    orchestrator = OrchestratorAgent()

    state = orchestrator.create_initial_plan(
        "Build a RAG application"
    )

    engine = NexusEngine(
        build_registry()
    )

    result = engine.run(state)

    for task in result.tasks.values():
        assert task.status == TaskStatus.COMPLETED


def test_engine_generates_artifacts():
    orchestrator = OrchestratorAgent()

    state = orchestrator.create_initial_plan(
        "Build a RAG application"
    )

    engine = NexusEngine(
        build_registry()
    )

    result = engine.run(state)

    assert len(result.artifacts) == 7


def test_engine_runs_multiple_iterations():
    orchestrator = OrchestratorAgent()

    state = orchestrator.create_initial_plan(
        "Build a RAG application"
    )

    engine = NexusEngine(
        build_registry()
    )

    result = engine.run(state)

    assert result.iteration >= 5


def test_engine_detects_stalled_workflow():
    registry = AgentRegistry()

    state = NexusState(
        user_request="Build something"
    )

    blocked_task = AgentTask(
        title="Blocked task",
        description="Depends on missing task",
        assigned_agent=AgentRole.CODER,
        dependencies=[
            "missing-task-id"
        ],
    )

    state.add_task(blocked_task)

    engine = NexusEngine(registry)

    with pytest.raises(
        WorkflowStalled
    ):
        engine.run(state)

    assert state.failed is True


def test_engine_invokes_repair_loop_on_failed_tests():
    orchestrator = OrchestratorAgent()

    state = orchestrator.create_initial_plan(
        "Build application"
    )

    repair_loop = PassingRepairLoop()

    engine = NexusEngine(
        build_repair_registry(),
        repair_loop=repair_loop,
    )

    result = engine.run(state)

    assert repair_loop.calls == 1

    assert result.completed is True
    assert result.failed is False

    repaired_reports = [
        artifact
        for artifact in result.artifacts.values()
        if (
            artifact.type == ArtifactType.TEST_RESULT
            and artifact.content.get("passed") is True
        )
    ]

    assert len(repaired_reports) >= 1


def test_engine_attaches_repaired_test_artifact_to_tester_task():
    orchestrator = OrchestratorAgent()

    state = orchestrator.create_initial_plan(
        "Build application"
    )

    engine = NexusEngine(
        build_repair_registry(),
        repair_loop=PassingRepairLoop(),
    )

    result = engine.run(state)

    tester_task = next(
        task
        for task in result.tasks.values()
        if task.assigned_agent == AgentRole.TESTER
    )

    output_artifacts = [
        result.artifacts[artifact_id]
        for artifact_id
        in tester_task.output_artifact_ids
    ]

    assert any(
        artifact.type == ArtifactType.TEST_RESULT
        and artifact.content.get("passed") is True
        for artifact in output_artifacts
    )


def test_engine_fails_when_repair_budget_is_exhausted():
    orchestrator = OrchestratorAgent()

    state = orchestrator.create_initial_plan(
        "Build application"
    )

    repair_loop = FailingRepairLoop()

    engine = NexusEngine(
        build_repair_registry(),
        repair_loop=repair_loop,
    )

    with pytest.raises(
        WorkflowRepairFailed,
        match="retry budget",
    ):
        engine.run(state)

    assert repair_loop.calls == 1
    assert state.failed is True

    assert any(
        "Autonomous repair exhausted"
        in error
        for error in state.errors
    )


def test_engine_without_repair_loop_preserves_old_behavior():
    orchestrator = OrchestratorAgent()

    state = orchestrator.create_initial_plan(
        "Build application"
    )

    engine = NexusEngine(
        build_repair_registry()
    )

    result = engine.run(state)

    assert result.completed is True
    assert result.failed is False

    tester_task = next(
        task
        for task in result.tasks.values()
        if task.assigned_agent == AgentRole.TESTER
    )

    test_artifacts = [
        result.artifacts[artifact_id]
        for artifact_id
        in tester_task.output_artifact_ids
    ]

    assert any(
        artifact.type == ArtifactType.TEST_RESULT
        and artifact.content.get("passed") is False
        for artifact in test_artifacts
    )
