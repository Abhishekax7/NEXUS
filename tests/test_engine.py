import pytest

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
from app.agents.requirements import RequirementsAgent
from app.core.engine import (
    NexusEngine,
    WorkflowStalled,
)
from app.core.models import (
    AgentRole,
    AgentTask,
    TaskStatus,
)
from app.core.state import NexusState


def build_registry():
    registry = AgentRegistry()

    registry.register(
        AgentRole.REQUIREMENTS,
        RequirementsAgent,
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
        dependencies=["missing-task-id"],
    )

    state.add_task(blocked_task)

    engine = NexusEngine(registry)

    with pytest.raises(WorkflowStalled):
        engine.run(state)

    assert state.failed is True
