import pytest

from app.agents.base import BaseAgent
from app.agents.registry import AgentRegistry
from app.agents.requirements import RequirementsAgent
from app.core.models import (
    AgentRole,
    AgentTask,
    Artifact,
    ArtifactType,
    TaskStatus,
)
from app.core.runner import AgentRunner
from app.core.state import NexusState


class FailingAgent(BaseAgent):
    role = AgentRole.REQUIREMENTS

    def execute(
        self,
        task: AgentTask,
        state: NexusState,
    ) -> Artifact:
        raise RuntimeError(
            "Simulated agent failure"
        )


def create_task():
    return AgentTask(
        title="Analyze requirements",
        description="Analyze the user request.",
        assigned_agent=AgentRole.REQUIREMENTS,
    )


def test_runner_executes_registered_agent():
    registry = AgentRegistry()

    registry.register(
        AgentRole.REQUIREMENTS,
        RequirementsAgent,
    )

    runner = AgentRunner(registry)

    state = NexusState(
        user_request="Build a RAG application"
    )

    task = create_task()

    state.add_task(task)

    artifact = runner.run_task(
        task,
        state,
    )

    assert artifact.type == ArtifactType.REQUIREMENTS


def test_runner_marks_task_completed():
    registry = AgentRegistry()

    registry.register(
        AgentRole.REQUIREMENTS,
        RequirementsAgent,
    )

    runner = AgentRunner(registry)

    state = NexusState(
        user_request="Build a RAG application"
    )

    task = create_task()

    state.add_task(task)

    runner.run_task(
        task,
        state,
    )

    assert task.status == TaskStatus.COMPLETED


def test_runner_stores_artifact():
    registry = AgentRegistry()

    registry.register(
        AgentRole.REQUIREMENTS,
        RequirementsAgent,
    )

    runner = AgentRunner(registry)

    state = NexusState(
        user_request="Build a RAG application"
    )

    task = create_task()

    state.add_task(task)

    artifact = runner.run_task(
        task,
        state,
    )

    assert artifact.id in state.artifacts
    assert artifact.id in task.output_artifact_ids


def test_runner_clears_active_task():
    registry = AgentRegistry()

    registry.register(
        AgentRole.REQUIREMENTS,
        RequirementsAgent,
    )

    runner = AgentRunner(registry)

    state = NexusState(
        user_request="Build a RAG application"
    )

    task = create_task()

    state.add_task(task)

    runner.run_task(
        task,
        state,
    )

    assert state.active_task_id is None


def test_runner_marks_failed_agent():
    registry = AgentRegistry()

    registry.register(
        AgentRole.REQUIREMENTS,
        FailingAgent,
    )

    runner = AgentRunner(registry)

    state = NexusState(
        user_request="Build something"
    )

    task = create_task()

    state.add_task(task)

    with pytest.raises(
        RuntimeError,
        match="Simulated agent failure",
    ):
        runner.run_task(
            task,
            state,
        )

    assert task.status == TaskStatus.FAILED
    assert task.error == "Simulated agent failure"
    assert len(state.errors) == 1



