import pytest

from app.agents.base import BaseAgent
from app.agents.registry import AgentRegistry
from app.core.models import (
    AgentRole,
    AgentTask,
    Artifact,
    ArtifactType,
    TaskStatus,
)
from app.core.runner import AgentRunner
from app.core.state import NexusState


class SuccessfulAgent(BaseAgent):
    role = AgentRole.REQUIREMENTS

    def execute(
        self,
        task: AgentTask,
        state: NexusState,
    ) -> Artifact:
        return Artifact(
            type=ArtifactType.REQUIREMENTS,
            name="test_requirements",
            content={
                "objective": "Test objective",
            },
            created_by=self.role,
        )


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


def create_task() -> AgentTask:
    return AgentTask(
        title="Analyze requirements",
        description="Analyze the user request.",
        assigned_agent=AgentRole.REQUIREMENTS,
    )


def build_success_registry() -> AgentRegistry:
    registry = AgentRegistry()

    registry.register(
        AgentRole.REQUIREMENTS,
        SuccessfulAgent,
    )

    return registry


def test_runner_executes_registered_agent():
    registry = build_success_registry()

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
    assert artifact.created_by == AgentRole.REQUIREMENTS


def test_runner_marks_task_completed():
    registry = build_success_registry()

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
    registry = build_success_registry()

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
    registry = build_success_registry()

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
    assert state.active_task_id is None
    
def test_runner_attaches_dependency_artifacts():
    registry = build_success_registry()

    runner = AgentRunner(registry)

    state = NexusState(
        user_request="Build an application"
    )

    dependency_task = AgentTask(
        title="Previous task",
        description="Produce dependency artifact.",
        assigned_agent=AgentRole.REQUIREMENTS,
    )

    dependency_artifact = Artifact(
        type=ArtifactType.REQUIREMENTS,
        name="dependency_output",
        content={
            "objective": "Dependency output",
        },
        created_by=AgentRole.REQUIREMENTS,
    )

    state.add_task(
        dependency_task
    )

    state.add_artifact(
        dependency_artifact
    )

    dependency_task.output_artifact_ids.append(
        dependency_artifact.id
    )

    task = AgentTask(
        title="Dependent task",
        description="Consumes dependency output.",
        assigned_agent=AgentRole.REQUIREMENTS,
        dependencies=[
            dependency_task.id
        ],
    )

    state.add_task(task)

    runner.run_task(
        task,
        state,
    )

    assert (
        dependency_artifact.id
        in task.input_artifact_ids
    )


def test_dependency_artifact_is_not_duplicated():
    registry = build_success_registry()

    runner = AgentRunner(registry)

    state = NexusState(
        user_request="Build an application"
    )

    dependency_task = AgentTask(
        title="Previous task",
        description="Produce output.",
        assigned_agent=AgentRole.REQUIREMENTS,
    )

    dependency_artifact = Artifact(
        type=ArtifactType.REQUIREMENTS,
        name="dependency_output",
        content={
            "objective": "Dependency output",
        },
        created_by=AgentRole.REQUIREMENTS,
    )

    state.add_task(
        dependency_task
    )

    state.add_artifact(
        dependency_artifact
    )

    dependency_task.output_artifact_ids.append(
        dependency_artifact.id
    )

    task = AgentTask(
        title="Dependent task",
        description="Consume output.",
        assigned_agent=AgentRole.REQUIREMENTS,
        dependencies=[
            dependency_task.id
        ],
        input_artifact_ids=[
            dependency_artifact.id
        ],
    )

    state.add_task(task)

    runner.run_task(
        task,
        state,
    )

    assert (
        task.input_artifact_ids.count(
            dependency_artifact.id
        )
        == 1
    )


def test_runner_preserves_artifact_provenance():
    registry = build_success_registry()

    runner = AgentRunner(registry)

    state = NexusState(
        user_request="Build an application"
    )

    first_task = create_task()

    state.add_task(
        first_task
    )

    first_artifact = runner.run_task(
        first_task,
        state,
    )

    second_task = AgentTask(
        title="Consume previous output",
        description="Use previous artifact.",
        assigned_agent=AgentRole.REQUIREMENTS,
        dependencies=[
            first_task.id
        ],
    )

    state.add_task(
        second_task
    )

    runner.run_task(
        second_task,
        state,
    )

    assert (
        first_artifact.id
        in second_task.input_artifact_ids
    )
