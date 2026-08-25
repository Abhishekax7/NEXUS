from app.core.execution import complete_task, start_task
from app.core.models import AgentRole, AgentTask
from app.core.scheduler import (
    all_tasks_completed,
    dependencies_completed,
    get_blocked_tasks,
    get_ready_tasks,
    has_unfinished_tasks,
)
from app.core.state import NexusState


def build_state():
    state = NexusState(
        user_request="Build an AI application"
    )

    requirements = AgentTask(
        title="Analyze requirements",
        description="Analyze the user request.",
        assigned_agent=AgentRole.REQUIREMENTS,
    )

    architecture = AgentTask(
        title="Design architecture",
        description="Design the system.",
        assigned_agent=AgentRole.ARCHITECT,
        dependencies=[requirements.id],
    )

    coder = AgentTask(
        title="Implement application",
        description="Write the application code.",
        assigned_agent=AgentRole.CODER,
        dependencies=[architecture.id],
    )

    state.add_task(requirements)
    state.add_task(architecture)
    state.add_task(coder)

    return state, requirements, architecture, coder


def test_task_without_dependencies_is_ready():
    state, requirements, _, _ = build_state()

    assert dependencies_completed(requirements, state)


def test_dependency_prevents_task_from_running():
    state, _, architecture, _ = build_state()

    assert dependencies_completed(
        architecture,
        state,
    ) is False


def test_only_root_task_is_initially_ready():
    state, requirements, _, _ = build_state()

    ready = get_ready_tasks(state)

    assert len(ready) == 1
    assert ready[0].id == requirements.id


def test_architecture_becomes_ready_after_requirements():
    state, requirements, architecture, _ = build_state()

    start_task(requirements)
    complete_task(requirements)

    ready = get_ready_tasks(state)

    assert len(ready) == 1
    assert ready[0].id == architecture.id


def test_coder_remains_blocked_until_architecture_finishes():
    state, requirements, _, coder = build_state()

    start_task(requirements)
    complete_task(requirements)

    blocked = get_blocked_tasks(state)

    blocked_ids = [task.id for task in blocked]

    assert coder.id in blocked_ids


def test_state_reports_unfinished_tasks():
    state, _, _, _ = build_state()

    assert has_unfinished_tasks(state)


def test_all_tasks_completed():
    state, requirements, architecture, coder = build_state()

    for task in [
        requirements,
        architecture,
        coder,
    ]:
        start_task(task)
        complete_task(task)

    assert all_tasks_completed(state)
