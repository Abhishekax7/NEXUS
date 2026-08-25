from app.agents.orchestrator import (
    OrchestratorAgent,
)
from app.core.models import AgentRole
from app.core.scheduler import (
    get_ready_tasks,
)


def build_state():
    orchestrator = OrchestratorAgent()

    return orchestrator.create_initial_plan(
        "Build a RAG application"
    )


def get_task_by_role(
    state,
    role,
):
    return next(
        task
        for task in state.tasks.values()
        if task.assigned_agent == role
    )


def test_orchestrator_creates_execution_plan():
    state = build_state()

    assert len(state.tasks) == 7


def test_first_task_is_requirements_agent():
    state = build_state()

    ready_tasks = get_ready_tasks(
        state
    )

    assert len(ready_tasks) == 1

    assert (
        ready_tasks[0].assigned_agent
        == AgentRole.REQUIREMENTS
    )


def test_execution_order_contains_all_tasks():
    state = build_state()

    assert (
        len(state.execution_order)
        == len(state.tasks)
    )


def test_research_depends_on_requirements():
    state = build_state()

    requirements_task = get_task_by_role(
        state,
        AgentRole.REQUIREMENTS,
    )

    research_task = get_task_by_role(
        state,
        AgentRole.RESEARCH,
    )

    assert (
        requirements_task.id
        in research_task.dependencies
    )


def test_architect_depends_on_requirements_and_research():
    state = build_state()

    requirements_task = get_task_by_role(
        state,
        AgentRole.REQUIREMENTS,
    )

    research_task = get_task_by_role(
        state,
        AgentRole.RESEARCH,
    )

    architecture_task = get_task_by_role(
        state,
        AgentRole.ARCHITECT,
    )

    assert (
        requirements_task.id
        in architecture_task.dependencies
    )

    assert (
        research_task.id
        in architecture_task.dependencies
    )


def test_testing_and_security_run_after_implementation():
    state = build_state()

    coder_task = get_task_by_role(
        state,
        AgentRole.CODER,
    )

    tester_task = get_task_by_role(
        state,
        AgentRole.TESTER,
    )

    security_task = get_task_by_role(
        state,
        AgentRole.SECURITY,
    )

    assert (
        coder_task.id
        in tester_task.dependencies
    )

    assert (
        coder_task.id
        in security_task.dependencies
    )


def test_critic_waits_for_testing_and_security():
    state = build_state()

    tester_task = get_task_by_role(
        state,
        AgentRole.TESTER,
    )

    security_task = get_task_by_role(
        state,
        AgentRole.SECURITY,
    )

    critic_task = get_task_by_role(
        state,
        AgentRole.CRITIC,
    )

    assert (
        tester_task.id
        in critic_task.dependencies
    )

    assert (
        security_task.id
        in critic_task.dependencies
    )
