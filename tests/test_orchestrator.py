from app.agents.orchestrator import OrchestratorAgent
from app.core.models import AgentRole
from app.core.scheduler import get_ready_tasks


def test_orchestrator_creates_execution_plan():
    orchestrator = OrchestratorAgent()

    state = orchestrator.create_initial_plan(
        "Build a RAG application"
    )

    assert len(state.tasks) == 7


def test_first_task_is_requirements_agent():
    orchestrator = OrchestratorAgent()

    state = orchestrator.create_initial_plan(
        "Build a RAG application"
    )

    ready_tasks = get_ready_tasks(state)

    assert len(ready_tasks) == 1
    assert ready_tasks[0].assigned_agent == AgentRole.REQUIREMENTS


def test_execution_order_contains_all_tasks():
    orchestrator = OrchestratorAgent()

    state = orchestrator.create_initial_plan(
        "Build a RAG application"
    )

    assert len(state.execution_order) == len(state.tasks)


def test_testing_and_security_run_after_implementation():
    orchestrator = OrchestratorAgent()

    state = orchestrator.create_initial_plan(
        "Build a RAG application"
    )

    coder_task = next(
        task
        for task in state.tasks.values()
        if task.assigned_agent == AgentRole.CODER
    )

    tester_task = next(
        task
        for task in state.tasks.values()
        if task.assigned_agent == AgentRole.TESTER
    )

    security_task = next(
        task
        for task in state.tasks.values()
        if task.assigned_agent == AgentRole.SECURITY
    )

    assert coder_task.id in tester_task.dependencies
    assert coder_task.id in security_task.dependencies


def test_critic_waits_for_testing_and_security():
    orchestrator = OrchestratorAgent()

    state = orchestrator.create_initial_plan(
        "Build a RAG application"
    )

    tester_task = next(
        task
        for task in state.tasks.values()
        if task.assigned_agent == AgentRole.TESTER
    )

    security_task = next(
        task
        for task in state.tasks.values()
        if task.assigned_agent == AgentRole.SECURITY
    )

    critic_task = next(
        task
        for task in state.tasks.values()
        if task.assigned_agent == AgentRole.CRITIC
    )

    assert tester_task.id in critic_task.dependencies
    assert security_task.id in critic_task.dependencies
