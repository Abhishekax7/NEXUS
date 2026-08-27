from app.agents.orchestrator import (
    OrchestratorAgent,
)

from app.core.models import (
    AgentRole,
)

from scripts.demo_nexus import (
    DEFAULT_REQUEST,
    build_demo_state,
    enum_value,
    short_id,
)


def test_default_request_is_not_empty():
    assert DEFAULT_REQUEST.strip()


def test_build_demo_state_uses_orchestrator_plan():
    state = build_demo_state(
        "Build a test service."
    )

    assert (
        state.user_request
        == "Build a test service."
    )

    assert (
        len(state.tasks)
        == 7
    )

    assert (
        len(state.execution_order)
        == 7
    )


def test_demo_plan_contains_expected_agents():
    state = build_demo_state(
        "Build a test service."
    )

    roles = {
        task.assigned_agent
        for task
        in state.tasks.values()
    }

    expected = {
        AgentRole.REQUIREMENTS,
        AgentRole.RESEARCH,
        AgentRole.ARCHITECT,
        AgentRole.CODER,
        AgentRole.TESTER,
        AgentRole.SECURITY,
        AgentRole.CRITIC,
    }

    assert expected.issubset(
        roles
    )


def test_demo_plan_contains_dependencies():
    state = build_demo_state(
        "Build a test service."
    )

    dependent_tasks = [
        task
        for task
        in state.tasks.values()
        if task.dependencies
    ]

    assert (
        len(dependent_tasks)
        >= 1
    )


def test_enum_value_handles_enum():
    assert (
        enum_value(
            AgentRole.CODER
        )
        == "coder"
    )


def test_enum_value_handles_plain_value():
    assert (
        enum_value(
            "plain"
        )
        == "plain"
    )


def test_short_id_returns_first_eight_chars():
    value = (
        "12345678-abcdefgh"
    )

    assert (
        short_id(value)
        == "12345678"
    )


def test_orchestrator_produces_requirements_first():
    state = (
        OrchestratorAgent()
        .create_initial_plan(
            "Build something."
        )
    )

    first_task = state.tasks[
        state.execution_order[0]
    ]

    assert (
        first_task.assigned_agent
        == AgentRole.REQUIREMENTS
    )
