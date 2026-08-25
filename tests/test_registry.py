import pytest

from app.agents.registry import (
    AgentNotRegistered,
    AgentRegistry,
)
from app.agents.requirements import RequirementsAgent
from app.core.models import AgentRole


def test_agent_can_be_registered():
    registry = AgentRegistry()

    registry.register(
        AgentRole.REQUIREMENTS,
        RequirementsAgent,
    )

    assert registry.is_registered(
        AgentRole.REQUIREMENTS
    )


def test_registered_agent_can_be_resolved():
    registry = AgentRegistry()

    registry.register(
        AgentRole.REQUIREMENTS,
        RequirementsAgent,
    )

    agent = registry.get_agent(
        AgentRole.REQUIREMENTS
    )

    assert isinstance(
        agent,
        RequirementsAgent,
    )


def test_unregistered_agent_raises_error():
    registry = AgentRegistry()

    with pytest.raises(AgentNotRegistered):
        registry.get_agent(
            AgentRole.TESTER
        )


def test_registry_reports_roles():
    registry = AgentRegistry()

    registry.register(
        AgentRole.REQUIREMENTS,
        RequirementsAgent,
    )

    roles = registry.registered_roles()

    assert AgentRole.REQUIREMENTS in roles
