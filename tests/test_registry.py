import pytest

from app.agents.base import BaseAgent
from app.agents.registry import (
    AgentRegistry,
    AgentRegistryError,
)
from app.core.models import (
    AgentRole,
    AgentTask,
    Artifact,
    ArtifactType,
)
from app.core.state import NexusState


class FakeRequirementsAgent(
    BaseAgent
):
    role = AgentRole.REQUIREMENTS

    def __init__(
        self,
        marker=None,
    ):
        self.marker = marker

    def execute(
        self,
        task: AgentTask,
        state: NexusState,
    ) -> Artifact:
        return Artifact(
            type=ArtifactType.REQUIREMENTS,
            name="fake_requirements",
            content={
                "marker": self.marker
            },
            created_by=self.role,
        )


def test_agent_class_can_be_registered():
    registry = AgentRegistry()

    registry.register(
        AgentRole.REQUIREMENTS,
        FakeRequirementsAgent,
    )

    assert registry.is_registered(
        AgentRole.REQUIREMENTS
    )


def test_registered_class_can_be_resolved():
    registry = AgentRegistry()

    registry.register(
        AgentRole.REQUIREMENTS,
        FakeRequirementsAgent,
    )

    agent = registry.get_agent(
        AgentRole.REQUIREMENTS
    )

    assert isinstance(
        agent,
        FakeRequirementsAgent,
    )


def test_factory_can_be_registered():
    registry = AgentRegistry()

    registry.register(
        AgentRole.REQUIREMENTS,
        lambda: FakeRequirementsAgent(
            marker="factory"
        ),
    )

    agent = registry.get_agent(
        AgentRole.REQUIREMENTS
    )

    assert isinstance(
        agent,
        FakeRequirementsAgent,
    )

    assert (
        agent.marker
        == "factory"
    )


def test_factory_can_inject_dependency():
    registry = AgentRegistry()

    dependency = {
        "name": "memory"
    }

    registry.register(
        AgentRole.REQUIREMENTS,
        lambda: FakeRequirementsAgent(
            marker=dependency
        ),
    )

    agent = registry.get_agent(
        AgentRole.REQUIREMENTS
    )

    assert (
        agent.marker
        is dependency
    )


def test_prebuilt_agent_instance_can_be_registered():
    registry = AgentRegistry()

    instance = FakeRequirementsAgent(
        marker="instance"
    )

    registry.register(
        AgentRole.REQUIREMENTS,
        instance,
    )

    resolved = registry.get_agent(
        AgentRole.REQUIREMENTS
    )

    assert (
        resolved
        is instance
    )


def test_class_resolution_creates_agent():
    registry = AgentRegistry()

    registry.register(
        AgentRole.REQUIREMENTS,
        FakeRequirementsAgent,
    )

    first = registry.get_agent(
        AgentRole.REQUIREMENTS
    )

    second = registry.get_agent(
        AgentRole.REQUIREMENTS
    )

    assert first is not second


def test_factory_resolution_can_create_agents():
    registry = AgentRegistry()

    registry.register(
        AgentRole.REQUIREMENTS,
        lambda: FakeRequirementsAgent(),
    )

    first = registry.get_agent(
        AgentRole.REQUIREMENTS
    )

    second = registry.get_agent(
        AgentRole.REQUIREMENTS
    )

    assert first is not second


def test_prebuilt_instance_is_reused():
    registry = AgentRegistry()

    instance = FakeRequirementsAgent()

    registry.register(
        AgentRole.REQUIREMENTS,
        instance,
    )

    first = registry.get_agent(
        AgentRole.REQUIREMENTS
    )

    second = registry.get_agent(
        AgentRole.REQUIREMENTS
    )

    assert first is second


def test_unregistered_agent_raises_error():
    registry = AgentRegistry()

    with pytest.raises(
        AgentRegistryError,
        match="No agent registered",
    ):
        registry.get_agent(
            AgentRole.REQUIREMENTS
        )


def test_registry_reports_roles():
    registry = AgentRegistry()

    registry.register(
        AgentRole.REQUIREMENTS,
        FakeRequirementsAgent,
    )

    roles = registry.registered_roles()

    assert (
        AgentRole.REQUIREMENTS
        in roles
    )


def test_provider_must_produce_base_agent():
    registry = AgentRegistry()

    registry.register(
        AgentRole.REQUIREMENTS,
        lambda: "not-an-agent",
    )

    with pytest.raises(
        AgentRegistryError,
        match="did not produce",
    ):
        registry.get_agent(
            AgentRole.REQUIREMENTS
        )


def test_factory_creation_failure_is_wrapped():
    registry = AgentRegistry()

    def broken_factory():
        raise RuntimeError(
            "factory failed"
        )

    registry.register(
        AgentRole.REQUIREMENTS,
        broken_factory,
    )

    with pytest.raises(
        AgentRegistryError,
        match="Failed to create agent",
    ):
        registry.get_agent(
            AgentRole.REQUIREMENTS
        )
