from app.agents.architect import ArchitectAgent
from app.agents.coder import CoderAgent
from app.agents.requirements import RequirementsAgent
from app.agents.research import ResearchAgent
from app.agents.tester import (
    TesterAgent as NexusTesterAgent,
)
from app.core.engine import NexusEngine
from app.core.models import AgentRole
from app.core.repair_loop import RepairLoop
from app.core.runtime import (
    build_default_registry,
    build_nexus_engine,
    build_repair_loop,
)


def test_default_registry_contains_core_agents():
    registry = build_default_registry()

    expected_roles = {
        AgentRole.REQUIREMENTS,
        AgentRole.RESEARCH,
        AgentRole.ARCHITECT,
        AgentRole.CODER,
        AgentRole.TESTER,
        AgentRole.SECURITY,
        AgentRole.CRITIC,
    }

    assert set(
        registry.registered_roles()
    ) == expected_roles


def test_registry_resolves_real_requirements_agent():
    registry = build_default_registry()

    agent = registry.get_agent(
        AgentRole.REQUIREMENTS
    )

    assert isinstance(
        agent,
        RequirementsAgent,
    )


def test_registry_resolves_real_research_agent():
    registry = build_default_registry()

    agent = registry.get_agent(
        AgentRole.RESEARCH
    )

    assert isinstance(
        agent,
        ResearchAgent,
    )


def test_registry_resolves_real_architect_agent():
    registry = build_default_registry()

    agent = registry.get_agent(
        AgentRole.ARCHITECT
    )

    assert isinstance(
        agent,
        ArchitectAgent,
    )


def test_registry_resolves_real_coder_agent():
    registry = build_default_registry()

    agent = registry.get_agent(
        AgentRole.CODER
    )

    assert isinstance(
        agent,
        CoderAgent,
    )


def test_registry_resolves_real_tester_agent():
    registry = build_default_registry()

    agent = registry.get_agent(
        AgentRole.TESTER
    )

    assert isinstance(
        agent,
        NexusTesterAgent,
    )


def test_build_repair_loop_returns_repair_loop(
    tmp_path,
):
    loop = build_repair_loop(
        workspace_root=str(tmp_path),
        command_timeout=5,
        max_repairs=3,
    )

    assert isinstance(
        loop,
        RepairLoop,
    )

    assert loop.max_repairs == 3


def test_build_engine_with_self_healing(
    tmp_path,
):
    engine = build_nexus_engine(
        workspace_root=str(tmp_path),
        enable_self_healing=True,
    )

    assert isinstance(
        engine,
        NexusEngine,
    )

    assert engine.repair_loop is not None


def test_build_engine_without_self_healing(
    tmp_path,
):
    engine = build_nexus_engine(
        workspace_root=str(tmp_path),
        enable_self_healing=False,
    )

    assert isinstance(
        engine,
        NexusEngine,
    )

    assert engine.repair_loop is None
