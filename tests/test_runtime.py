from app.agents.architect import ArchitectAgent
from app.agents.coder import CoderAgent
from app.agents.critic import CriticAgent
from app.agents.requirements import RequirementsAgent
from app.agents.research import ResearchAgent
from app.agents.security import SecurityAgent
from app.agents.tester import (
    TesterAgent as NexusTesterAgent,
)
from app.core.engine import NexusEngine
from app.core.models import AgentRole
from app.core.repair_loop import RepairLoop
from app.core.runtime import (
    build_default_registry,
    build_memory_manager,
    build_nexus_engine,
    build_repair_loop,
)
from app.memory.manager import MemoryManager


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


def test_registry_resolves_real_security_agent():
    registry = build_default_registry()

    agent = registry.get_agent(
        AgentRole.SECURITY
    )

    assert isinstance(
        agent,
        SecurityAgent,
    )


def test_registry_resolves_real_critic_agent():
    registry = build_default_registry()

    agent = registry.get_agent(
        AgentRole.CRITIC
    )

    assert isinstance(
        agent,
        CriticAgent,
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


def test_build_memory_manager(
    tmp_path,
):
    db_path = (
        tmp_path
        / "memory.db"
    )

    manager = build_memory_manager(
        memory_db_path=str(db_path)
    )

    assert isinstance(
        manager,
        MemoryManager,
    )

    assert manager.store.count() == 0


def test_build_engine_with_self_healing_and_memory(
    tmp_path,
):
    engine = build_nexus_engine(
        workspace_root=str(
            tmp_path
            / "workspace"
        ),
        memory_db_path=str(
            tmp_path
            / "memory.db"
        ),
        enable_self_healing=True,
        enable_memory=True,
    )

    assert isinstance(
        engine,
        NexusEngine,
    )

    assert engine.repair_loop is not None
    assert engine.memory_manager is not None


def test_build_engine_without_self_healing(
    tmp_path,
):
    engine = build_nexus_engine(
        workspace_root=str(
            tmp_path
            / "workspace"
        ),
        memory_db_path=str(
            tmp_path
            / "memory.db"
        ),
        enable_self_healing=False,
        enable_memory=True,
    )

    assert engine.repair_loop is None

    assert engine.memory_manager is not None


def test_build_engine_without_memory(
    tmp_path,
):
    engine = build_nexus_engine(
        workspace_root=str(
            tmp_path
            / "workspace"
        ),
        memory_db_path=str(
            tmp_path
            / "memory.db"
        ),
        enable_self_healing=True,
        enable_memory=False,
    )

    assert engine.repair_loop is not None

    assert engine.memory_manager is None


def test_build_engine_without_optional_subsystems(
    tmp_path,
):
    engine = build_nexus_engine(
        workspace_root=str(
            tmp_path
            / "workspace"
        ),
        memory_db_path=str(
            tmp_path
            / "memory.db"
        ),
        enable_self_healing=False,
        enable_memory=False,
    )

    assert engine.repair_loop is None
    assert engine.memory_manager is None
