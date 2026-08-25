from pathlib import Path

from app.agents.architect import ArchitectAgent
from app.agents.coder import CoderAgent
from app.agents.debugger import DebuggerAgent
from app.agents.placeholders import (
    CriticAgent,
    SecurityAgent,
)
from app.agents.registry import AgentRegistry
from app.agents.requirements import RequirementsAgent
from app.agents.research import ResearchAgent
from app.agents.tester import TesterAgent
from app.core.engine import NexusEngine
from app.core.models import AgentRole
from app.core.repair_loop import RepairLoop
from app.tools.executor import CommandExecutor
from app.tools.patcher import PatchApplicator
from app.tools.workspace import WorkspaceWriter


DEFAULT_WORKSPACE_ROOT = "workspace"
DEFAULT_COMMAND_TIMEOUT = 20
DEFAULT_MAX_REPAIRS = 2


def build_default_registry() -> AgentRegistry:
    """
    Build the production NEXUS agent registry.

    These are the agents used by the main execution engine.
    """

    registry = AgentRegistry()

    registry.register(
        AgentRole.REQUIREMENTS,
        RequirementsAgent,
    )

    registry.register(
        AgentRole.RESEARCH,
        ResearchAgent,
    )

    registry.register(
        AgentRole.ARCHITECT,
        ArchitectAgent,
    )

    registry.register(
        AgentRole.CODER,
        CoderAgent,
    )

    registry.register(
        AgentRole.TESTER,
        TesterAgent,
    )

    registry.register(
        AgentRole.SECURITY,
        SecurityAgent,
    )

    registry.register(
        AgentRole.CRITIC,
        CriticAgent,
    )

    return registry


def build_repair_loop(
    workspace_root: str = DEFAULT_WORKSPACE_ROOT,
    command_timeout: int = DEFAULT_COMMAND_TIMEOUT,
    max_repairs: int = DEFAULT_MAX_REPAIRS,
) -> RepairLoop:
    """
    Build the autonomous test-debug-patch-retest subsystem.
    """

    workspace_writer = WorkspaceWriter(
        root=workspace_root
    )

    executor = CommandExecutor(
        timeout_seconds=command_timeout
    )

    tester = TesterAgent(
        workspace_writer=workspace_writer,
        executor=executor,
    )

    debugger = DebuggerAgent()

    patcher = PatchApplicator(
        root=workspace_root
    )

    return RepairLoop(
        tester=tester,
        debugger=debugger,
        patcher=patcher,
        max_repairs=max_repairs,
    )


def build_nexus_engine(
    workspace_root: str = DEFAULT_WORKSPACE_ROOT,
    command_timeout: int = DEFAULT_COMMAND_TIMEOUT,
    max_repairs: int = DEFAULT_MAX_REPAIRS,
    enable_self_healing: bool = True,
) -> NexusEngine:
    """
    Assemble the production NEXUS execution engine.
    """

    registry = build_default_registry()

    repair_loop = None

    if enable_self_healing:
        repair_loop = build_repair_loop(
            workspace_root=workspace_root,
            command_timeout=command_timeout,
            max_repairs=max_repairs,
        )

    return NexusEngine(
        registry=registry,
        repair_loop=repair_loop,
    )

