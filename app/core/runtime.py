from typing import Optional

from app.agents.architect import ArchitectAgent
from app.agents.coder import CoderAgent
from app.agents.critic import CriticAgent
from app.agents.debugger import DebuggerAgent
from app.agents.registry import AgentRegistry
from app.agents.requirements import RequirementsAgent
from app.agents.research import ResearchAgent
from app.agents.security import SecurityAgent
from app.agents.tester import TesterAgent
from app.core.engine import NexusEngine
from app.core.models import AgentRole
from app.core.repair_loop import RepairLoop
from app.memory.manager import MemoryManager
from app.memory.retriever import MemoryRetriever
from app.memory.store import MemoryStore
from app.tools.executor import CommandExecutor
from app.tools.patcher import PatchApplicator
from app.tools.workspace import WorkspaceWriter


DEFAULT_WORKSPACE_ROOT = "workspace"
DEFAULT_MEMORY_DB_PATH = "data/nexus_memory.db"
DEFAULT_COMMAND_TIMEOUT = 20
DEFAULT_MAX_REPAIRS = 2


def build_default_registry(
    memory_retriever: Optional[
        MemoryRetriever
    ] = None,
) -> AgentRegistry:
    """
    Build the production NEXUS agent registry.

    Memory-aware agents receive the shared
    MemoryRetriever through dependency injection.
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
        lambda: ArchitectAgent(
            memory_retriever=memory_retriever
        ),
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


def build_memory_manager(
    memory_db_path: str = DEFAULT_MEMORY_DB_PATH,
) -> MemoryManager:
    """
    Build the persistent NEXUS memory subsystem.
    """

    store = MemoryStore(
        db_path=memory_db_path
    )

    return MemoryManager(
        store=store
    )


def build_memory_retriever(
    memory_manager: MemoryManager,
) -> MemoryRetriever:
    """
    Build a retriever over the same persistent
    SQLite store used by the MemoryManager.
    """

    return MemoryRetriever(
        store=memory_manager.store
    )


def build_repair_loop(
    workspace_root: str = DEFAULT_WORKSPACE_ROOT,
    command_timeout: int = DEFAULT_COMMAND_TIMEOUT,
    max_repairs: int = DEFAULT_MAX_REPAIRS,
    memory_retriever: Optional[
        MemoryRetriever
    ] = None,
) -> RepairLoop:
    """
    Build the autonomous test-debug-patch-retest
    subsystem.
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

    debugger = DebuggerAgent(
        memory_retriever=memory_retriever
    )

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
    memory_db_path: str = DEFAULT_MEMORY_DB_PATH,
    command_timeout: int = DEFAULT_COMMAND_TIMEOUT,
    max_repairs: int = DEFAULT_MAX_REPAIRS,
    enable_self_healing: bool = True,
    enable_memory: bool = True,
) -> NexusEngine:
    """
    Assemble the production NEXUS execution engine.

    With memory enabled:

    - MemoryManager persists new experience.
    - MemoryRetriever retrieves prior experience.
    - Architect receives past architectural,
      security, and critic lessons.
    - Debugger receives prior failures and repairs.
    """

    memory_manager = None
    memory_retriever = None

    if enable_memory:
        memory_manager = build_memory_manager(
            memory_db_path=memory_db_path
        )

        memory_retriever = (
            build_memory_retriever(
                memory_manager
            )
        )

    registry = build_default_registry(
        memory_retriever=memory_retriever
    )

    repair_loop = None

    if enable_self_healing:
        repair_loop = build_repair_loop(
            workspace_root=workspace_root,
            command_timeout=command_timeout,
            max_repairs=max_repairs,
            memory_retriever=memory_retriever,
        )

    return NexusEngine(
        registry=registry,
        repair_loop=repair_loop,
        memory_manager=memory_manager,
    )
