from typing import Optional

from app.agents.architect import ArchitectAgent
from app.agents.coder import CoderAgent
from app.agents.critic import CriticAgent
from app.agents.debugger import DebuggerAgent
from app.agents.registry import AgentRegistry
from app.agents.replanner import ReplannerAgent
from app.agents.requirements import RequirementsAgent
from app.agents.research import ResearchAgent
from app.agents.security import SecurityAgent
from app.agents.tester import TesterAgent
from app.core.engine import NexusEngine
from app.core.models import AgentRole
from app.core.plan_mutator import PlanMutator
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
DEFAULT_MAX_REPLANS = 3


def build_default_registry(
    memory_retriever: Optional[
        MemoryRetriever
    ] = None,
) -> AgentRegistry:
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
        lambda: CriticAgent(
            memory_retriever=memory_retriever
        ),
    )

    return registry


def build_memory_manager(
    memory_db_path: str = DEFAULT_MEMORY_DB_PATH,
) -> MemoryManager:
    store = MemoryStore(
        db_path=memory_db_path
    )

    return MemoryManager(
        store=store
    )


def build_memory_retriever(
    memory_manager: MemoryManager,
) -> MemoryRetriever:
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


def build_replanner() -> ReplannerAgent:
    return ReplannerAgent()


def build_plan_mutator() -> PlanMutator:
    return PlanMutator()


def build_nexus_engine(
    workspace_root: str = DEFAULT_WORKSPACE_ROOT,
    memory_db_path: str = DEFAULT_MEMORY_DB_PATH,
    command_timeout: int = DEFAULT_COMMAND_TIMEOUT,
    max_repairs: int = DEFAULT_MAX_REPAIRS,
    max_replans: int = DEFAULT_MAX_REPLANS,
    enable_self_healing: bool = True,
    enable_memory: bool = True,
    enable_replanning: bool = True,
) -> NexusEngine:
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

    replanner = None
    plan_mutator = None

    if enable_replanning:
        replanner = build_replanner()
        plan_mutator = build_plan_mutator()

    return NexusEngine(
        registry=registry,
        repair_loop=repair_loop,
        memory_manager=memory_manager,
        replanner=replanner,
        plan_mutator=plan_mutator,
        max_replans=max_replans,
    )

