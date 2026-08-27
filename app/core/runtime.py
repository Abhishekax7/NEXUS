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

from app.approval.gate import ApprovalGate
from app.approval.manager import ApprovalManager
from app.approval.policy import ApprovalPolicy

from app.checkpointing.service import (
    CheckpointService,
)
from app.checkpointing.store import (
    CheckpointStore,
)

from app.core.engine import NexusEngine
from app.core.models import AgentRole
from app.core.plan_mutator import PlanMutator
from app.core.repair_loop import RepairLoop

from app.evaluation.benchmark import (
    BenchmarkEngine,
)
from app.evaluation.engine import (
    EvaluationEngine,
)
from app.evaluation.history import (
    EvaluationHistoryStore,
)
from app.evaluation.service import (
    EvaluationService,
)

from app.memory.manager import MemoryManager
from app.memory.retriever import MemoryRetriever
from app.memory.store import MemoryStore

from app.observability.service import (
    ObservabilityService,
)
from app.observability.store import (
    TraceStore,
)

from app.tools.executor import CommandExecutor
from app.tools.executor_runtime import ToolExecutor
from app.tools.patcher import PatchApplicator
from app.tools.production import (
    build_production_tool_registry,
)
from app.tools.registry import ToolRegistry
from app.tools.runtime import ToolRuntime
from app.tools.selector import ToolSelector
from app.tools.workspace import WorkspaceWriter


DEFAULT_WORKSPACE_ROOT = "workspace"

DEFAULT_MEMORY_DB_PATH = (
    "data/nexus_memory.db"
)

DEFAULT_EVALUATION_DB_PATH = (
    "data/nexus_evaluations.db"
)

DEFAULT_TRACE_DB_PATH = (
    "data/nexus_traces.db"
)

DEFAULT_CHECKPOINT_DB_PATH = (
    "data/nexus_checkpoints.db"
)

DEFAULT_COMMAND_TIMEOUT = 20
DEFAULT_MAX_REPAIRS = 2
DEFAULT_MAX_REPLANS = 3


def build_default_registry(
    memory_retriever: Optional[
        MemoryRetriever
    ] = None,
    tool_runtime: Optional[
        ToolRuntime
    ] = None,
) -> AgentRegistry:
    registry = AgentRegistry()

    registry.register(
        AgentRole.REQUIREMENTS,
        RequirementsAgent,
    )

    registry.register(
        AgentRole.RESEARCH,
        lambda: ResearchAgent(
            tool_runtime=tool_runtime
        ),
    )

    registry.register(
        AgentRole.ARCHITECT,
        lambda: ArchitectAgent(
            memory_retriever=(
                memory_retriever
            )
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
            memory_retriever=(
                memory_retriever
            )
        ),
    )

    return registry


def build_memory_manager(
    memory_db_path: str = (
        DEFAULT_MEMORY_DB_PATH
    ),
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
    workspace_root: str = (
        DEFAULT_WORKSPACE_ROOT
    ),
    command_timeout: int = (
        DEFAULT_COMMAND_TIMEOUT
    ),
    max_repairs: int = (
        DEFAULT_MAX_REPAIRS
    ),
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


def build_approval_manager(
    require_medium_risk_approval: bool = False,
) -> ApprovalManager:
    policy = ApprovalPolicy(
        require_medium_risk_approval=(
            require_medium_risk_approval
        )
    )

    return ApprovalManager(
        policy=policy
    )


def build_approval_gate(
    approval_manager: ApprovalManager,
) -> ApprovalGate:
    return ApprovalGate(
        manager=approval_manager
    )


def build_tool_registry(
    workspace_root: str = (
        DEFAULT_WORKSPACE_ROOT
    ),
    memory_retriever: Optional[
        MemoryRetriever
    ] = None,
) -> ToolRegistry:
    return build_production_tool_registry(
        workspace_root=workspace_root,
        memory_retriever=memory_retriever,
    )


def build_tool_runtime(
    tool_registry: ToolRegistry,
    approval_gate: Optional[
        ApprovalGate
    ] = None,
) -> ToolRuntime:
    selector = ToolSelector(
        registry=tool_registry
    )

    executor = ToolExecutor(
        registry=tool_registry
    )

    return ToolRuntime(
        selector=selector,
        executor=executor,
        approval_gate=approval_gate,
    )


def build_evaluation_service(
    evaluation_db_path: str = (
        DEFAULT_EVALUATION_DB_PATH
    ),
    auto_create_baseline: bool = True,
) -> EvaluationService:
    evaluation_engine = (
        EvaluationEngine()
    )

    history_store = (
        EvaluationHistoryStore(
            db_path=evaluation_db_path
        )
    )

    benchmark_engine = (
        BenchmarkEngine()
    )

    return EvaluationService(
        evaluation_engine=(
            evaluation_engine
        ),
        history_store=history_store,
        benchmark_engine=(
            benchmark_engine
        ),
        auto_create_baseline=(
            auto_create_baseline
        ),
    )


def build_observability_service(
    trace_db_path: str = (
        DEFAULT_TRACE_DB_PATH
    ),
) -> ObservabilityService:
    trace_store = TraceStore(
        db_path=trace_db_path
    )

    return ObservabilityService(
        store=trace_store
    )


def build_checkpoint_service(
    checkpoint_db_path: str = (
        DEFAULT_CHECKPOINT_DB_PATH
    ),
) -> CheckpointService:
    checkpoint_store = CheckpointStore(
        db_path=checkpoint_db_path
    )

    return CheckpointService(
        store=checkpoint_store
    )


def build_nexus_engine(
    workspace_root: str = (
        DEFAULT_WORKSPACE_ROOT
    ),
    memory_db_path: str = (
        DEFAULT_MEMORY_DB_PATH
    ),
    evaluation_db_path: str = (
        DEFAULT_EVALUATION_DB_PATH
    ),
    trace_db_path: str = (
        DEFAULT_TRACE_DB_PATH
    ),
    checkpoint_db_path: str = (
        DEFAULT_CHECKPOINT_DB_PATH
    ),
    command_timeout: int = (
        DEFAULT_COMMAND_TIMEOUT
    ),
    max_repairs: int = (
        DEFAULT_MAX_REPAIRS
    ),
    max_replans: int = (
        DEFAULT_MAX_REPLANS
    ),
    enable_self_healing: bool = True,
    enable_memory: bool = True,
    enable_replanning: bool = True,
    enable_tools: bool = True,
    enable_evaluation: bool = True,
    enable_observability: bool = True,
    enable_approvals: bool = True,
    enable_checkpointing: bool = True,
    require_medium_risk_approval: bool = False,
    auto_create_evaluation_baseline: bool = True,
) -> NexusEngine:
    # ---------------------------------
    # Persistent memory
    # ---------------------------------

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

    # ---------------------------------
    # Self-healing
    # ---------------------------------

    repair_loop = None

    if enable_self_healing:
        repair_loop = build_repair_loop(
            workspace_root=workspace_root,
            command_timeout=command_timeout,
            max_repairs=max_repairs,
            memory_retriever=memory_retriever,
        )

    # ---------------------------------
    # Dynamic replanning
    # ---------------------------------

    replanner = None
    plan_mutator = None

    if enable_replanning:
        replanner = build_replanner()
        plan_mutator = build_plan_mutator()

    # ---------------------------------
    # Human approval
    # ---------------------------------

    approval_manager = None
    approval_gate = None

    if enable_approvals:
        approval_manager = (
            build_approval_manager(
                require_medium_risk_approval=(
                    require_medium_risk_approval
                )
            )
        )

        approval_gate = (
            build_approval_gate(
                approval_manager
            )
        )

    # ---------------------------------
    # Tool intelligence
    # ---------------------------------

    tool_registry = None
    tool_runtime = None

    if enable_tools:
        tool_registry = (
            build_tool_registry(
                workspace_root=workspace_root,
                memory_retriever=memory_retriever,
            )
        )

        tool_runtime = (
            build_tool_runtime(
                tool_registry,
                approval_gate=approval_gate,
            )
        )

    # ---------------------------------
    # Evaluation
    # ---------------------------------

    evaluation_service = None

    if enable_evaluation:
        evaluation_service = (
            build_evaluation_service(
                evaluation_db_path=(
                    evaluation_db_path
                ),
                auto_create_baseline=(
                    auto_create_evaluation_baseline
                ),
            )
        )

    # ---------------------------------
    # Observability
    # ---------------------------------

    observability_service = None

    if enable_observability:
        observability_service = (
            build_observability_service(
                trace_db_path=trace_db_path
            )
        )

    # ---------------------------------
    # Checkpointing + recovery
    # ---------------------------------

    checkpoint_service = None

    if enable_checkpointing:
        checkpoint_service = (
            build_checkpoint_service(
                checkpoint_db_path=(
                    checkpoint_db_path
                )
            )
        )

    # ---------------------------------
    # Agents
    # ---------------------------------

    registry = build_default_registry(
        memory_retriever=memory_retriever,
        tool_runtime=tool_runtime,
    )

    # ---------------------------------
    # Engine
    # ---------------------------------

    engine = NexusEngine(
        registry=registry,
        repair_loop=repair_loop,
        memory_manager=memory_manager,
        replanner=replanner,
        plan_mutator=plan_mutator,
        max_replans=max_replans,
        evaluation_service=(
            evaluation_service
        ),
        observability_service=(
            observability_service
        ),
        checkpoint_service=(
            checkpoint_service
        ),
    )

    engine.tool_registry = (
        tool_registry
    )

    engine.tool_runtime = (
        tool_runtime
    )

    engine.approval_manager = (
        approval_manager
    )

    engine.approval_gate = (
        approval_gate
    )

    return engine
