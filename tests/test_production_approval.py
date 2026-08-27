from app.approval.gate import (
    ApprovalGate,
)
from app.approval.manager import (
    ApprovalManager,
)
from app.core.runtime import (
    build_approval_gate,
    build_approval_manager,
    build_nexus_engine,
)
from app.tools.runtime import (
    ToolRuntime,
)


def test_runtime_builds_approval_manager():
    manager = (
        build_approval_manager()
    )

    assert isinstance(
        manager,
        ApprovalManager,
    )


def test_runtime_builds_approval_gate():
    manager = (
        build_approval_manager()
    )

    gate = build_approval_gate(
        manager
    )

    assert isinstance(
        gate,
        ApprovalGate,
    )

    assert (
        gate.manager
        is manager
    )


def test_engine_has_approvals_enabled_by_default(
    tmp_path,
):
    engine = build_nexus_engine(
        workspace_root=str(
            tmp_path / "workspace"
        ),
        memory_db_path=str(
            tmp_path / "memory.db"
        ),
        evaluation_db_path=str(
            tmp_path / "evaluations.db"
        ),
        trace_db_path=str(
            tmp_path / "traces.db"
        ),
        enable_self_healing=False,
        enable_memory=False,
        enable_replanning=False,
        enable_evaluation=False,
        enable_observability=False,
        enable_tools=True,
    )

    assert (
        engine.approval_manager
        is not None
    )

    assert (
        engine.approval_gate
        is not None
    )


def test_engine_can_disable_approvals(
    tmp_path,
):
    engine = build_nexus_engine(
        workspace_root=str(
            tmp_path / "workspace"
        ),
        enable_self_healing=False,
        enable_memory=False,
        enable_replanning=False,
        enable_evaluation=False,
        enable_observability=False,
        enable_tools=True,
        enable_approvals=False,
    )

    assert (
        engine.approval_manager
        is None
    )

    assert (
        engine.approval_gate
        is None
    )

    assert (
        engine.tool_runtime
        .approval_gate
        is None
    )


def test_tool_runtime_receives_engine_gate(
    tmp_path,
):
    engine = build_nexus_engine(
        workspace_root=str(
            tmp_path / "workspace"
        ),
        enable_self_healing=False,
        enable_memory=False,
        enable_replanning=False,
        enable_evaluation=False,
        enable_observability=False,
        enable_tools=True,
        enable_approvals=True,
    )

    assert isinstance(
        engine.tool_runtime,
        ToolRuntime,
    )

    assert (
        engine.tool_runtime
        .approval_gate
        is engine.approval_gate
    )


def test_tool_runtime_uses_same_manager(
    tmp_path,
):
    engine = build_nexus_engine(
        workspace_root=str(
            tmp_path / "workspace"
        ),
        enable_self_healing=False,
        enable_memory=False,
        enable_replanning=False,
        enable_evaluation=False,
        enable_observability=False,
        enable_tools=True,
        enable_approvals=True,
    )

    assert (
        engine.tool_runtime
        .approval_gate
        .manager
        is engine.approval_manager
    )


def test_medium_risk_policy_setting_is_propagated(
    tmp_path,
):
    engine = build_nexus_engine(
        workspace_root=str(
            tmp_path / "workspace"
        ),
        enable_self_healing=False,
        enable_memory=False,
        enable_replanning=False,
        enable_evaluation=False,
        enable_observability=False,
        enable_tools=True,
        enable_approvals=True,
        require_medium_risk_approval=True,
    )

    assert (
        engine.approval_manager
        .policy
        .require_medium_risk_approval
        is True
    )


def test_approvals_can_coexist_with_all_subsystems(
    tmp_path,
):
    engine = build_nexus_engine(
        workspace_root=str(
            tmp_path / "workspace"
        ),
        memory_db_path=str(
            tmp_path / "memory.db"
        ),
        evaluation_db_path=str(
            tmp_path / "evaluations.db"
        ),
        trace_db_path=str(
            tmp_path / "traces.db"
        ),
        enable_self_healing=True,
        enable_memory=True,
        enable_replanning=True,
        enable_tools=True,
        enable_evaluation=True,
        enable_observability=True,
        enable_approvals=True,
    )

    assert (
        engine.memory_manager
        is not None
    )

    assert (
        engine.repair_loop
        is not None
    )

    assert (
        engine.replanner
        is not None
    )

    assert (
        engine.tool_runtime
        is not None
    )

    assert (
        engine.evaluation_service
        is not None
    )

    assert (
        engine.observability_service
        is not None
    )

    assert (
        engine.approval_manager
        is not None
    )

    assert (
        engine.approval_gate
        is not None
    )
