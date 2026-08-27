from app.core.runtime import (
    build_nexus_engine,
    build_observability_service,
)
from app.observability.service import (
    ObservabilityService,
)


def test_runtime_builds_observability_service(
    tmp_path,
):
    db_path = (
        tmp_path
        / "traces.db"
    )

    service = (
        build_observability_service(
            trace_db_path=str(
                db_path
            )
        )
    )

    assert isinstance(
        service,
        ObservabilityService,
    )

    assert (
        service.store.db_path
        == db_path
    )

    assert db_path.exists()


def test_production_engine_has_observability_enabled(
    tmp_path,
):
    engine = build_nexus_engine(
        workspace_root=str(
            tmp_path / "workspace"
        ),
        trace_db_path=str(
            tmp_path / "traces.db"
        ),
        enable_self_healing=False,
        enable_memory=False,
        enable_replanning=False,
        enable_tools=False,
        enable_evaluation=False,
        enable_observability=True,
    )

    assert (
        engine.observability_service
        is not None
    )


def test_production_engine_can_disable_observability(
    tmp_path,
):
    engine = build_nexus_engine(
        workspace_root=str(
            tmp_path / "workspace"
        ),
        trace_db_path=str(
            tmp_path / "traces.db"
        ),
        enable_self_healing=False,
        enable_memory=False,
        enable_replanning=False,
        enable_tools=False,
        enable_evaluation=False,
        enable_observability=False,
    )

    assert (
        engine.observability_service
        is None
    )


def test_engine_starts_without_last_trace_collector(
    tmp_path,
):
    engine = build_nexus_engine(
        workspace_root=str(
            tmp_path / "workspace"
        ),
        trace_db_path=str(
            tmp_path / "traces.db"
        ),
        enable_self_healing=False,
        enable_memory=False,
        enable_replanning=False,
        enable_tools=False,
        enable_evaluation=False,
        enable_observability=True,
    )

    assert (
        engine.last_trace_collector
        is None
    )


def test_trace_database_parent_is_created(
    tmp_path,
):
    db_path = (
        tmp_path
        / "nested"
        / "observability"
        / "traces.db"
    )

    engine = build_nexus_engine(
        workspace_root=str(
            tmp_path / "workspace"
        ),
        trace_db_path=str(
            db_path
        ),
        enable_self_healing=False,
        enable_memory=False,
        enable_replanning=False,
        enable_tools=False,
        enable_evaluation=False,
        enable_observability=True,
    )

    assert (
        engine.observability_service
        is not None
    )

    assert db_path.exists()


def test_trace_store_uses_configured_database(
    tmp_path,
):
    db_path = (
        tmp_path
        / "custom-traces.db"
    )

    engine = build_nexus_engine(
        workspace_root=str(
            tmp_path / "workspace"
        ),
        trace_db_path=str(
            db_path
        ),
        enable_self_healing=False,
        enable_memory=False,
        enable_replanning=False,
        enable_tools=False,
        enable_evaluation=False,
        enable_observability=True,
    )

    assert (
        engine.observability_service
        .store
        .db_path
        == db_path
    )


def test_observability_coexists_with_all_subsystems(
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
    )

    assert engine.repair_loop is not None

    assert (
        engine.memory_manager
        is not None
    )

    assert engine.replanner is not None

    assert (
        engine.tool_registry
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


def test_persistent_databases_are_separate(
    tmp_path,
):
    memory_path = (
        tmp_path / "memory.db"
    )

    evaluation_path = (
        tmp_path / "evaluations.db"
    )

    trace_path = (
        tmp_path / "traces.db"
    )

    engine = build_nexus_engine(
        workspace_root=str(
            tmp_path / "workspace"
        ),
        memory_db_path=str(
            memory_path
        ),
        evaluation_db_path=str(
            evaluation_path
        ),
        trace_db_path=str(
            trace_path
        ),
        enable_self_healing=False,
        enable_memory=True,
        enable_replanning=False,
        enable_tools=False,
        enable_evaluation=True,
        enable_observability=True,
    )

    assert (
        engine.memory_manager
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

    assert len(
        {
            memory_path,
            evaluation_path,
            trace_path,
        }
    ) == 3

    assert memory_path.exists()
    assert evaluation_path.exists()
    assert trace_path.exists()


def test_observability_store_starts_empty(
    tmp_path,
):
    engine = build_nexus_engine(
        workspace_root=str(
            tmp_path / "workspace"
        ),
        trace_db_path=str(
            tmp_path / "traces.db"
        ),
        enable_self_healing=False,
        enable_memory=False,
        enable_replanning=False,
        enable_tools=False,
        enable_evaluation=False,
        enable_observability=True,
    )

    assert (
        engine.observability_service
        .store
        .count()
        == 0
    )


def test_observability_persists_across_engine_instances(
    tmp_path,
):
    db_path = (
        tmp_path / "traces.db"
    )

    first = build_nexus_engine(
        workspace_root=str(
            tmp_path / "workspace-1"
        ),
        trace_db_path=str(
            db_path
        ),
        enable_self_healing=False,
        enable_memory=False,
        enable_replanning=False,
        enable_tools=False,
        enable_evaluation=False,
        enable_observability=True,
    )

    collector = (
        first.observability_service
        .start_run(
            "persistent-run",
            task_count=0,
        )
    )

    first.observability_service.complete_run(
        collector
    )

    second = build_nexus_engine(
        workspace_root=str(
            tmp_path / "workspace-2"
        ),
        trace_db_path=str(
            db_path
        ),
        enable_self_healing=False,
        enable_memory=False,
        enable_replanning=False,
        enable_tools=False,
        enable_evaluation=False,
        enable_observability=True,
    )

    trace = (
        second.observability_service
        .get_trace(
            "persistent-run"
        )
    )

    assert trace is not None

    assert (
        trace.run_id
        == "persistent-run"
    )
