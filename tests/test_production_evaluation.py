from app.core.runtime import (
    build_evaluation_service,
    build_nexus_engine,
)
from app.evaluation.service import (
    EvaluationService,
)


def test_runtime_builds_evaluation_service(
    tmp_path,
):
    service = build_evaluation_service(
        evaluation_db_path=str(
            tmp_path
            / "evaluations.db"
        )
    )

    assert isinstance(
        service,
        EvaluationService,
    )

    assert (
        service.history_store.db_path
        == (
            tmp_path
            / "evaluations.db"
        )
    )


def test_production_engine_has_evaluation_enabled(
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
        evaluation_db_path=str(
            tmp_path
            / "evaluations.db"
        ),
        enable_self_healing=False,
        enable_memory=False,
        enable_replanning=False,
        enable_tools=False,
        enable_evaluation=True,
    )

    assert (
        engine.evaluation_service
        is not None
    )


def test_production_engine_can_disable_evaluation(
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
        evaluation_db_path=str(
            tmp_path
            / "evaluations.db"
        ),
        enable_self_healing=False,
        enable_memory=False,
        enable_replanning=False,
        enable_tools=False,
        enable_evaluation=False,
    )

    assert (
        engine.evaluation_service
        is None
    )


def test_engine_starts_without_last_evaluation(
    tmp_path,
):
    engine = build_nexus_engine(
        workspace_root=str(
            tmp_path
            / "workspace"
        ),
        evaluation_db_path=str(
            tmp_path
            / "evaluations.db"
        ),
        enable_self_healing=False,
        enable_memory=False,
        enable_replanning=False,
        enable_tools=False,
        enable_evaluation=True,
    )

    assert (
        engine.last_evaluation_result
        is None
    )


def test_evaluation_database_is_created(
    tmp_path,
):
    db_path = (
        tmp_path
        / "nested"
        / "evaluation"
        / "nexus.db"
    )

    engine = build_nexus_engine(
        workspace_root=str(
            tmp_path
            / "workspace"
        ),
        evaluation_db_path=str(
            db_path
        ),
        enable_self_healing=False,
        enable_memory=False,
        enable_replanning=False,
        enable_tools=False,
        enable_evaluation=True,
    )

    assert (
        engine.evaluation_service
        is not None
    )

    assert db_path.exists()


def test_engine_exposes_same_evaluation_service(
    tmp_path,
):
    engine = build_nexus_engine(
        workspace_root=str(
            tmp_path
            / "workspace"
        ),
        evaluation_db_path=str(
            tmp_path
            / "evaluations.db"
        ),
        enable_self_healing=False,
        enable_memory=False,
        enable_replanning=False,
        enable_tools=False,
        enable_evaluation=True,
    )

    service = (
        engine.evaluation_service
    )

    assert service is not None

    assert (
        service.history_store
        is not None
    )

    assert (
        service.evaluation_engine
        is not None
    )

    assert (
        service.benchmark_engine
        is not None
    )


def test_runtime_passes_auto_baseline_setting(
    tmp_path,
):
    engine = build_nexus_engine(
        workspace_root=str(
            tmp_path
            / "workspace"
        ),
        evaluation_db_path=str(
            tmp_path
            / "evaluations.db"
        ),
        enable_self_healing=False,
        enable_memory=False,
        enable_replanning=False,
        enable_tools=False,
        enable_evaluation=True,
        auto_create_evaluation_baseline=False,
    )

    assert (
        engine.evaluation_service
        is not None
    )

    assert (
        engine.evaluation_service
        .auto_create_baseline
        is False
    )


def test_evaluation_history_persists_between_engines(
    tmp_path,
):
    db_path = (
        tmp_path
        / "evaluations.db"
    )

    first = build_nexus_engine(
        workspace_root=str(
            tmp_path
            / "workspace-1"
        ),
        evaluation_db_path=str(
            db_path
        ),
        enable_self_healing=False,
        enable_memory=False,
        enable_replanning=False,
        enable_tools=False,
        enable_evaluation=True,
    )

    second = build_nexus_engine(
        workspace_root=str(
            tmp_path
            / "workspace-2"
        ),
        evaluation_db_path=str(
            db_path
        ),
        enable_self_healing=False,
        enable_memory=False,
        enable_replanning=False,
        enable_tools=False,
        enable_evaluation=True,
    )

    assert (
        first.evaluation_service
        is not None
    )

    assert (
        second.evaluation_service
        is not None
    )

    assert (
        first.evaluation_service
        .history_store.db_path
        ==
        second.evaluation_service
        .history_store.db_path
    )


def test_evaluation_and_memory_databases_are_separate(
    tmp_path,
):
    memory_path = (
        tmp_path
        / "memory.db"
    )

    evaluation_path = (
        tmp_path
        / "evaluations.db"
    )

    engine = build_nexus_engine(
        workspace_root=str(
            tmp_path
            / "workspace"
        ),
        memory_db_path=str(
            memory_path
        ),
        evaluation_db_path=str(
            evaluation_path
        ),
        enable_self_healing=False,
        enable_memory=True,
        enable_replanning=False,
        enable_tools=False,
        enable_evaluation=True,
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
        evaluation_path.exists()
    )

    assert (
        memory_path
        != evaluation_path
    )


def test_evaluation_can_coexist_with_all_subsystems(
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
        evaluation_db_path=str(
            tmp_path
            / "evaluations.db"
        ),
        enable_self_healing=True,
        enable_memory=True,
        enable_replanning=True,
        enable_tools=True,
        enable_evaluation=True,
    )

    assert (
        engine.repair_loop
        is not None
    )

    assert (
        engine.memory_manager
        is not None
    )

    assert (
        engine.replanner
        is not None
    )

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
