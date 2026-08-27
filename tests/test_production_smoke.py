from fastapi import FastAPI

from app.api.server import (
    build_event_bus,
    build_job_manager,
    build_production_app,
)

from app.core.runtime import (
    build_nexus_engine,
)

from app.events.bus import (
    EventBus,
)

from app.jobs.manager import (
    JobManager,
)


def test_production_engine_boots():
    engine = build_nexus_engine()

    assert engine is not None
    assert engine.registry is not None


def test_production_engine_has_memory():
    engine = build_nexus_engine()

    assert (
        engine.memory_manager
        is not None
    )


def test_production_engine_has_self_healing():
    engine = build_nexus_engine()

    assert (
        engine.repair_loop
        is not None
    )


def test_production_engine_has_replanning():
    engine = build_nexus_engine()

    assert (
        engine.replanner
        is not None
    )

    assert (
        engine.plan_mutator
        is not None
    )


def test_production_engine_has_tool_runtime():
    engine = build_nexus_engine()

    assert (
        engine.tool_registry
        is not None
    )

    assert (
        engine.tool_runtime
        is not None
    )


def test_production_engine_has_approval_system():
    engine = build_nexus_engine()

    assert (
        engine.approval_manager
        is not None
    )

    assert (
        engine.approval_gate
        is not None
    )


def test_production_engine_has_governance():
    engine = build_nexus_engine()

    assert (
        engine.governance_service
        is not None
    )


def test_production_engine_has_evaluation():
    engine = build_nexus_engine()

    assert (
        engine.evaluation_service
        is not None
    )


def test_production_engine_has_observability():
    engine = build_nexus_engine()

    assert (
        engine.observability_service
        is not None
    )


def test_production_engine_has_checkpointing():
    engine = build_nexus_engine()

    assert (
        engine.checkpoint_service
        is not None
    )


def test_production_event_bus_boots():
    event_bus = build_event_bus()

    assert isinstance(
        event_bus,
        EventBus,
    )


def test_production_job_manager_boots():
    engine = build_nexus_engine()

    event_bus = build_event_bus()

    manager = build_job_manager(
        engine,
        event_bus,
    )

    assert isinstance(
        manager,
        JobManager,
    )

    assert (
        manager.worker.engine
        is engine
    )

    assert (
        manager.event_bus
        is event_bus
    )


def test_job_manager_shares_governance():
    engine = build_nexus_engine()

    event_bus = build_event_bus()

    manager = build_job_manager(
        engine,
        event_bus,
    )

    assert (
        manager.governance_service
        is engine.governance_service
    )


def test_job_manager_backward_compatible():
    engine = build_nexus_engine()

    manager = build_job_manager(
        engine
    )

    assert isinstance(
        manager,
        JobManager,
    )

    assert (
        manager.event_bus
        is not None
    )


def test_complete_production_app_boots():
    app = build_production_app()

    assert isinstance(
        app,
        FastAPI,
    )


def test_production_app_exposes_core_routes():
    app = build_production_app()

    paths = {
        route.path
        for route in app.routes
    }

    required_paths = {
        "/health",
        "/runs",
        "/jobs",
        "/events",
        "/events/stream",
        "/approvals",
        "/docs",
        "/openapi.json",
    }

    assert required_paths.issubset(
        paths
    )


def test_production_app_has_unique_core_services():
    app = build_production_app()

    assert app is not None

    paths = [
        route.path
        for route in app.routes
    ]

    assert paths.count(
        "/health"
    ) == 1

    assert paths.count(
        "/events"
    ) == 1


def test_production_stack_is_independently_buildable():
    first = build_production_app()
    second = build_production_app()

    assert first is not second

    assert isinstance(
        first,
        FastAPI,
    )

    assert isinstance(
        second,
        FastAPI,
    )
