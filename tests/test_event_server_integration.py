from fastapi import (
    FastAPI,
)

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


def build_engine():
    return build_nexus_engine(
        enable_self_healing=False,
        enable_memory=False,
        enable_replanning=False,
        enable_evaluation=False,
        enable_observability=False,
        enable_checkpointing=False,
    )


def test_build_event_bus_returns_bus():
    bus = build_event_bus()

    assert isinstance(
        bus,
        EventBus,
    )


def test_job_manager_shares_event_bus():
    engine = build_engine()

    bus = build_event_bus()

    engine.event_bus = bus

    manager = build_job_manager(
        engine,
        bus,
    )

    assert (
        manager.event_bus
        is bus
    )


def test_job_manager_keeps_governance():
    engine = build_engine()

    bus = build_event_bus()

    engine.event_bus = bus

    manager = build_job_manager(
        engine,
        bus,
    )

    assert (
        manager.governance_service
        is engine.governance_service
    )


def test_production_app_builds():
    app = build_production_app()

    assert isinstance(
        app,
        FastAPI,
    )


def test_production_app_has_event_routes():
    app = build_production_app()

    paths = {
        route.path
        for route
        in app.routes
    }

    assert "/events" in paths

    assert (
        "/events/stream"
        in paths
    )


def test_production_app_keeps_job_routes():
    app = build_production_app()

    paths = {
        route.path
        for route
        in app.routes
    }

    assert "/jobs" in paths

    assert (
        "/jobs/{job_id}"
        in paths
    )


def test_production_app_keeps_run_routes():
    app = build_production_app()

    paths = {
        route.path
        for route
        in app.routes
    }

    assert "/runs" in paths

    assert (
        "/runs/{run_id}"
        in paths
    )

    assert "/health" in paths
