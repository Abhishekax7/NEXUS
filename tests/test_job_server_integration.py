from fastapi import (
    FastAPI,
)

from app.api.server import (
    build_job_manager,
    build_production_app,
)

from app.core.runtime import (
    build_nexus_engine,
)

from app.jobs.manager import (
    JobManager,
)
from app.jobs.queue import (
    PriorityJobQueue,
)
from app.jobs.worker import (
    WorkflowWorker,
)


def build_engine():
    return build_nexus_engine(
        enable_self_healing=False,
        enable_memory=False,
        enable_replanning=False,
        enable_tools=False,
        enable_evaluation=False,
        enable_observability=False,
        enable_approvals=False,
        enable_checkpointing=False,
    )


def test_build_job_manager_returns_manager():
    manager = build_job_manager(
        build_engine()
    )

    assert isinstance(
        manager,
        JobManager,
    )


def test_job_manager_uses_priority_queue():
    manager = build_job_manager(
        build_engine()
    )

    assert isinstance(
        manager.queue,
        PriorityJobQueue,
    )


def test_job_manager_uses_workflow_worker():
    manager = build_job_manager(
        build_engine()
    )

    assert isinstance(
        manager.worker,
        WorkflowWorker,
    )


def test_worker_uses_same_engine():
    engine = build_engine()

    manager = build_job_manager(
        engine
    )

    assert (
        manager.worker.engine
        is engine
    )


def test_production_app_builds():
    app = build_production_app()

    assert isinstance(
        app,
        FastAPI,
    )


def test_production_app_has_job_routes():
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

    assert (
        "/jobs/execute-next"
        in paths
    )

    assert (
        "/jobs/{job_id}/execute"
        in paths
    )

    assert (
        "/jobs/{job_id}/cancel"
        in paths
    )

    assert (
        "/jobs/{job_id}/retry"
        in paths
    )


def test_production_app_keeps_phase_16_routes():
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

    assert "/docs" in paths
