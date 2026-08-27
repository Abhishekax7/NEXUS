from fastapi import FastAPI

from app.api.server import (
    app,
    build_production_app,
)


def test_production_app_is_fastapi():
    assert isinstance(
        app,
        FastAPI,
    )


def test_production_app_title():
    assert (
        app.title
        == "NEXUS Control Plane"
    )


def test_factory_builds_new_app():
    first = (
        build_production_app()
    )

    second = (
        build_production_app()
    )

    assert isinstance(
        first,
        FastAPI,
    )

    assert isinstance(
        second,
        FastAPI,
    )

    assert first is not second


def test_production_app_has_health_route():
    paths = {
        route.path
        for route in app.routes
    }

    assert "/health" in paths


def test_production_app_has_docs():
    paths = {
        route.path
        for route in app.routes
    }

    assert "/docs" in paths


def test_production_app_has_run_routes():
    paths = {
        route.path
        for route in app.routes
    }

    assert (
        "/runs/{run_id}"
        in paths
    )

    assert (
        "/runs/{run_id}/summary"
        in paths
    )

    assert (
        "/runs/{run_id}/recovery"
        in paths
    )

    assert (
        "/runs/{run_id}/resume"
        in paths
    )


def test_production_app_has_control_routes():
    paths = {
        route.path
        for route in app.routes
    }

    assert (
        "/runs/{run_id}/trace"
        in paths
    )

    assert (
        "/runs/{run_id}/evaluation"
        in paths
    )

    assert (
        "/approvals"
        in paths
    )
