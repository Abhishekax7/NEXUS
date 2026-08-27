from fastapi.testclient import (
    TestClient,
)

from app.agents.registry import (
    AgentRegistry,
)

from app.api.app import (
    create_app,
)
from app.api.control_plane import (
    NexusControlPlane,
)

from app.core.engine import (
    NexusEngine,
)

from app.events.bus import (
    EventBus,
)
from app.events.models import (
    EventType,
    NexusEvent,
)


def build_client():
    engine = NexusEngine(
        registry=AgentRegistry()
    )

    event_bus = EventBus()

    engine.event_bus = (
        event_bus
    )

    plane = NexusControlPlane(
        engine
    )

    app = create_app(
        plane
    )

    return (
        TestClient(app),
        event_bus,
    )


def test_events_endpoint_returns_history():
    client, bus = build_client()

    bus.publish(
        NexusEvent(
            type=EventType.RUN_CREATED,
            run_id="run-1",
            source="test",
            message="Run created.",
        )
    )

    response = client.get(
        "/events"
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["count"] == 1

    assert (
        payload["events"][0]["type"]
        == "run.created"
    )


def test_events_endpoint_filters_run():
    client, bus = build_client()

    bus.publish(
        NexusEvent(
            type=EventType.RUN_CREATED,
            run_id="run-1",
            source="test",
            message="Run one.",
        )
    )

    bus.publish(
        NexusEvent(
            type=EventType.RUN_CREATED,
            run_id="run-2",
            source="test",
            message="Run two.",
        )
    )

    response = client.get(
        "/events",
        params={
            "run_id": "run-2",
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["count"] == 1

    assert (
        payload["events"][0]["run_id"]
        == "run-2"
    )


def test_events_endpoint_filters_job():
    client, bus = build_client()

    bus.publish(
        NexusEvent(
            type=EventType.JOB_STARTED,
            job_id="job-1",
            source="test",
            message="Job one.",
        )
    )

    bus.publish(
        NexusEvent(
            type=EventType.JOB_STARTED,
            job_id="job-2",
            source="test",
            message="Job two.",
        )
    )

    response = client.get(
        "/events",
        params={
            "job_id": "job-2",
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["count"] == 1

    assert (
        payload["events"][0]["job_id"]
        == "job-2"
    )


def test_events_endpoint_respects_limit():
    client, bus = build_client()

    for index in range(5):
        bus.publish(
            NexusEvent(
                type=EventType.TASK_COMPLETED,
                source="test",
                message=(
                    f"Event {index}"
                ),
            )
        )

    response = client.get(
        "/events",
        params={
            "limit": 2,
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["count"] == 2
    assert payload["total"] == 5


def test_invalid_event_limit_returns_400():
    client, _ = build_client()

    response = client.get(
        "/events",
        params={
            "limit": 0,
        },
    )

    assert response.status_code == 400


def test_events_without_bus_returns_400():
    engine = NexusEngine(
        registry=AgentRegistry()
    )

    plane = NexusControlPlane(
        engine
    )

    client = TestClient(
        create_app(
            plane
        )
    )

    response = client.get(
        "/events"
    )

    assert response.status_code == 400


def test_openapi_contains_event_routes():
    client, _ = build_client()

    response = client.get(
        "/openapi.json"
    )

    assert response.status_code == 200

    paths = response.json()["paths"]

    assert "/events" in paths
    assert "/events/stream" in paths
