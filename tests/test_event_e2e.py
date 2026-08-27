from fastapi.testclient import (
    TestClient,
)

from app.agents.base import (
    BaseAgent,
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
from app.core.models import (
    AgentRole,
    AgentTask,
    Artifact,
    ArtifactType,
)

from app.events.bus import (
    EventBus,
)
from app.events.models import (
    EventType,
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


class SuccessfulAgent(
    BaseAgent
):
    role = AgentRole.CODER

    def execute(
        self,
        task,
        state,
    ):
        return Artifact(
            type=ArtifactType.CODE,
            name="event_e2e_code",
            content={
                "files": [],
            },
            created_by=self.role,
        )


def build_stack():
    registry = AgentRegistry()

    registry.register(
        AgentRole.CODER,
        SuccessfulAgent,
    )

    engine = NexusEngine(
        registry=registry
    )

    event_bus = EventBus()

    engine.event_bus = (
        event_bus
    )

    worker = WorkflowWorker(
        engine=engine
    )

    manager = JobManager(
        queue=PriorityJobQueue(),
        worker=worker,
        event_bus=event_bus,
    )

    plane = NexusControlPlane(
        engine,
        job_manager=manager,
    )

    client = TestClient(
        create_app(
            plane
        )
    )

    return (
        client,
        plane,
        event_bus,
    )


def create_run_with_task(
    client,
    plane,
):
    response = client.post(
        "/runs",
        json={
            "user_request":
                "Build live event workflow."
        },
    )

    assert (
        response.status_code
        == 201
    )

    run_id = response.json()[
        "run_id"
    ]

    state = plane.get_state(
        run_id
    )

    state.add_task(
        AgentTask(
            title="Generate code",
            description=(
                "Generate event-aware code."
            ),
            assigned_agent=(
                AgentRole.CODER
            ),
        )
    )

    return run_id


def submit_job(
    client,
    run_id,
):
    response = client.post(
        "/jobs",
        json={
            "run_id": run_id,
            "priority": "normal",
            "max_attempts": 1,
            "metadata": {
                "source":
                    "phase-19-e2e"
            },
        },
    )

    assert (
        response.status_code
        == 201
    )

    return response.json()[
        "job_id"
    ]


def test_full_job_lifecycle_appears_in_event_history():
    client, plane, _ = (
        build_stack()
    )

    run_id = create_run_with_task(
        client,
        plane,
    )

    job_id = submit_job(
        client,
        run_id,
    )

    execution = client.post(
        f"/jobs/{job_id}/execute"
    )

    assert (
        execution.status_code
        == 200
    )

    response = client.get(
        "/events",
        params={
            "run_id": run_id,
        },
    )

    assert (
        response.status_code
        == 200
    )

    payload = response.json()

    types = [
        event["type"]
        for event
        in payload["events"]
    ]

    assert (
        "job.queued"
        in types
    )

    assert (
        "job.started"
        in types
    )

    assert (
        "job.completed"
        in types
    )


def test_event_history_can_filter_specific_job():
    client, plane, _ = (
        build_stack()
    )

    run_id = create_run_with_task(
        client,
        plane,
    )

    job_id = submit_job(
        client,
        run_id,
    )

    client.post(
        f"/jobs/{job_id}/execute"
    )

    response = client.get(
        "/events",
        params={
            "job_id": job_id,
        },
    )

    assert (
        response.status_code
        == 200
    )

    payload = response.json()

    assert (
        payload["count"]
        >= 3
    )

    assert all(
        event["job_id"]
        == job_id
        for event
        in payload["events"]
    )


def test_event_history_order_matches_lifecycle():
    client, plane, _ = (
        build_stack()
    )

    run_id = create_run_with_task(
        client,
        plane,
    )

    job_id = submit_job(
        client,
        run_id,
    )

    client.post(
        f"/jobs/{job_id}/execute"
    )

    response = client.get(
        "/events",
        params={
            "job_id": job_id,
        },
    )

    types = [
        event["type"]
        for event
        in response.json()[
            "events"
        ]
    ]

    assert types[:3] == [
        EventType.JOB_QUEUED.value,
        EventType.JOB_STARTED.value,
        EventType.JOB_COMPLETED.value,
    ]


def test_cancelled_job_emits_cancel_event():
    client, plane, _ = (
        build_stack()
    )

    run_id = create_run_with_task(
        client,
        plane,
    )

    job_id = submit_job(
        client,
        run_id,
    )

    response = client.post(
        f"/jobs/{job_id}/cancel"
    )

    assert (
        response.status_code
        == 200
    )

    events = client.get(
        "/events",
        params={
            "job_id": job_id,
        },
    ).json()[
        "events"
    ]

    assert (
        events[-1]["type"]
        == "job.cancelled"
    )


def test_event_history_limit_returns_latest_events():
    client, plane, _ = (
        build_stack()
    )

    run_id = create_run_with_task(
        client,
        plane,
    )

    job_id = submit_job(
        client,
        run_id,
    )

    client.post(
        f"/jobs/{job_id}/execute"
    )

    response = client.get(
        "/events",
        params={
            "job_id": job_id,
            "limit": 2,
        },
    )

    payload = response.json()

    assert payload["count"] == 2

    assert (
        payload["total"]
        >= 3
    )


def test_event_api_uses_same_live_bus():
    client, plane, bus = (
        build_stack()
    )

    run_id = create_run_with_task(
        client,
        plane,
    )

    job_id = submit_job(
        client,
        run_id,
    )

    direct_count = (
        bus.event_count()
    )

    api_count = client.get(
        "/events"
    ).json()[
        "count"
    ]

    assert (
        direct_count
        == api_count
    )

    assert (
        direct_count
        >= 1
    )


def test_openapi_exposes_live_event_endpoints():
    client, _, _ = (
        build_stack()
    )

    paths = client.get(
        "/openapi.json"
    ).json()[
        "paths"
    ]

    assert "/events" in paths

    assert (
        "/events/stream"
        in paths
    )
