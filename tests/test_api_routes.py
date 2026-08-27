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

from app.checkpointing.service import (
    CheckpointService,
)
from app.checkpointing.store import (
    CheckpointStore,
)

from app.core.engine import (
    NexusEngine,
)
from app.core.models import (
    AgentRole,
    AgentTask,
    TaskStatus,
)
from app.core.state import (
    NexusState,
)


def build_client(
    tmp_path,
):
    engine = NexusEngine(
        registry=AgentRegistry(),
        checkpoint_service=(
            CheckpointService(
                store=CheckpointStore(
                    db_path=str(
                        tmp_path
                        / "checkpoints.db"
                    )
                )
            )
        ),
    )

    plane = NexusControlPlane(
        engine
    )

    app = create_app(
        plane
    )

    client = TestClient(
        app
    )

    return (
        client,
        plane,
        engine,
    )


def build_state():
    state = NexusState(
        user_request=(
            "Build a FastAPI service."
        )
    )

    task = AgentTask(
        title="Implement API",
        description=(
            "Build the API service."
        ),
        assigned_agent=(
            AgentRole.CODER
        ),
    )

    state.add_task(
        task
    )

    return (
        state,
        task,
    )


def test_health_endpoint(
    tmp_path,
):
    client, _, _ = (
        build_client(
            tmp_path
        )
    )

    response = client.get(
        "/health"
    )

    assert (
        response.status_code
        == 200
    )

    payload = (
        response.json()
    )

    assert (
        payload["status"]
        == "ok"
    )

    assert (
        payload["service"]
        == "nexus"
    )


def test_get_registered_run(
    tmp_path,
):
    client, plane, _ = (
        build_client(
            tmp_path
        )
    )

    state, _ = build_state()

    plane.register_state(
        state
    )

    response = client.get(
        f"/runs/{state.run_id}"
    )

    assert (
        response.status_code
        == 200
    )

    payload = (
        response.json()
    )

    assert (
        payload["run_id"]
        == state.run_id
    )

    assert (
        payload["status"]
        == "created"
    )

    assert (
        payload["task_count"]
        == 1
    )


def test_missing_run_returns_404(
    tmp_path,
):
    client, _, _ = (
        build_client(
            tmp_path
        )
    )

    response = client.get(
        "/runs/missing-run"
    )

    assert (
        response.status_code
        == 404
    )


def test_run_summary_endpoint(
    tmp_path,
):
    client, plane, _ = (
        build_client(
            tmp_path
        )
    )

    state, task = build_state()

    task.status = (
        TaskStatus.COMPLETED
    )

    plane.register_state(
        state
    )

    response = client.get(
        f"/runs/{state.run_id}/summary"
    )

    assert (
        response.status_code
        == 200
    )

    payload = (
        response.json()
    )

    assert (
        payload[
            "completed_task_count"
        ]
        == 1
    )

    assert (
        payload[
            "failed_task_count"
        ]
        == 0
    )


def test_recovery_endpoint_for_active_checkpoint(
    tmp_path,
):
    client, _, engine = (
        build_client(
            tmp_path
        )
    )

    state, _ = build_state()

    engine.checkpoint_service.workflow_started(
        state
    )

    response = client.get(
        f"/runs/{state.run_id}/recovery"
    )

    assert (
        response.status_code
        == 200
    )

    payload = (
        response.json()
    )

    assert (
        payload["recoverable"]
        is True
    )

    assert (
        payload["status"]
        == "recoverable"
    )


def test_recovery_endpoint_for_missing_run(
    tmp_path,
):
    client, _, _ = (
        build_client(
            tmp_path
        )
    )

    response = client.get(
        "/runs/missing/recovery"
    )

    assert (
        response.status_code
        == 200
    )

    payload = (
        response.json()
    )

    assert (
        payload["status"]
        == "not_found"
    )

    assert (
        payload["recoverable"]
        is False
    )


def test_pending_approvals_without_manager_returns_400(
    tmp_path,
):
    client, _, _ = (
        build_client(
            tmp_path
        )
    )

    response = client.get(
        "/approvals"
    )

    assert (
        response.status_code
        == 400
    )


def test_trace_without_observability_returns_400(
    tmp_path,
):
    client, _, _ = (
        build_client(
            tmp_path
        )
    )

    response = client.get(
        "/runs/run-1/trace"
    )

    assert (
        response.status_code
        == 400
    )


def test_evaluation_without_service_returns_400(
    tmp_path,
):
    client, _, _ = (
        build_client(
            tmp_path
        )
    )

    response = client.get(
        "/runs/run-1/evaluation"
    )

    assert (
        response.status_code
        == 400
    )


def test_invalid_route_returns_404(
    tmp_path,
):
    client, _, _ = (
        build_client(
            tmp_path
        )
    )

    response = client.get(
        "/does-not-exist"
    )

    assert (
        response.status_code
        == 404
    )


def test_openapi_schema_is_available(
    tmp_path,
):
    client, _, _ = (
        build_client(
            tmp_path
        )
    )

    response = client.get(
        "/openapi.json"
    )

    assert (
        response.status_code
        == 200
    )

    payload = response.json()

    assert (
        payload["info"]["title"]
        == "NEXUS Control Plane"
    )


def test_docs_endpoint_is_available(
    tmp_path,
):
    client, _, _ = (
        build_client(
            tmp_path
        )
    )

    response = client.get(
        "/docs"
    )

    assert (
        response.status_code
        == 200
    )
