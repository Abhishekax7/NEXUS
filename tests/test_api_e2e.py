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


def build_stack(
    tmp_path,
):
    """
    Build an isolated API stack using
    temporary checkpoint persistence.
    """

    checkpoint_service = (
        CheckpointService(
            store=CheckpointStore(
                db_path=str(
                    tmp_path
                    / "checkpoints.db"
                )
            )
        )
    )

    engine = NexusEngine(
        registry=AgentRegistry(),
        checkpoint_service=(
            checkpoint_service
        ),
    )

    control_plane = (
        NexusControlPlane(
            engine
        )
    )

    app = create_app(
        control_plane
    )

    client = TestClient(
        app
    )

    return (
        client,
        control_plane,
        engine,
    )


def test_create_run_through_api(
    tmp_path,
):
    client, _, _ = (
        build_stack(
            tmp_path
        )
    )

    response = client.post(
        "/runs",
        json={
            "user_request":
                "Build a secure API.",
            "metadata": {
                "source":
                    "e2e-test",
            },
        },
    )

    assert (
        response.status_code
        == 201
    )

    payload = (
        response.json()
    )

    assert (
        payload["status"]
        == "created"
    )

    assert (
        payload["user_request"]
        == "Build a secure API."
    )

    assert (
        payload["metadata"][
            "source"
        ]
        == "e2e-test"
    )

    assert payload["run_id"]


def test_created_run_can_be_read_back(
    tmp_path,
):
    client, _, _ = (
        build_stack(
            tmp_path
        )
    )

    created = client.post(
        "/runs",
        json={
            "user_request":
                "Build an agent."
        },
    )

    assert (
        created.status_code
        == 201
    )

    run_id = (
        created.json()[
            "run_id"
        ]
    )

    response = client.get(
        f"/runs/{run_id}"
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
        == run_id
    )

    assert (
        payload["user_request"]
        == "Build an agent."
    )


def test_created_run_can_be_summarized(
    tmp_path,
):
    client, _, _ = (
        build_stack(
            tmp_path
        )
    )

    created = client.post(
        "/runs",
        json={
            "user_request":
                "Build a workflow."
        },
    )

    run_id = (
        created.json()[
            "run_id"
        ]
    )

    response = client.get(
        f"/runs/{run_id}/summary"
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
        == run_id
    )

    assert (
        payload["status"]
        == "created"
    )

    assert (
        payload["task_count"]
        == 0
    )

    assert (
        payload[
            "completed_task_count"
        ]
        == 0
    )


def test_created_run_has_checkpoint(
    tmp_path,
):
    client, _, _ = (
        build_stack(
            tmp_path
        )
    )

    created = client.post(
        "/runs",
        json={
            "user_request":
                "Build recoverable workflow."
        },
    )

    run_id = (
        created.json()[
            "run_id"
        ]
    )

    response = client.get(
        f"/runs/{run_id}/recovery"
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
        == run_id
    )

    assert (
        payload["recoverable"]
        is True
    )

    assert (
        payload["status"]
        == "recoverable"
    )

    assert (
        payload[
            "latest_checkpoint_id"
        ]
        is not None
    )


def test_checkpointed_run_survives_memory_loss(
    tmp_path,
):
    client, plane, _ = (
        build_stack(
            tmp_path
        )
    )

    created = client.post(
        "/runs",
        json={
            "user_request":
                "Persist this run."
        },
    )

    run_id = (
        created.json()[
            "run_id"
        ]
    )

    plane._runs.clear()

    response = client.get(
        f"/runs/{run_id}"
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
        == run_id
    )

    assert (
        payload["user_request"]
        == "Persist this run."
    )


def test_invalid_create_run_is_rejected(
    tmp_path,
):
    client, _, _ = (
        build_stack(
            tmp_path
        )
    )

    response = client.post(
        "/runs",
        json={
            "user_request": ""
        },
    )

    assert (
        response.status_code
        == 422
    )


def test_missing_run_returns_404_end_to_end(
    tmp_path,
):
    client, _, _ = (
        build_stack(
            tmp_path
        )
    )

    response = client.get(
        "/runs/does-not-exist"
    )

    assert (
        response.status_code
        == 404
    )


def test_openapi_contains_complete_run_api(
    tmp_path,
):
    client, _, _ = (
        build_stack(
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

    paths = (
        response.json()[
            "paths"
        ]
    )

    assert "/runs" in paths

    assert (
        "post"
        in paths["/runs"]
    )

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

    assert (
        "/runs/{run_id}/trace"
        in paths
    )

    assert (
        "/runs/{run_id}/evaluation"
        in paths
    )


def test_control_plane_health_and_docs(
    tmp_path,
):
    client, _, _ = (
        build_stack(
            tmp_path
        )
    )

    health = client.get(
        "/health"
    )

    docs = client.get(
        "/docs"
    )

    assert (
        health.status_code
        == 200
    )

    assert (
        docs.status_code
        == 200
    )

    payload = (
        health.json()
    )

    assert (
        payload["status"]
        == "ok"
    )

    assert (
        payload["service"]
        == "nexus"
    )
