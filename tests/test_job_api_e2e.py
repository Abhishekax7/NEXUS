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
from app.core.state import (
    NexusState,
)

from app.jobs.manager import (
    JobManager,
)
from app.jobs.models import (
    JobPriority,
    JobStatus,
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
            name="async_e2e_code",
            content={
                "files": [],
            },
            created_by=self.role,
        )


class FailingAgent(
    BaseAgent
):
    role = AgentRole.CODER

    def execute(
        self,
        task,
        state,
    ):
        raise RuntimeError(
            "Async E2E failure."
        )


def build_stack(
    failing=False,
):
    registry = AgentRegistry()

    if failing:
        registry.register(
            AgentRole.CODER,
            FailingAgent,
        )
    else:
        registry.register(
            AgentRole.CODER,
            SuccessfulAgent,
        )

    engine = NexusEngine(
        registry=registry
    )

    queue = PriorityJobQueue()

    worker = WorkflowWorker(
        engine=engine
    )

    manager = JobManager(
        queue=queue,
        worker=worker,
    )

    plane = NexusControlPlane(
        engine,
        job_manager=manager,
    )

    app = create_app(
        plane
    )

    return (
        TestClient(app),
        plane,
        manager,
    )


def add_coder_task(
    plane,
    run_id,
):
    state = plane.get_state(
        run_id
    )

    task = AgentTask(
        title="Implement feature",
        description=(
            "Generate application code."
        ),
        assigned_agent=(
            AgentRole.CODER
        ),
    )

    state.add_task(
        task
    )

    return state


def create_run(
    client,
):
    response = client.post(
        "/runs",
        json={
            "user_request":
                "Build async NEXUS workflow."
        },
    )

    assert (
        response.status_code
        == 201
    )

    return response.json()[
        "run_id"
    ]


def submit_job(
    client,
    run_id,
    *,
    priority="normal",
    max_attempts=1,
):
    response = client.post(
        "/jobs",
        json={
            "run_id": run_id,
            "priority": priority,
            "max_attempts":
                max_attempts,
            "metadata": {
                "source":
                    "phase-17-e2e"
            },
        },
    )

    assert (
        response.status_code
        == 201
    )

    return response.json()


def test_run_can_be_submitted_as_job():
    client, plane, _ = (
        build_stack()
    )

    run_id = create_run(
        client
    )

    add_coder_task(
        plane,
        run_id,
    )

    payload = submit_job(
        client,
        run_id,
    )

    assert (
        payload["run_id"]
        == run_id
    )

    assert (
        payload["status"]
        == JobStatus.QUEUED.value
    )

    assert (
        payload["queued"]
        is True
    )

    assert (
        payload["terminal"]
        is False
    )


def test_submitted_job_can_be_inspected():
    client, plane, _ = (
        build_stack()
    )

    run_id = create_run(
        client
    )

    add_coder_task(
        plane,
        run_id,
    )

    submitted = submit_job(
        client,
        run_id,
    )

    job_id = submitted[
        "job_id"
    ]

    response = client.get(
        f"/jobs/{job_id}"
    )

    assert (
        response.status_code
        == 200
    )

    payload = response.json()

    assert (
        payload["job_id"]
        == job_id
    )

    assert (
        payload["status"]
        == "queued"
    )


def test_pending_jobs_endpoint_lists_job():
    client, plane, _ = (
        build_stack()
    )

    run_id = create_run(
        client
    )

    add_coder_task(
        plane,
        run_id,
    )

    submitted = submit_job(
        client,
        run_id,
    )

    response = client.get(
        "/jobs"
    )

    assert (
        response.status_code
        == 200
    )

    payload = response.json()

    assert payload["count"] == 1

    assert (
        payload["jobs"][0][
            "job_id"
        ]
        == submitted["job_id"]
    )


def test_job_executes_through_http():
    client, plane, _ = (
        build_stack()
    )

    run_id = create_run(
        client
    )

    add_coder_task(
        plane,
        run_id,
    )

    submitted = submit_job(
        client,
        run_id,
    )

    job_id = submitted[
        "job_id"
    ]

    response = client.post(
        f"/jobs/{job_id}/execute"
    )

    assert (
        response.status_code
        == 200
    )

    payload = response.json()

    assert (
        payload["job_id"]
        == job_id
    )

    assert (
        payload["success"]
        is True
    )

    assert (
        payload["status"]
        == "completed"
    )


def test_completed_job_status_is_persisted():
    client, plane, _ = (
        build_stack()
    )

    run_id = create_run(
        client
    )

    add_coder_task(
        plane,
        run_id,
    )

    submitted = submit_job(
        client,
        run_id,
    )

    job_id = submitted[
        "job_id"
    ]

    client.post(
        f"/jobs/{job_id}/execute"
    )

    response = client.get(
        f"/jobs/{job_id}"
    )

    assert (
        response.status_code
        == 200
    )

    payload = response.json()

    assert (
        payload["status"]
        == "completed"
    )

    assert (
        payload["terminal"]
        is True
    )

    assert (
        payload["queued"]
        is False
    )


def test_execute_next_processes_highest_priority():
    client, plane, _ = (
        build_stack()
    )

    low_run = create_run(
        client
    )

    high_run = create_run(
        client
    )

    add_coder_task(
        plane,
        low_run,
    )

    add_coder_task(
        plane,
        high_run,
    )

    low = submit_job(
        client,
        low_run,
        priority="low",
    )

    high = submit_job(
        client,
        high_run,
        priority="high",
    )

    response = client.post(
        "/jobs/execute-next"
    )

    assert (
        response.status_code
        == 200
    )

    payload = response.json()

    assert (
        payload["job_id"]
        == high["job_id"]
    )

    low_status = client.get(
        f"/jobs/{low['job_id']}"
    ).json()

    assert (
        low_status["status"]
        == "queued"
    )


def test_queued_job_can_be_cancelled():
    client, plane, _ = (
        build_stack()
    )

    run_id = create_run(
        client
    )

    add_coder_task(
        plane,
        run_id,
    )

    submitted = submit_job(
        client,
        run_id,
    )

    job_id = submitted[
        "job_id"
    ]

    response = client.post(
        f"/jobs/{job_id}/cancel"
    )

    assert (
        response.status_code
        == 200
    )

    payload = response.json()

    assert (
        payload["status"]
        == "cancelled"
    )

    assert (
        payload["terminal"]
        is True
    )


def test_failed_job_can_be_retried():
    client, plane, _ = (
        build_stack(
            failing=True
        )
    )

    run_id = create_run(
        client
    )

    add_coder_task(
        plane,
        run_id,
    )

    submitted = submit_job(
        client,
        run_id,
        max_attempts=2,
    )

    job_id = submitted[
        "job_id"
    ]

    execution = client.post(
        f"/jobs/{job_id}/execute"
    )

    assert (
        execution.status_code
        == 200
    )

    assert (
        execution.json()[
            "status"
        ]
        == "failed"
    )

    retry = client.post(
        f"/jobs/{job_id}/retry"
    )

    assert (
        retry.status_code
        == 200
    )

    payload = retry.json()

    assert (
        payload["status"]
        == "queued"
    )

    assert (
        payload["attempt"]
        == 1
    )


def test_missing_job_returns_404():
    client, _, _ = (
        build_stack()
    )

    response = client.get(
        "/jobs/missing-job"
    )

    assert (
        response.status_code
        == 404
    )


def test_execute_next_without_jobs_returns_404():
    client, _, _ = (
        build_stack()
    )

    response = client.post(
        "/jobs/execute-next"
    )

    assert (
        response.status_code
        == 404
    )


def test_openapi_contains_job_routes():
    client, _, _ = (
        build_stack()
    )

    response = client.get(
        "/openapi.json"
    )

    assert (
        response.status_code
        == 200
    )

    paths = response.json()[
        "paths"
    ]

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
