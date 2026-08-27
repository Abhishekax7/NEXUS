import pytest

from app.agents.base import (
    BaseAgent,
)
from app.agents.registry import (
    AgentRegistry,
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

from app.jobs.models import (
    JobStateError,
    JobStatus,
    WorkflowJob,
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
            name="generated_code",
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
            "Simulated worker failure."
        )


def build_state():
    state = NexusState(
        user_request=(
            "Build an async workflow."
        )
    )

    task = AgentTask(
        title="Implement workflow",
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


def build_success_worker():
    registry = AgentRegistry()

    registry.register(
        AgentRole.CODER,
        SuccessfulAgent,
    )

    engine = NexusEngine(
        registry=registry
    )

    return WorkflowWorker(
        engine=engine
    )


def build_failure_worker():
    registry = AgentRegistry()

    registry.register(
        AgentRole.CODER,
        FailingAgent,
    )

    engine = NexusEngine(
        registry=registry
    )

    return WorkflowWorker(
        engine=engine
    )


def test_successful_job_completes():
    state = build_state()

    job = WorkflowJob(
        run_id=state.run_id
    )

    worker = (
        build_success_worker()
    )

    result = worker.execute(
        job,
        state,
    )

    assert result.success is True

    assert (
        result.status
        == JobStatus.COMPLETED
    )

    assert (
        job.status
        == JobStatus.COMPLETED
    )


def test_successful_job_tracks_timestamps():
    state = build_state()

    job = WorkflowJob(
        run_id=state.run_id
    )

    worker = (
        build_success_worker()
    )

    worker.execute(
        job,
        state,
    )

    assert (
        job.started_at
        is not None
    )

    assert (
        job.completed_at
        is not None
    )

    assert (
        job.completed_at
        >= job.started_at
    )


def test_successful_job_increments_attempt():
    state = build_state()

    job = WorkflowJob(
        run_id=state.run_id
    )

    worker = (
        build_success_worker()
    )

    worker.execute(
        job,
        state,
    )

    assert job.attempt == 1


def test_successful_result_contains_metadata():
    state = build_state()

    job = WorkflowJob(
        run_id=state.run_id
    )

    result = (
        build_success_worker()
        .execute(
            job,
            state,
        )
    )

    assert (
        result.metadata[
            "task_count"
        ]
        == 1
    )

    assert (
        result.metadata[
            "artifact_count"
        ]
        == 1
    )


def test_failed_agent_marks_job_failed():
    state = build_state()

    job = WorkflowJob(
        run_id=state.run_id
    )

    worker = (
        build_failure_worker()
    )

    result = worker.execute(
        job,
        state,
    )

    assert result.success is False

    assert (
        result.status
        == JobStatus.FAILED
    )

    assert (
        job.status
        == JobStatus.FAILED
    )


def test_failure_result_contains_error():
    state = build_state()

    job = WorkflowJob(
        run_id=state.run_id
    )

    result = (
        build_failure_worker()
        .execute(
            job,
            state,
        )
    )

    assert (
        "Simulated worker failure"
        in result.error
    )


def test_non_queued_job_cannot_execute():
    state = build_state()

    job = WorkflowJob(
        run_id=state.run_id,
        status=JobStatus.RUNNING,
    )

    worker = (
        build_success_worker()
    )

    with pytest.raises(
        JobStateError
    ):
        worker.execute(
            job,
            state,
        )


def test_mismatched_run_id_is_rejected():
    state = build_state()

    job = WorkflowJob(
        run_id="different-run"
    )

    worker = (
        build_success_worker()
    )

    with pytest.raises(
        JobStateError
    ):
        worker.execute(
            job,
            state,
        )


def test_failed_job_can_retry_when_budget_remains():
    state = build_state()

    job = WorkflowJob(
        run_id=state.run_id,
        max_attempts=2,
    )

    worker = (
        build_failure_worker()
    )

    worker.execute(
        job,
        state,
    )

    assert (
        worker.can_retry(
            job
        )
        is True
    )


def test_failed_job_cannot_retry_after_budget_exhausted():
    state = build_state()

    job = WorkflowJob(
        run_id=state.run_id,
        max_attempts=1,
    )

    worker = (
        build_failure_worker()
    )

    worker.execute(
        job,
        state,
    )

    assert (
        worker.can_retry(
            job
        )
        is False
    )


def test_prepare_retry_resets_execution_state():
    state = build_state()

    job = WorkflowJob(
        run_id=state.run_id,
        max_attempts=2,
    )

    worker = (
        build_failure_worker()
    )

    worker.execute(
        job,
        state,
    )

    worker.prepare_retry(
        job
    )

    assert (
        job.status
        == JobStatus.QUEUED
    )

    assert (
        job.started_at
        is None
    )

    assert (
        job.completed_at
        is None
    )

    assert job.attempt == 1


def test_prepare_retry_fails_without_budget():
    state = build_state()

    job = WorkflowJob(
        run_id=state.run_id,
        max_attempts=1,
    )

    worker = (
        build_failure_worker()
    )

    worker.execute(
        job,
        state,
    )

    with pytest.raises(
        JobStateError
    ):
        worker.prepare_retry(
            job
        )


def test_queued_job_can_be_cancelled():
    state = build_state()

    job = WorkflowJob(
        run_id=state.run_id
    )

    worker = (
        build_success_worker()
    )

    worker.cancel(
        job
    )

    assert (
        job.status
        == JobStatus.CANCELLED
    )

    assert (
        job.completed_at
        is not None
    )


def test_running_job_cannot_be_cancelled():
    state = build_state()

    job = WorkflowJob(
        run_id=state.run_id,
        status=JobStatus.RUNNING,
    )

    worker = (
        build_success_worker()
    )

    with pytest.raises(
        JobStateError
    ):
        worker.cancel(
            job
        )
