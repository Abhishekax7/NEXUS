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

from app.jobs.manager import (
    JobManager,
)
from app.jobs.models import (
    JobNotFoundError,
    JobPriority,
    JobStateError,
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
            "Manager test failure."
        )


def build_state():
    state = NexusState(
        user_request=(
            "Build managed workflow."
        )
    )

    task = AgentTask(
        title="Implement workflow",
        description="Generate code.",
        assigned_agent=(
            AgentRole.CODER
        ),
    )

    state.add_task(
        task
    )

    return state


def build_manager(
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

    worker = WorkflowWorker(
        engine=engine
    )

    return JobManager(
        queue=PriorityJobQueue(),
        worker=worker,
    )


def test_state_can_be_submitted():
    manager = build_manager()

    state = build_state()

    job = manager.submit(
        state
    )

    assert (
        job.status
        == JobStatus.QUEUED
    )

    assert (
        job.run_id
        == state.run_id
    )

    assert (
        manager.job_count()
        == 1
    )


def test_submitted_job_enters_queue():
    manager = build_manager()

    state = build_state()

    job = manager.submit(
        state
    )

    assert (
        manager.queue.contains(
            job.id
        )
        is True
    )


def test_job_can_be_retrieved():
    manager = build_manager()

    state = build_state()

    job = manager.submit(
        state
    )

    assert (
        manager.get_job(
            job.id
        )
        is job
    )


def test_state_can_be_retrieved():
    manager = build_manager()

    state = build_state()

    job = manager.submit(
        state
    )

    assert (
        manager.get_state(
            job.id
        )
        is state
    )


def test_missing_job_raises():
    manager = build_manager()

    with pytest.raises(
        JobNotFoundError
    ):
        manager.get_job(
            "missing"
        )


def test_snapshot_reports_queued():
    manager = build_manager()

    state = build_state()

    job = manager.submit(
        state
    )

    snapshot = manager.snapshot(
        job.id
    )

    assert (
        snapshot.queued
        is True
    )

    assert (
        snapshot.running
        is False
    )

    assert (
        snapshot.terminal
        is False
    )


def test_execute_next_completes_job():
    manager = build_manager()

    state = build_state()

    job = manager.submit(
        state
    )

    result = (
        manager.execute_next()
    )

    assert result is not None

    assert result.success is True

    assert (
        job.status
        == JobStatus.COMPLETED
    )


def test_execute_next_empty_returns_none():
    manager = build_manager()

    assert (
        manager.execute_next()
        is None
    )


def test_result_is_recorded():
    manager = build_manager()

    state = build_state()

    job = manager.submit(
        state
    )

    manager.execute_next()

    result = manager.get_result(
        job.id
    )

    assert result is not None

    assert (
        result.success
        is True
    )


def test_execute_specific_job():
    manager = build_manager()

    first_state = build_state()
    second_state = build_state()

    first = manager.submit(
        first_state
    )

    second = manager.submit(
        second_state
    )

    result = manager.execute_job(
        second.id
    )

    assert (
        result.job_id
        == second.id
    )

    assert (
        second.status
        == JobStatus.COMPLETED
    )

    assert (
        first.status
        == JobStatus.QUEUED
    )


def test_completed_job_cannot_execute_again():
    manager = build_manager()

    state = build_state()

    job = manager.submit(
        state
    )

    manager.execute_job(
        job.id
    )

    with pytest.raises(
        JobStateError
    ):
        manager.execute_job(
            job.id
        )


def test_priority_is_preserved():
    manager = build_manager()

    state = build_state()

    job = manager.submit(
        state,
        priority=(
            JobPriority.HIGH
        ),
    )

    assert (
        job.priority
        == JobPriority.HIGH
    )


def test_execute_next_respects_priority():
    manager = build_manager()

    low_state = build_state()
    high_state = build_state()

    low = manager.submit(
        low_state,
        priority=(
            JobPriority.LOW
        ),
    )

    high = manager.submit(
        high_state,
        priority=(
            JobPriority.HIGH
        ),
    )

    result = (
        manager.execute_next()
    )

    assert (
        result.job_id
        == high.id
    )

    assert (
        low.status
        == JobStatus.QUEUED
    )


def test_queued_job_can_be_cancelled():
    manager = build_manager()

    state = build_state()

    job = manager.submit(
        state
    )

    cancelled = manager.cancel(
        job.id
    )

    assert (
        cancelled.status
        == JobStatus.CANCELLED
    )

    assert (
        manager.queue.contains(
            job.id
        )
        is False
    )


def test_failed_job_can_be_retried():
    manager = build_manager(
        failing=True
    )

    state = build_state()

    job = manager.submit(
        state,
        max_attempts=2,
    )

    result = (
        manager.execute_next()
    )

    assert (
        result.success
        is False
    )

    assert (
        job.status
        == JobStatus.FAILED
    )

    retried = manager.retry(
        job.id
    )

    assert (
        retried.status
        == JobStatus.QUEUED
    )

    assert (
        manager.queue.contains(
            job.id
        )
        is True
    )


def test_retry_clears_previous_result():
    manager = build_manager(
        failing=True
    )

    state = build_state()

    job = manager.submit(
        state,
        max_attempts=2,
    )

    manager.execute_next()

    assert (
        manager.get_result(
            job.id
        )
        is not None
    )

    manager.retry(
        job.id
    )

    assert (
        manager.get_result(
            job.id
        )
        is None
    )


def test_failed_job_without_budget_cannot_retry():
    manager = build_manager(
        failing=True
    )

    state = build_state()

    job = manager.submit(
        state,
        max_attempts=1,
    )

    manager.execute_next()

    with pytest.raises(
        JobStateError
    ):
        manager.retry(
            job.id
        )


def test_pending_jobs_are_priority_ordered():
    manager = build_manager()

    low = manager.submit(
        build_state(),
        priority=JobPriority.LOW,
    )

    normal = manager.submit(
        build_state(),
        priority=JobPriority.NORMAL,
    )

    high = manager.submit(
        build_state(),
        priority=JobPriority.HIGH,
    )

    pending = (
        manager.pending_jobs()
    )

    assert [
        job.id
        for job in pending
    ] == [
        high.id,
        normal.id,
        low.id,
    ]


def test_result_count_tracks_execution():
    manager = build_manager()

    first = manager.submit(
        build_state()
    )

    second = manager.submit(
        build_state()
    )

    manager.execute_job(
        first.id
    )

    manager.execute_job(
        second.id
    )

    assert (
        manager.result_count()
        == 2
    )
