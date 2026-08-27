from app.core.models import (
    AgentRole,
)
from app.core.state import (
    NexusState,
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
from app.jobs.models import (
    JobPriority,
    JobStatus,
)
from app.jobs.queue import (
    PriorityJobQueue,
)


class FakeResult:
    def __init__(
        self,
        success=True,
    ):
        self.success = success


class FakeWorker:
    def __init__(
        self,
        *,
        success=True,
        raises=False,
    ):
        self.success = success
        self.raises = raises

    def execute(
        self,
        job,
        state,
    ):
        if self.raises:
            raise RuntimeError(
                "Worker exploded."
            )

        if self.success:
            job.status = (
                JobStatus.COMPLETED
            )
        else:
            job.status = (
                JobStatus.FAILED
            )

        return FakeResult(
            success=self.success
        )

    def cancel(
        self,
        job,
    ):
        job.status = (
            JobStatus.CANCELLED
        )

        return job

    def prepare_retry(
        self,
        job,
    ):
        job.status = (
            JobStatus.QUEUED
        )

        return job


def build_state():
    return NexusState(
        user_request=(
            "Test event telemetry."
        )
    )


def build_manager(
    *,
    success=True,
    raises=False,
):
    bus = EventBus()

    manager = JobManager(
        queue=PriorityJobQueue(),
        worker=FakeWorker(
            success=success,
            raises=raises,
        ),
        event_bus=bus,
    )

    return manager, bus


def test_submit_publishes_job_queued():
    manager, bus = (
        build_manager()
    )

    state = build_state()

    job = manager.submit(
        state
    )

    page = bus.history()

    assert page.count == 1

    event = page.events[0]

    assert (
        event.type
        == EventType.JOB_QUEUED
    )

    assert (
        event.run_id
        == state.run_id
    )

    assert (
        event.job_id
        == job.id
    )


def test_execute_publishes_started_and_completed():
    manager, bus = (
        build_manager()
    )

    state = build_state()

    job = manager.submit(
        state
    )

    manager.execute_job(
        job.id
    )

    types = [
        event.type
        for event
        in bus.history().events
    ]

    assert types == [
        EventType.JOB_QUEUED,
        EventType.JOB_STARTED,
        EventType.JOB_COMPLETED,
    ]


def test_unsuccessful_result_publishes_failed():
    manager, bus = (
        build_manager(
            success=False
        )
    )

    state = build_state()

    job = manager.submit(
        state
    )

    manager.execute_job(
        job.id
    )

    types = [
        event.type
        for event
        in bus.history().events
    ]

    assert types[-1] == (
        EventType.JOB_FAILED
    )


def test_exception_publishes_failed():
    manager, bus = (
        build_manager(
            raises=True
        )
    )

    state = build_state()

    job = manager.submit(
        state
    )

    try:
        manager.execute_job(
            job.id
        )
    except RuntimeError:
        pass

    events = (
        bus.history().events
    )

    assert (
        events[-1].type
        == EventType.JOB_FAILED
    )

    assert (
        events[-1].payload[
            "error"
        ]
        == "Worker exploded."
    )


def test_cancel_publishes_cancelled():
    manager, bus = (
        build_manager()
    )

    state = build_state()

    job = manager.submit(
        state
    )

    manager.cancel(
        job.id
    )

    assert (
        bus.history()
        .events[-1]
        .type
        == EventType.JOB_CANCELLED
    )


def test_retry_publishes_retried():
    manager, bus = (
        build_manager(
            success=False
        )
    )

    state = build_state()

    job = manager.submit(
        state,
        max_attempts=2,
    )

    result = manager.execute_job(
        job.id
    )

    assert (
        result.success
        is False
    )

    assert (
        job.status
        == JobStatus.FAILED
    )

    manager.retry(
        job.id
    )

    assert (
        bus.history()
        .events[-1]
        .type
        == EventType.JOB_RETRIED
    )

    assert (
        job.status
        == JobStatus.QUEUED
    )

def test_no_event_bus_remains_supported():
    manager = JobManager(
        queue=PriorityJobQueue(),
        worker=FakeWorker(),
    )

    state = build_state()

    job = manager.submit(
        state,
        priority=(
            JobPriority.NORMAL
        ),
    )

    result = manager.execute_job(
        job.id
    )

    assert result.success is True
