import pytest

from app.jobs.models import (
    JobNotFoundError,
    JobPriority,
    JobStateError,
    JobStatus,
    WorkflowJob,
)

from app.jobs.queue import (
    PriorityJobQueue,
)


def build_job(
    name,
    priority=(
        JobPriority.NORMAL
    ),
):
    return WorkflowJob(
        run_id=name,
        priority=priority,
    )


def test_queue_starts_empty():
    queue = PriorityJobQueue()

    assert queue.empty() is True

    assert queue.size() == 0


def test_job_can_be_enqueued():
    queue = PriorityJobQueue()

    job = build_job(
        "run-1"
    )

    queue.enqueue(
        job
    )

    assert (
        queue.size()
        == 1
    )

    assert (
        queue.contains(
            job.id
        )
        is True
    )


def test_dequeue_returns_job():
    queue = PriorityJobQueue()

    job = build_job(
        "run-1"
    )

    queue.enqueue(
        job
    )

    result = queue.dequeue()

    assert result is job

    assert queue.empty() is True


def test_dequeue_empty_returns_none():
    queue = PriorityJobQueue()

    assert (
        queue.dequeue()
        is None
    )


def test_critical_runs_before_high():
    queue = PriorityJobQueue()

    high = build_job(
        "high",
        JobPriority.HIGH,
    )

    critical = build_job(
        "critical",
        JobPriority.CRITICAL,
    )

    queue.enqueue(
        high
    )

    queue.enqueue(
        critical
    )

    assert (
        queue.dequeue()
        is critical
    )

    assert (
        queue.dequeue()
        is high
    )


def test_high_runs_before_normal():
    queue = PriorityJobQueue()

    normal = build_job(
        "normal",
        JobPriority.NORMAL,
    )

    high = build_job(
        "high",
        JobPriority.HIGH,
    )

    queue.enqueue(
        normal
    )

    queue.enqueue(
        high
    )

    assert (
        queue.dequeue()
        is high
    )


def test_normal_runs_before_low():
    queue = PriorityJobQueue()

    low = build_job(
        "low",
        JobPriority.LOW,
    )

    normal = build_job(
        "normal",
        JobPriority.NORMAL,
    )

    queue.enqueue(
        low
    )

    queue.enqueue(
        normal
    )

    assert (
        queue.dequeue()
        is normal
    )


def test_same_priority_preserves_fifo():
    queue = PriorityJobQueue()

    first = build_job(
        "first"
    )

    second = build_job(
        "second"
    )

    third = build_job(
        "third"
    )

    queue.enqueue(
        first
    )

    queue.enqueue(
        second
    )

    queue.enqueue(
        third
    )

    assert (
        queue.dequeue()
        is first
    )

    assert (
        queue.dequeue()
        is second
    )

    assert (
        queue.dequeue()
        is third
    )


def test_full_priority_order():
    queue = PriorityJobQueue()

    low = build_job(
        "low",
        JobPriority.LOW,
    )

    normal = build_job(
        "normal",
        JobPriority.NORMAL,
    )

    high = build_job(
        "high",
        JobPriority.HIGH,
    )

    critical = build_job(
        "critical",
        JobPriority.CRITICAL,
    )

    queue.enqueue(
        low
    )

    queue.enqueue(
        normal
    )

    queue.enqueue(
        high
    )

    queue.enqueue(
        critical
    )

    result = [
        queue.dequeue().priority
        for _ in range(4)
    ]

    assert result == [
        JobPriority.CRITICAL,
        JobPriority.HIGH,
        JobPriority.NORMAL,
        JobPriority.LOW,
    ]


def test_peek_does_not_remove_job():
    queue = PriorityJobQueue()

    job = build_job(
        "run-1"
    )

    queue.enqueue(
        job
    )

    assert queue.peek() is job

    assert (
        queue.size()
        == 1
    )


def test_remove_deletes_job():
    queue = PriorityJobQueue()

    job = build_job(
        "run-1"
    )

    queue.enqueue(
        job
    )

    removed = queue.remove(
        job.id
    )

    assert removed is job

    assert (
        queue.contains(
            job.id
        )
        is False
    )


def test_remove_missing_job_fails():
    queue = PriorityJobQueue()

    with pytest.raises(
        JobNotFoundError
    ):
        queue.remove(
            "missing"
        )


def test_duplicate_job_is_rejected():
    queue = PriorityJobQueue()

    job = build_job(
        "run-1"
    )

    queue.enqueue(
        job
    )

    with pytest.raises(
        JobStateError
    ):
        queue.enqueue(
            job
        )


def test_non_queued_job_is_rejected():
    queue = PriorityJobQueue()

    job = build_job(
        "run-1"
    )

    job.status = (
        JobStatus.RUNNING
    )

    with pytest.raises(
        JobStateError
    ):
        queue.enqueue(
            job
        )


def test_clear_returns_removed_count():
    queue = PriorityJobQueue()

    for index in range(3):
        queue.enqueue(
            build_job(
                f"run-{index}"
            )
        )

    removed = queue.clear()

    assert removed == 3

    assert queue.empty() is True


def test_queued_jobs_returns_execution_order():
    queue = PriorityJobQueue()

    low = build_job(
        "low",
        JobPriority.LOW,
    )

    high = build_job(
        "high",
        JobPriority.HIGH,
    )

    normal = build_job(
        "normal",
        JobPriority.NORMAL,
    )

    queue.enqueue(
        low
    )

    queue.enqueue(
        high
    )

    queue.enqueue(
        normal
    )

    ordered = queue.queued_jobs()

    assert [
        job.priority
        for job
        in ordered
    ] == [
        JobPriority.HIGH,
        JobPriority.NORMAL,
        JobPriority.LOW,
    ]
