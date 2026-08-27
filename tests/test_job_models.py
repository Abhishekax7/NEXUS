import pytest

from app.jobs.models import (
    JobPriority,
    JobStatus,
    WorkflowJob,
)


def test_job_defaults_to_queued():
    job = WorkflowJob(
        run_id="run-1"
    )

    assert (
        job.status
        == JobStatus.QUEUED
    )


def test_job_defaults_to_normal_priority():
    job = WorkflowJob(
        run_id="run-1"
    )

    assert (
        job.priority
        == JobPriority.NORMAL
    )


def test_job_has_unique_id():
    first = WorkflowJob(
        run_id="run-1"
    )

    second = WorkflowJob(
        run_id="run-2"
    )

    assert (
        first.id
        != second.id
    )


def test_job_tracks_run_id():
    job = WorkflowJob(
        run_id="run-abc"
    )

    assert (
        job.run_id
        == "run-abc"
    )


def test_job_starts_without_execution_times():
    job = WorkflowJob(
        run_id="run-1"
    )

    assert (
        job.started_at
        is None
    )

    assert (
        job.completed_at
        is None
    )


def test_job_attempt_starts_at_zero():
    job = WorkflowJob(
        run_id="run-1"
    )

    assert (
        job.attempt
        == 0
    )


def test_default_max_attempts_is_one():
    job = WorkflowJob(
        run_id="run-1"
    )

    assert (
        job.max_attempts
        == 1
    )


def test_custom_priority_is_supported():
    job = WorkflowJob(
        run_id="run-1",
        priority=(
            JobPriority.HIGH
        ),
    )

    assert (
        job.priority
        == JobPriority.HIGH
    )


def test_priority_order_is_deterministic():
    assert (
        JobPriority.CRITICAL.value
        < JobPriority.HIGH.value
        < JobPriority.NORMAL.value
        < JobPriority.LOW.value
    )


def test_empty_run_id_is_rejected():
    with pytest.raises(
        Exception
    ):
        WorkflowJob(
            run_id=""
        )


def test_invalid_max_attempts_is_rejected():
    with pytest.raises(
        Exception
    ):
        WorkflowJob(
            run_id="run-1",
            max_attempts=0,
        )


def test_metadata_is_isolated():
    first = WorkflowJob(
        run_id="run-1"
    )

    second = WorkflowJob(
        run_id="run-2"
    )

    first.metadata[
        "source"
    ] = "api"

    assert (
        "source"
        not in second.metadata
    )
