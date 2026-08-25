import pytest

from app.core.execution import (
    InvalidTaskTransition,
    complete_task,
    fail_task,
    retry_task,
    start_task,
    transition_task,
)

from app.core.models import AgentRole, AgentTask, TaskStatus


def create_task() -> AgentTask:
    return AgentTask(
        title="Design architecture",
        description="Create the system architecture.",
        assigned_agent=AgentRole.ARCHITECT,
    )


def test_pending_task_can_start():
    task = create_task()

    start_task(task)

    assert task.status == TaskStatus.RUNNING


def test_running_task_can_complete():
    task = create_task()

    start_task(task)
    complete_task(task)

    assert task.status == TaskStatus.COMPLETED


def test_running_task_can_fail():
    task = create_task()

    start_task(task)
    fail_task(task, "Architecture generation failed")

    assert task.status == TaskStatus.FAILED
    assert task.error == "Architecture generation failed"


def test_failed_task_can_retry():
    task = create_task()

    start_task(task)
    fail_task(task, "Temporary model failure")
    retry_task(task)

    assert task.status == TaskStatus.RETRYING
    assert task.retry_count == 1


def test_retrying_task_can_run_again():
    task = create_task()

    start_task(task)
    fail_task(task, "Temporary failure")
    retry_task(task)
    start_task(task)

    assert task.status == TaskStatus.RUNNING


def test_completed_task_cannot_restart():
    task = create_task()

    start_task(task)
    complete_task(task)

    with pytest.raises(InvalidTaskTransition):
        start_task(task)


def test_retry_limit_is_enforced():
    task = create_task()

    task.max_retries = 1

    start_task(task)
    fail_task(task, "First failure")
    retry_task(task)

    transition_task(task, TaskStatus.RUNNING)
    fail_task(task, "Second failure")

    with pytest.raises(InvalidTaskTransition):
        retry_task(task)
