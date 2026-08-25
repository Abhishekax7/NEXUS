from typing import Dict, Set

from app.core.models import AgentTask, TaskStatus


class InvalidTaskTransition(Exception):
    """Raised when a task attempts an illegal state transition."""


ALLOWED_TRANSITIONS: Dict[TaskStatus, Set[TaskStatus]] = {
    TaskStatus.PENDING: {
        TaskStatus.RUNNING,
        TaskStatus.BLOCKED,
        TaskStatus.WAITING_FOR_HUMAN,
        TaskStatus.FAILED,
    },

    TaskStatus.RUNNING: {
        TaskStatus.COMPLETED,
        TaskStatus.FAILED,
        TaskStatus.BLOCKED,
        TaskStatus.WAITING_FOR_HUMAN,
    },

    TaskStatus.BLOCKED: {
        TaskStatus.PENDING,
        TaskStatus.RUNNING,
        TaskStatus.FAILED,
    },

    TaskStatus.WAITING_FOR_HUMAN: {
        TaskStatus.PENDING,
        TaskStatus.RUNNING,
        TaskStatus.FAILED,
    },

    TaskStatus.FAILED: {
        TaskStatus.RETRYING,
    },

    TaskStatus.RETRYING: {
        TaskStatus.RUNNING,
        TaskStatus.FAILED,
    },

    TaskStatus.COMPLETED: set(),
}


def can_transition(
    current_status: TaskStatus,
    new_status: TaskStatus,
) -> bool:
    return new_status in ALLOWED_TRANSITIONS[current_status]


def transition_task(
    task: AgentTask,
    new_status: TaskStatus,
) -> AgentTask:

    if task.status == new_status:
        return task

    if not can_transition(task.status, new_status):
        raise InvalidTaskTransition(
            f"Illegal task transition: "
            f"{task.status.value} -> {new_status.value}"
        )

    task.status = new_status

    return task


def start_task(task: AgentTask) -> AgentTask:
    return transition_task(task, TaskStatus.RUNNING)


def complete_task(task: AgentTask) -> AgentTask:
    task.error = None
    return transition_task(task, TaskStatus.COMPLETED)


def fail_task(
    task: AgentTask,
    error: str,
) -> AgentTask:

    task.error = error

    return transition_task(task, TaskStatus.FAILED)


def retry_task(task: AgentTask) -> AgentTask:

    if task.retry_count >= task.max_retries:
        raise InvalidTaskTransition(
            f"Task {task.id} exceeded maximum retries "
            f"({task.max_retries})."
        )

    transition_task(task, TaskStatus.RETRYING)

    task.retry_count += 1

    return task
