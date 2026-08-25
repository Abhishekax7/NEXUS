from typing import List

from app.core.models import AgentTask, TaskStatus
from app.core.state import NexusState


def dependencies_completed(
    task: AgentTask,
    state: NexusState,
) -> bool:
    for dependency_id in task.dependencies:
        dependency = state.get_task(dependency_id)

        if dependency is None:
            return False

        if dependency.status != TaskStatus.COMPLETED:
            return False

    return True


def get_ready_tasks(state: NexusState) -> List[AgentTask]:
    ready_tasks = []

    for task in state.tasks.values():
        if task.status != TaskStatus.PENDING:
            continue

        if dependencies_completed(task, state):
            ready_tasks.append(task)

    return ready_tasks


def get_blocked_tasks(state: NexusState) -> List[AgentTask]:
    blocked_tasks = []

    for task in state.tasks.values():
        if task.status not in {
            TaskStatus.PENDING,
            TaskStatus.BLOCKED,
        }:
            continue

        if not dependencies_completed(task, state):
            blocked_tasks.append(task)

    return blocked_tasks


def has_unfinished_tasks(state: NexusState) -> bool:
    return any(
        task.status != TaskStatus.COMPLETED
        for task in state.tasks.values()
    )


def all_tasks_completed(state: NexusState) -> bool:
    if not state.tasks:
        return False

    return all(
        task.status == TaskStatus.COMPLETED
        for task in state.tasks.values()
    )
