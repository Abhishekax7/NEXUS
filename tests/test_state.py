from app.core.models import AgentRole, AgentTask, TaskStatus
from app.core.state import NexusState


def test_nexus_state_can_store_task():
    state = NexusState(
        user_request="Build a RAG application"
    )

    task = AgentTask(
        title="Analyze requirements",
        description="Understand the user's requested application.",
        assigned_agent=AgentRole.REQUIREMENTS,
    )

    state.add_task(task)

    assert task.id in state.tasks
    assert state.tasks[task.id].status == TaskStatus.PENDING
    assert state.completed is False
