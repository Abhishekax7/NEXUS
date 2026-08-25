from app.agents.registry import AgentRegistry
from app.core.execution import (
    complete_task,
    fail_task,
    start_task,
)
from app.core.models import AgentTask
from app.core.state import NexusState


class AgentRunner:
    def __init__(self, registry: AgentRegistry):
        self.registry = registry

    def run_task(
        self,
        task: AgentTask,
        state: NexusState,
    ):
        agent = self.registry.get_agent(
            task.assigned_agent
        )

        start_task(task)

        state.active_task_id = task.id

        try:
            artifact = agent.execute(
                task,
                state,
            )

            state.add_artifact(artifact)

            task.output_artifact_ids.append(
                artifact.id
            )

            complete_task(task)

            state.active_task_id = None

            return artifact

        except Exception as exc:
            fail_task(
                task,
                str(exc),
            )

            state.errors.append(
                f"{task.id}: {exc}"
            )

            state.active_task_id = None

            raise
