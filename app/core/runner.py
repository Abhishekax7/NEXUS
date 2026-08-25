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

    def _attach_dependency_artifacts(
        self,
        task: AgentTask,
        state: NexusState,
    ) -> None:
        """
        Attach artifacts produced by dependency tasks
        as explicit inputs to the current task.
        """

        for dependency_id in task.dependencies:
            dependency_task = state.get_task(
                dependency_id
            )

            if dependency_task is None:
                continue

            for artifact_id in (
                dependency_task.output_artifact_ids
            ):
                if (
                    artifact_id
                    not in task.input_artifact_ids
                ):
                    task.input_artifact_ids.append(
                        artifact_id
                    )

    def run_task(
        self,
        task: AgentTask,
        state: NexusState,
    ):
        agent = self.registry.get_agent(
            task.assigned_agent
        )

        self._attach_dependency_artifacts(
            task,
            state,
        )

        start_task(task)

        state.active_task_id = task.id

        try:
            artifact = agent.execute(
                task,
                state,
            )

            state.add_artifact(
                artifact
            )

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
