from app.agents.registry import AgentRegistry
from app.core.runner import AgentRunner
from app.core.scheduler import (
    all_tasks_completed,
    get_ready_tasks,
)
from app.core.state import NexusState


class WorkflowStalled(Exception):
    """Raised when unfinished tasks exist but none are runnable."""


class NexusEngine:
    def __init__(self, registry: AgentRegistry):
        self.registry = registry
        self.runner = AgentRunner(registry)

    def run(
        self,
        state: NexusState,
    ) -> NexusState:

        while not all_tasks_completed(state):
            ready_tasks = get_ready_tasks(state)

            if not ready_tasks:
                state.failed = True
                state.errors.append(
                    "Workflow stalled: unfinished tasks exist "
                    "but no tasks are ready."
                )

                raise WorkflowStalled(
                    "NEXUS workflow stalled."
                )

            for task in ready_tasks:
                try:
                    self.runner.run_task(
                        task,
                        state,
                    )

                except Exception:
                    state.failed = True
                    raise

            state.iteration += 1

        state.completed = True
        state.failed = False

        return state
