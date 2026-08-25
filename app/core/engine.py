from typing import Optional

from app.agents.registry import AgentRegistry
from app.core.models import AgentRole
from app.core.repair_loop import RepairLoop
from app.core.runner import AgentRunner
from app.core.scheduler import (
    all_tasks_completed,
    get_ready_tasks,
)
from app.core.state import NexusState


class WorkflowStalled(Exception):
    """Raised when unfinished tasks exist but none are runnable."""


class WorkflowRepairFailed(Exception):
    """Raised when autonomous repair cannot recover failing code."""


class NexusEngine:
    def __init__(
        self,
        registry: AgentRegistry,
        repair_loop: Optional[RepairLoop] = None,
    ):
        self.registry = registry

        self.runner = AgentRunner(
            registry
        )

        self.repair_loop = repair_loop

    def _handle_test_result(
        self,
        task,
        artifact,
        state: NexusState,
    ) -> None:
        """
        If the Tester Agent reports failure and a repair
        loop is configured, automatically invoke bounded
        self-healing.
        """

        if (
            task.assigned_agent
            != AgentRole.TESTER
        ):
            return

        if artifact.content.get(
            "passed"
        ) is True:
            return

        if self.repair_loop is None:
            return

        repair_result = (
            self.repair_loop.run(
                state
            )
        )

        final_test_artifact = (
            repair_result.final_test_artifact
        )

        if (
            final_test_artifact.id
            not in state.artifacts
        ):
            state.add_artifact(
                final_test_artifact
            )

        if (
            final_test_artifact.id
            not in task.output_artifact_ids
        ):
            task.output_artifact_ids.append(
                final_test_artifact.id
            )

        if not repair_result.passed:
            state.failed = True

            message = (
                "Autonomous repair exhausted its "
                f"retry budget after "
                f"{repair_result.attempts} repair attempts."
            )

            state.errors.append(
                message
            )

            raise WorkflowRepairFailed(
                message
            )

    def run(
        self,
        state: NexusState,
    ) -> NexusState:

        while not all_tasks_completed(
            state
        ):
            ready_tasks = get_ready_tasks(
                state
            )

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
                    artifact = (
                        self.runner.run_task(
                            task,
                            state,
                        )
                    )

                    self._handle_test_result(
                        task,
                        artifact,
                        state,
                    )

                except Exception:
                    state.failed = True
                    raise

            state.iteration += 1

        state.completed = True
        state.failed = False

        return state
