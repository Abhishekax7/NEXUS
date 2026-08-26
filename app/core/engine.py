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
from app.memory.manager import MemoryManager


class WorkflowStalled(Exception):
    """Raised when unfinished tasks exist but none are runnable."""


class WorkflowRepairFailed(Exception):
    """Raised when autonomous repair cannot recover failing code."""


class NexusEngine:
    def __init__(
        self,
        registry: AgentRegistry,
        repair_loop: Optional[RepairLoop] = None,
        memory_manager: Optional[MemoryManager] = None,
    ):
        self.registry = registry

        self.runner = AgentRunner(
            registry
        )

        self.repair_loop = repair_loop
        self.memory_manager = memory_manager

    def _remember_task(
        self,
        task,
        state: NexusState,
    ) -> None:
        if self.memory_manager is None:
            return

        self.memory_manager.remember_task(
            task,
            state,
        )

    def _remember_artifact(
        self,
        artifact,
        state: NexusState,
    ) -> None:
        if self.memory_manager is None:
            return

        self.memory_manager.record_important_artifact(
            artifact,
            state,
        )

    def _remember_repair_artifacts(
        self,
        repair_result,
        state: NexusState,
    ) -> None:
        if self.memory_manager is None:
            return

        for debug_artifact in (
            repair_result.debug_artifacts
        ):
            self.memory_manager.record_repair(
                debug_artifact,
                state,
            )

    def _handle_test_result(
        self,
        task,
        artifact,
        state: NexusState,
    ) -> None:
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

        self._remember_repair_artifacts(
            repair_result,
            state,
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

        self._remember_artifact(
            final_test_artifact,
            state,
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

                    self._remember_task(
                        task,
                        state,
                    )

                    self._remember_artifact(
                        artifact,
                        state,
                    )

                    self._handle_test_result(
                        task,
                        artifact,
                        state,
                    )

                except Exception:
                    state.failed = True

                    self._remember_task(
                        task,
                        state,
                    )

                    raise

            state.iteration += 1

        state.completed = True
        state.failed = False

        return state
