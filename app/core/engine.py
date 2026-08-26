from typing import Optional

from app.agents.registry import AgentRegistry
from app.agents.replanner import ReplannerAgent
from app.core.models import AgentRole
from app.core.plan_mutator import (
    PlanMutator,
    PlanMutationResult,
)
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


class ReplanningLimitExceeded(Exception):
    """Raised when the workflow exceeds its replan budget."""


class NexusEngine:
    def __init__(
        self,
        registry: AgentRegistry,
        repair_loop: Optional[RepairLoop] = None,
        memory_manager: Optional[MemoryManager] = None,
        replanner: Optional[ReplannerAgent] = None,
        plan_mutator: Optional[PlanMutator] = None,
        max_replans: int = 3,
    ):
        if max_replans < 0:
            raise ValueError(
                "max_replans cannot be negative."
            )

        self.registry = registry

        self.runner = AgentRunner(
            registry
        )

        self.repair_loop = repair_loop
        self.memory_manager = memory_manager

        self.replanner = replanner

        self.plan_mutator = (
            plan_mutator
            or PlanMutator()
        )

        self.max_replans = max_replans

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

    def _get_replan_count(
        self,
        state: NexusState,
    ) -> int:
        return int(
            state.metadata.get(
                "replan_count",
                0,
            )
        )

    def _increment_replan_count(
        self,
        state: NexusState,
    ) -> int:
        count = (
            self._get_replan_count(
                state
            )
            + 1
        )

        state.metadata[
            "replan_count"
        ] = count

        return count

    def _record_replan_result(
        self,
        result: PlanMutationResult,
        state: NexusState,
    ) -> None:
        history = state.metadata.setdefault(
            "replan_history",
            [],
        )

        history.append(
            {
                "action": result.action.value,
                "added_task_id":
                    result.added_task_id,
                "removed_task_id":
                    result.removed_task_id,
                "replaced_task_id":
                    result.replaced_task_id,
            }
        )

    def _maybe_replan(
        self,
        state: NexusState,
    ) -> None:
        if self.replanner is None:
            return

        decision = self.replanner.decide(
            state
        )

        if not decision.should_replan:
            return

        current_count = (
            self._get_replan_count(
                state
            )
        )

        if (
            current_count
            >= self.max_replans
        ):
            state.failed = True

            message = (
                "NEXUS exceeded the maximum "
                f"replanning budget of "
                f"{self.max_replans}."
            )

            state.errors.append(
                message
            )

            raise ReplanningLimitExceeded(
                message
            )

        result = (
            self.plan_mutator.apply(
                decision,
                state,
            )
        )

        self._increment_replan_count(
            state
        )

        self._record_replan_result(
            result,
            state,
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

            self._maybe_replan(
                state
            )

        state.completed = True
        state.failed = False

        return state
