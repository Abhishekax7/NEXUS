from app.core.models import (
    AgentTask,
    TaskStatus,
)
from app.core.replanning import (
    ProposedTask,
    ReplanAction,
    ReplanningDecision,
    ReplanningError,
    validate_replanning_decision,
)
from app.core.state import NexusState


class PlanMutationError(Exception):
    """Raised when a replanning decision cannot be applied safely."""


class PlanMutationResult:
    def __init__(
        self,
        action: ReplanAction,
        added_task_id: str | None = None,
        removed_task_id: str | None = None,
        replaced_task_id: str | None = None,
    ):
        self.action = action
        self.added_task_id = added_task_id
        self.removed_task_id = removed_task_id
        self.replaced_task_id = replaced_task_id


class PlanMutator:
    def _resolve_dependency_ids(
        self,
        proposed_task: ProposedTask,
        state: NexusState,
    ) -> list[str]:
        dependency_ids = []

        for role in proposed_task.depends_on_roles:
            candidates = [
                task
                for task in state.tasks.values()
                if task.assigned_agent == role
            ]

            if not candidates:
                raise PlanMutationError(
                    "No task exists for dependency role: "
                    f"{role.value}"
                )

            completed = [
                task
                for task in candidates
                if task.status == TaskStatus.COMPLETED
            ]

            if completed:
                selected = completed[-1]
            else:
                selected = candidates[-1]

            dependency_ids.append(
                selected.id
            )

        return dependency_ids

    def _create_task(
        self,
        proposed_task: ProposedTask,
        state: NexusState,
    ) -> AgentTask:
        dependencies = (
            self._resolve_dependency_ids(
                proposed_task,
                state,
            )
        )

        return AgentTask(
            title=proposed_task.title,
            description=proposed_task.description,
            assigned_agent=proposed_task.assigned_agent,
            dependencies=dependencies,
            metadata=dict(
                proposed_task.metadata
            ),
        )

    def _ensure_target_mutable(
        self,
        task: AgentTask,
    ) -> None:
        if task.status == TaskStatus.RUNNING:
            raise PlanMutationError(
                "Cannot mutate a running task."
            )

        if task.status == TaskStatus.COMPLETED:
            raise PlanMutationError(
                "Cannot mutate a completed task."
            )

    def _ensure_no_dependents(
        self,
        target_task_id: str,
        state: NexusState,
    ) -> None:
        dependents = [
            task
            for task in state.tasks.values()
            if target_task_id
            in task.dependencies
        ]

        if dependents:
            titles = ", ".join(
                task.title
                for task in dependents
            )

            raise PlanMutationError(
                "Cannot remove task with dependents: "
                f"{titles}"
            )

    def _replace_dependencies(
        self,
        old_task_id: str,
        new_task_id: str,
        state: NexusState,
    ) -> None:
        for task in state.tasks.values():
            if old_task_id not in task.dependencies:
                continue

            task.dependencies = [
                (
                    new_task_id
                    if dependency_id == old_task_id
                    else dependency_id
                )
                for dependency_id
                in task.dependencies
            ]

    def _apply_add(
        self,
        decision: ReplanningDecision,
        state: NexusState,
    ) -> PlanMutationResult:
        proposed = decision.proposed_task

        if proposed is None:
            raise PlanMutationError(
                "ADD_TASK requires proposed_task."
            )

        new_task = self._create_task(
            proposed,
            state,
        )

        state.add_task(
            new_task
        )

        return PlanMutationResult(
            action=ReplanAction.ADD_TASK,
            added_task_id=new_task.id,
        )

    def _apply_remove(
        self,
        decision: ReplanningDecision,
        state: NexusState,
    ) -> PlanMutationResult:
        target_id = (
            decision.target_task_id
        )

        if not target_id:
            raise PlanMutationError(
                "REMOVE_TASK requires target_task_id."
            )

        target = state.tasks.get(
            target_id
        )

        if target is None:
            raise PlanMutationError(
                "Target task does not exist."
            )

        self._ensure_target_mutable(
            target
        )

        self._ensure_no_dependents(
            target_id,
            state,
        )

        del state.tasks[
            target_id
        ]

        if (
            state.active_task_id
            == target_id
        ):
            state.active_task_id = None

        return PlanMutationResult(
            action=ReplanAction.REMOVE_TASK,
            removed_task_id=target_id,
        )

    def _apply_replace(
        self,
        decision: ReplanningDecision,
        state: NexusState,
    ) -> PlanMutationResult:
        target_id = (
            decision.target_task_id
        )

        proposed = (
            decision.proposed_task
        )

        if not target_id:
            raise PlanMutationError(
                "REPLACE_TASK requires target_task_id."
            )

        if proposed is None:
            raise PlanMutationError(
                "REPLACE_TASK requires proposed_task."
            )

        target = state.tasks.get(
            target_id
        )

        if target is None:
            raise PlanMutationError(
                "Target task does not exist."
            )

        self._ensure_target_mutable(
            target
        )

        new_task = self._create_task(
            proposed,
            state,
        )

        state.add_task(
            new_task
        )

        self._replace_dependencies(
            old_task_id=target_id,
            new_task_id=new_task.id,
            state=state,
        )

        del state.tasks[
            target_id
        ]

        if (
            state.active_task_id
            == target_id
        ):
            state.active_task_id = None

        return PlanMutationResult(
            action=ReplanAction.REPLACE_TASK,
            added_task_id=new_task.id,
            removed_task_id=target_id,
            replaced_task_id=target_id,
        )

    def apply(
        self,
        decision: ReplanningDecision,
        state: NexusState,
    ) -> PlanMutationResult:
        try:
            validate_replanning_decision(
                decision
            )
        except ReplanningError as exc:
            raise PlanMutationError(
                str(exc)
            ) from exc

        if not decision.should_replan:
            return PlanMutationResult(
                action=ReplanAction.KEEP_PLAN
            )

        if (
            decision.action
            == ReplanAction.ADD_TASK
        ):
            return self._apply_add(
                decision,
                state,
            )

        if (
            decision.action
            == ReplanAction.REMOVE_TASK
        ):
            return self._apply_remove(
                decision,
                state,
            )

        if (
            decision.action
            == ReplanAction.REPLACE_TASK
        ):
            return self._apply_replace(
                decision,
                state,
            )

        raise PlanMutationError(
            "Unsupported replanning action."
        )
