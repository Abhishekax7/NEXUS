from typing import Optional

from app.agents.registry import AgentRegistry
from app.agents.replanner import ReplannerAgent

from app.checkpointing.models import (
    CheckpointError,
    RecoveryStatus,
)
from app.checkpointing.service import (
    CheckpointService,
)

from app.core.models import (
    AgentRole,
    ArtifactType,
    TaskStatus,
)
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

from app.tools.executor import (
    CommandExecutor,
    ExecutionError,
)

from app.evaluation.service import (
    EvaluationService,
    EvaluationServiceResult,
)

from app.memory.manager import MemoryManager

from app.observability.collector import (
    TraceCollector,
)
from app.observability.service import (
    ObservabilityService,
)


class WorkflowStalled(Exception):
    """
    Raised when unfinished tasks exist
    but none are runnable.
    """


class WorkflowRepairFailed(Exception):
    """
    Raised when autonomous repair
    cannot recover failing code.
    """


class ReplanningLimitExceeded(Exception):
    """
    Raised when the workflow exceeds
    its replan budget.
    """


class NexusEngine:
    def __init__(
        self,
        registry: AgentRegistry,
        repair_loop: Optional[
            RepairLoop
        ] = None,
        memory_manager: Optional[
            MemoryManager
        ] = None,
        replanner: Optional[
            ReplannerAgent
        ] = None,
        plan_mutator: Optional[
            PlanMutator
        ] = None,
        max_replans: int = 3,
        evaluation_service: Optional[
            EvaluationService
        ] = None,
        observability_service: Optional[
            ObservabilityService
        ] = None,
        checkpoint_service: Optional[
            CheckpointService
        ] = None,
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

        self.memory_manager = (
            memory_manager
        )

        self.replanner = replanner

        self.plan_mutator = (
            plan_mutator
            or PlanMutator()
        )

        self.max_replans = (
            max_replans
        )

        self.evaluation_service = (
            evaluation_service
        )

        self.observability_service = (
            observability_service
        )

        self.checkpoint_service = (
            checkpoint_service
        )

        self.last_evaluation_result: Optional[
            EvaluationServiceResult
        ] = None

        self.last_trace_collector: Optional[
            TraceCollector
        ] = None

        self.tool_registry = None
        self.tool_runtime = None

        self.approval_manager = None
        self.approval_gate = None

    def _agent_role_value(
        self,
        role,
    ) -> str:
        value = getattr(
            role,
            "value",
            role,
        )

        return str(
            value
        )

    def _artifact_type_value(
        self,
        artifact,
    ) -> Optional[str]:
        artifact_type = getattr(
            artifact,
            "type",
            None,
        )

        if artifact_type is None:
            return None

        value = getattr(
            artifact_type,
            "value",
            artifact_type,
        )

        return str(
            value
        )

    def _start_trace(
        self,
        state: NexusState,
    ) -> Optional[
        TraceCollector
    ]:
        if (
            self.observability_service
            is None
        ):
            return None

        collector = (
            self.observability_service
            .start_run(
                state.run_id,
                task_count=len(
                    state.tasks
                ),
            )
        )

        self.last_trace_collector = (
            collector
        )

        return collector

    def _save_trace(
        self,
        collector: Optional[
            TraceCollector
        ],
    ) -> None:
        if (
            collector is None
            or self.observability_service
            is None
        ):
            return

        self.observability_service.save(
            collector.trace
        )

    def _checkpoint_started(
        self,
        state: NexusState,
    ) -> None:
        if self.checkpoint_service is None:
            return

        self.checkpoint_service.workflow_started(
            state
        )

    def _checkpoint_task_completed(
        self,
        state: NexusState,
        task_id: str,
    ) -> None:
        if self.checkpoint_service is None:
            return

        self.checkpoint_service.task_completed(
            state,
            task_id,
        )

    def _checkpoint_iteration_completed(
        self,
        state: NexusState,
    ) -> None:
        if self.checkpoint_service is None:
            return

        self.checkpoint_service.iteration_completed(
            state
        )

    def _checkpoint_repair_completed(
        self,
        state: NexusState,
        *,
        attempts: int,
        passed: bool,
    ) -> None:
        if self.checkpoint_service is None:
            return

        self.checkpoint_service.repair_completed(
            state,
            attempts=attempts,
            passed=passed,
        )

    def _checkpoint_replan_completed(
        self,
        state: NexusState,
        *,
        action: str,
    ) -> None:
        if self.checkpoint_service is None:
            return

        self.checkpoint_service.replan_completed(
            state,
            action=action,
        )

    def _checkpoint_completed(
        self,
        state: NexusState,
    ) -> None:
        if self.checkpoint_service is None:
            return

        self.checkpoint_service.workflow_completed(
            state
        )

    def _checkpoint_failed(
        self,
        state: NexusState,
        *,
        reason: str,
    ) -> None:
        if self.checkpoint_service is None:
            return

        self.checkpoint_service.workflow_failed(
            state,
            reason=reason,
        )

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
        collector: Optional[
            TraceCollector
        ] = None,
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

        if collector is not None:
            collector.repair_started()

        try:
            repair_result = (
                self.repair_loop.run(
                    state
                )
            )

        except Exception as exc:
            if collector is not None:
                collector.repair_failed(
                    str(exc)
                )

                self._save_trace(
                    collector
                )

            raise

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

        if collector is not None:
            collector.artifact_created(
                artifact_id=(
                    final_test_artifact.id
                ),
                agent_role=(
                    self._agent_role_value(
                        AgentRole.TESTER
                    )
                ),
                artifact_type=(
                    self._artifact_type_value(
                        final_test_artifact
                    )
                ),
            )

            collector.repair_completed(
                passed=(
                    repair_result.passed
                ),
                attempts=(
                    repair_result.attempts
                ),
            )

            self._save_trace(
                collector
            )

        self._checkpoint_repair_completed(
            state,
            attempts=(
                repair_result.attempts
            ),
            passed=(
                repair_result.passed
            ),
        )

        if not repair_result.passed:
            state.failed = True

            message = (
                "Autonomous repair exhausted its "
                f"retry budget after "
                f"{repair_result.attempts} "
                "repair attempts."
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
        history = (
            state.metadata.setdefault(
                "replan_history",
                [],
            )
        )

        history.append(
            {
                "action":
                    result.action.value,
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
        collector: Optional[
            TraceCollector
        ] = None,
    ) -> None:
        if self.replanner is None:
            return

        decision = (
            self.replanner.decide(
                state
            )
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

        if collector is not None:
            collector.replan_started()

        try:
            result = (
                self.plan_mutator.apply(
                    decision,
                    state,
                )
            )

        except Exception:
            self._save_trace(
                collector
            )

            raise

        self._increment_replan_count(
            state
        )

        self._record_replan_result(
            result,
            state,
        )

        if collector is not None:
            collector.replan_completed(
                action=(
                    result.action.value
                )
            )

            self._save_trace(
                collector
            )

        self._checkpoint_replan_completed(
            state,
            action=(
                result.action.value
            ),
        )

    def _evaluate_completed_run(
        self,
        state: NexusState,
        collector: Optional[
            TraceCollector
        ] = None,
    ) -> None:
        if self.evaluation_service is None:
            return

        result = (
            self.evaluation_service
            .evaluate_run(
                state
            )
        )

        self.last_evaluation_result = (
            result
        )

        state.metadata[
            "evaluation"
        ] = (
            result.evaluation.model_dump(
                mode="json"
            )
        )

        state.metadata[
            "evaluation_baseline_run_id"
        ] = (
            result.baseline_run_id
        )

        state.metadata[
            "evaluation_baseline_created"
        ] = (
            result.baseline_created
        )

        if result.benchmark is None:
            state.metadata[
                "evaluation_benchmark"
            ] = None

        else:
            state.metadata[
                "evaluation_benchmark"
            ] = (
                result.benchmark.model_dump(
                    mode="json"
                )
            )

        if collector is not None:
            regression_detected = False

            if result.benchmark is not None:
                regression_detected = bool(
                    result.benchmark
                    .regression_detected
                )

            collector.evaluation_completed(
                overall_score=(
                    result.evaluation
                    .overall_score
                ),
                regression_detected=(
                    regression_detected
                ),
            )

            self._save_trace(
                collector
            )

    def _task_depends_on_any(
        self,
        task,
        task_ids: set[str],
        state: NexusState,
    ) -> bool:
        """
        Return True when a task directly or
        transitively depends on one of the
        supplied task IDs.
        """

        pending = list(
            task.dependencies
        )

        visited: set[str] = set()

        while pending:
            dependency_id = (
                pending.pop()
            )

            if dependency_id in visited:
                continue

            visited.add(
                dependency_id
            )

            if dependency_id in task_ids:
                return True

            dependency_task = (
                state.tasks.get(
                    dependency_id
                )
            )

            if dependency_task is not None:
                pending.extend(
                    dependency_task.dependencies
                )

        return False

    def _invalidate_runtime_incompatible_code(
        self,
        state: NexusState,
    ) -> dict:
        """
        Validate restored CODE artifacts
        against the current execution policy.

        Older checkpoints may contain commands
        that were previously generated but are
        no longer executable by the current
        NEXUS runtime.

        When such an artifact is found, reset
        its producing Coder task and every
        downstream dependent task so the
        scheduler can regenerate compatible
        work.
        """

        executor = CommandExecutor()

        invalid_task_ids: set[str] = set()
        invalid_artifact_ids: set[str] = set()
        reasons: list[str] = []

        for task in state.tasks.values():
            if (
                task.assigned_agent
                != AgentRole.CODER
            ):
                continue

            for artifact_id in list(
                task.output_artifact_ids
            ):
                artifact = (
                    state.artifacts.get(
                        artifact_id
                    )
                )

                if artifact is None:
                    continue

                if (
                    artifact.type
                    != ArtifactType.CODE
                ):
                    continue

                content = artifact.content

                if not isinstance(
                    content,
                    dict,
                ):
                    invalid_task_ids.add(
                        task.id
                    )

                    invalid_artifact_ids.add(
                        artifact_id
                    )

                    reasons.append(
                        "CODE artifact content "
                        "is not an object."
                    )

                    continue

                run_commands = (
                    content.get(
                        "run_commands",
                        [],
                    )
                )

                test_commands = (
                    content.get(
                        "test_commands",
                        [],
                    )
                )

                if not isinstance(
                    run_commands,
                    list,
                ):
                    invalid_task_ids.add(
                        task.id
                    )

                    invalid_artifact_ids.add(
                        artifact_id
                    )

                    reasons.append(
                        "CODE artifact has "
                        "invalid run_commands."
                    )

                    continue

                if not isinstance(
                    test_commands,
                    list,
                ):
                    invalid_task_ids.add(
                        task.id
                    )

                    invalid_artifact_ids.add(
                        artifact_id
                    )

                    reasons.append(
                        "CODE artifact has "
                        "invalid test_commands."
                    )

                    continue

                commands = [
                    *run_commands,
                    *test_commands,
                ]

                try:
                    for command in commands:
                        if not isinstance(
                            command,
                            str,
                        ):
                            raise ExecutionError(
                                "Command must be "
                                "a string."
                            )

                        executor.validate_command(
                            command
                        )

                except ExecutionError as exc:
                    invalid_task_ids.add(
                        task.id
                    )

                    invalid_artifact_ids.add(
                        artifact_id
                    )

                    reasons.append(
                        str(exc)
                    )

        if not invalid_task_ids:
            return {
                "invalidated": False,
                "task_ids": [],
                "artifact_ids": [],
                "reset_task_ids": [],
                "reasons": [],
            }

        affected_task_ids = set(
            invalid_task_ids
        )

        for task in state.tasks.values():
            if self._task_depends_on_any(
                task,
                invalid_task_ids,
                state,
            ):
                affected_task_ids.add(
                    task.id
                )

        artifacts_to_remove = set(
            invalid_artifact_ids
        )

        for task_id in affected_task_ids:
            task = state.tasks.get(
                task_id
            )

            if task is None:
                continue

            artifacts_to_remove.update(
                task.output_artifact_ids
            )

            task.output_artifact_ids = []
            task.status = TaskStatus.PENDING

        for artifact_id in (
            artifacts_to_remove
        ):
            state.artifacts.pop(
                artifact_id,
                None,
            )

        return {
            "invalidated": True,
            "task_ids": sorted(
                invalid_task_ids
            ),
            "artifact_ids": sorted(
                invalid_artifact_ids
            ),
            "reset_task_ids": sorted(
                affected_task_ids
            ),
            "reasons": reasons,
        }

    def restore_run(
        self,
        run_id: str,
        *,
        allow_failed: bool = False,
    ) -> NexusState:
        """
        Restore the latest persisted state
        for an interrupted workflow.

        Failed runs require allow_failed=True.

        Completed tasks remain completed,
        while failed tasks become pending
        so the scheduler can retry them.
        """

        if self.checkpoint_service is None:
            raise CheckpointError(
                "Checkpointing is not configured."
            )

        info = (
            self.checkpoint_service
            .recovery_info(
                run_id
            )
        )

        if (
            info.status
            == RecoveryStatus.NOT_FOUND
        ):
            raise CheckpointError(
                "No checkpoint exists for run: "
                f"{run_id}"
            )

        if (
            info.status
            == RecoveryStatus.COMPLETED
        ):
            raise CheckpointError(
                "Completed workflow cannot "
                "be resumed: "
                f"{run_id}"
            )

        if (
            info.status
            == RecoveryStatus.FAILED
            and not allow_failed
        ):
            raise CheckpointError(
                "Workflow is marked failed. "
                "Pass allow_failed=True to "
                "retry from its latest state."
            )

        state = (
            self.checkpoint_service
            .restore_state(
                run_id
            )
        )

        if state is None:
            raise CheckpointError(
                "Could not restore workflow "
                f"state for run: {run_id}"
            )

        latest = (
            self.checkpoint_service
            .latest_checkpoint(
                run_id
            )
        )

        if (
            info.status
            == RecoveryStatus.FAILED
            and allow_failed
        ):
            state.failed = False
            state.completed = False

            for task in (
                state.tasks.values()
            ):
                if (
                    task.status
                    == TaskStatus.FAILED
                ):
                    task.status = (
                        TaskStatus.PENDING
                    )

        runtime_recovery = (
            self
            ._invalidate_runtime_incompatible_code(
                state
            )
        )

        state.metadata[
            "recovered_from_checkpoint"
        ] = {
            "checkpoint_id": (
                latest.id
                if latest is not None
                else None
            ),
            "sequence": (
                latest.sequence
                if latest is not None
                else None
            ),
            "checkpoint_type": (
                latest.checkpoint_type.value
                if latest is not None
                else None
            ),
            "runtime_artifacts_invalidated": (
                runtime_recovery[
                    "invalidated"
                ]
            ),
            "runtime_invalidated_task_ids": (
                runtime_recovery[
                    "task_ids"
                ]
            ),
            "runtime_invalidated_artifact_ids": (
                runtime_recovery[
                    "artifact_ids"
                ]
            ),
            "runtime_reset_task_ids": (
                runtime_recovery[
                    "reset_task_ids"
                ]
            ),
            "runtime_invalidation_reasons": (
                runtime_recovery[
                    "reasons"
                ]
            ),
        }

        return state

    def resume_run(
        self,
        run_id: str,
        *,
        allow_failed: bool = False,
    ) -> NexusState:
        """
        Restore and continue a persisted
        NEXUS workflow.
        """

        state = self.restore_run(
            run_id,
            allow_failed=allow_failed,
        )

        return self.run(
            state
        )

    def run(
        self,
        state: NexusState,
    ) -> NexusState:
        collector = self._start_trace(
            state
        )

        self._checkpoint_started(
            state
        )

        try:
            while not all_tasks_completed(
                state
            ):
                ready_tasks = (
                    get_ready_tasks(
                        state
                    )
                )

                if not ready_tasks:
                    state.failed = True

                    message = (
                        "Workflow stalled: "
                        "unfinished tasks exist "
                        "but no tasks are ready."
                    )

                    state.errors.append(
                        message
                    )

                    raise WorkflowStalled(
                        "NEXUS workflow stalled."
                    )

                for task in ready_tasks:
                    agent_role = (
                        self._agent_role_value(
                            task.assigned_agent
                        )
                    )

                    if collector is not None:
                        collector.task_started(
                            task_id=task.id,
                            agent_role=agent_role,
                        )

                        self._save_trace(
                            collector
                        )

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

                        if collector is not None:
                            collector.artifact_created(
                                artifact_id=(
                                    artifact.id
                                ),
                                agent_role=(
                                    agent_role
                                ),
                                artifact_type=(
                                    self._artifact_type_value(
                                        artifact
                                    )
                                ),
                            )

                        self._handle_test_result(
                            task,
                            artifact,
                            state,
                            collector=collector,
                        )

                        if collector is not None:
                            collector.task_completed(
                                task_id=task.id,
                                agent_role=agent_role,
                                artifact_id=(
                                    artifact.id
                                ),
                            )

                            self._save_trace(
                                collector
                            )

                        self._checkpoint_task_completed(
                            state,
                            task.id,
                        )

                    except Exception as exc:
                        state.failed = True

                        self._remember_task(
                            task,
                            state,
                        )

                        if collector is not None:
                            collector.task_failed(
                                task_id=task.id,
                                agent_role=(
                                    agent_role
                                ),
                                message=str(
                                    exc
                                ),
                            )

                            self._save_trace(
                                collector
                            )

                        raise

                state.iteration += 1

                self._maybe_replan(
                    state,
                    collector=collector,
                )

                self._checkpoint_iteration_completed(
                    state
                )

            state.completed = True
            state.failed = False

            self._evaluate_completed_run(
                state,
                collector=collector,
            )

            if (
                collector is not None
                and self.observability_service
                is not None
            ):
                trace = (
                    self.observability_service
                    .complete_run(
                        collector
                    )
                )

                state.metadata[
                    "observability"
                ] = {
                    "run_id":
                        trace.run_id,
                    "status":
                        trace.status.value,
                    "event_count":
                        len(trace.events),
                    "total_duration_ms":
                        trace.total_duration_ms,
                    "repair_count":
                        trace.repair_count,
                    "replan_count":
                        trace.replan_count,
                    "artifact_count":
                        trace.artifact_count,
                }

            self._checkpoint_completed(
                state
            )

            return state

        except Exception as exc:
            if (
                collector is not None
                and self.observability_service
                is not None
            ):
                self.observability_service.fail_run(
                    collector,
                    str(exc),
                )

            self._checkpoint_failed(
                state,
                reason=str(
                    exc
                ),
            )

            raise
