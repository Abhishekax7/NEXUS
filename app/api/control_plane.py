from typing import Optional

from app.api.schemas import (
    ApprovalResponse,
    EvaluationResponse,
    RecoveryResponse,
    ResumeRunResponse,
    RunResponse,
    RunStatus,
    RunSummaryResponse,
    TraceResponse,
)

from app.checkpointing.models import (
    RecoveryStatus,
)

from app.core.engine import (
    NexusEngine,
)
from app.core.models import (
    TaskStatus,
)
from app.core.state import (
    NexusState,
)


class ControlPlaneError(Exception):
    """
    Raised when the NEXUS API control
    plane cannot complete an operation.
    """


class RunNotFoundError(
    ControlPlaneError
):
    """
    Raised when a requested workflow
    run cannot be resolved.
    """


class NexusControlPlane:
    """
    Application-service layer between
    the HTTP API and NexusEngine.

    The control plane exposes stable
    operations without leaking engine
    implementation details to routes.
    """

    def __init__(
        self,
        engine: NexusEngine,
    ):
        self.engine = engine

        self._runs: dict[
            str,
            NexusState,
        ] = {}

    def register_state(
        self,
        state: NexusState,
    ) -> NexusState:
        self._runs[
            state.run_id
        ] = state

        return state

    def get_state(
        self,
        run_id: str,
    ) -> NexusState:
        state = self._runs.get(
            run_id
        )

        if state is not None:
            return state

        if (
            self.engine.checkpoint_service
            is not None
        ):
            restored = (
                self.engine
                .checkpoint_service
                .restore_state(
                    run_id
                )
            )

            if restored is not None:
                self._runs[
                    run_id
                ] = restored

                return restored

        raise RunNotFoundError(
            "NEXUS run not found: "
            f"{run_id}"
        )

    def _status_for_state(
        self,
        state: NexusState,
    ) -> RunStatus:
        if state.completed:
            return (
                RunStatus.COMPLETED
            )

        if state.failed:
            if (
                self.engine
                .checkpoint_service
                is not None
            ):
                recovery = (
                    self.engine
                    .checkpoint_service
                    .recovery_info(
                        state.run_id
                    )
                )

                if (
                    recovery.status
                    == RecoveryStatus
                    .RECOVERABLE
                ):
                    return (
                        RunStatus.RECOVERABLE
                    )

            return RunStatus.FAILED

        if state.iteration > 0:
            return RunStatus.RUNNING

        return RunStatus.CREATED

    def run_response(
        self,
        state: NexusState,
    ) -> RunResponse:
        return RunResponse(
            run_id=state.run_id,
            status=(
                self._status_for_state(
                    state
                )
            ),
            user_request=(
                state.user_request
            ),
            completed=state.completed,
            failed=state.failed,
            iteration=state.iteration,
            task_count=len(
                state.tasks
            ),
            artifact_count=len(
                state.artifacts
            ),
            metadata=dict(
                state.metadata
            ),
        )

    def run_summary(
        self,
        run_id: str,
    ) -> RunSummaryResponse:
        state = self.get_state(
            run_id
        )

        completed_tasks = sum(
            1
            for task
            in state.tasks.values()
            if (
                task.status
                == TaskStatus.COMPLETED
            )
        )

        failed_tasks = sum(
            1
            for task
            in state.tasks.values()
            if (
                task.status
                == TaskStatus.FAILED
            )
        )

        return RunSummaryResponse(
            run_id=state.run_id,
            status=(
                self._status_for_state(
                    state
                )
            ),
            completed=state.completed,
            failed=state.failed,
            iteration=state.iteration,
            task_count=len(
                state.tasks
            ),
            completed_task_count=(
                completed_tasks
            ),
            failed_task_count=(
                failed_tasks
            ),
            artifact_count=len(
                state.artifacts
            ),
        )

    def recovery(
        self,
        run_id: str,
    ) -> RecoveryResponse:
        if (
            self.engine.checkpoint_service
            is None
        ):
            raise ControlPlaneError(
                "Checkpointing is not "
                "configured."
            )

        info = (
            self.engine
            .checkpoint_service
            .recovery_info(
                run_id
            )
        )

        return RecoveryResponse(
            run_id=run_id,
            status=info.status.value,
            recoverable=(
                info.status
                == RecoveryStatus.RECOVERABLE
            ),
            latest_checkpoint_id=(
                info.latest_checkpoint_id
            ),
            checkpoint_sequence=(
                info.sequence
            ),
            checkpoint_type=(
                info.checkpoint_type.value
                if info.checkpoint_type
                is not None
                else None
            ),
            reason=info.reason,
        )

    def restore_run(
        self,
        run_id: str,
        *,
        allow_failed: bool = False,
    ) -> NexusState:
        state = self.engine.restore_run(
            run_id,
            allow_failed=allow_failed,
        )

        self.register_state(
            state
        )

        return state

    def resume_run(
        self,
        run_id: str,
        *,
        allow_failed: bool = False,
    ) -> ResumeRunResponse:
        state = self.engine.resume_run(
            run_id,
            allow_failed=allow_failed,
        )

        self.register_state(
            state
        )

        return ResumeRunResponse(
            run_id=state.run_id,
            status=(
                self._status_for_state(
                    state
                )
            ),
            resumed=True,
            recovered_from_checkpoint=(
                state.metadata.get(
                    "recovered_from_checkpoint"
                )
            ),
        )

    def trace(
        self,
        run_id: str,
    ) -> TraceResponse:
        service = (
            self.engine
            .observability_service
        )

        if service is None:
            raise ControlPlaneError(
                "Observability is not "
                "configured."
            )

        summary = service.get_summary(
            run_id
        )

        if summary is None:
            raise RunNotFoundError(
                "Trace not found for run: "
                f"{run_id}"
            )

        return TraceResponse(
            run_id=summary.run_id,
            status=summary.status.value,
            total_events=(
                summary.total_events
            ),
            total_duration_ms=(
                summary.total_duration_ms
            ),
            task_count=(
                summary.task_count
            ),
            completed_task_count=(
                summary.completed_task_count
            ),
            failed_task_count=(
                summary.failed_task_count
            ),
            repair_count=(
                summary.repair_count
            ),
            replan_count=(
                summary.replan_count
            ),
            artifact_count=(
                summary.artifact_count
            ),
            agents_used=(
                summary.agents_used
            ),
        )

    def evaluation(
        self,
        run_id: str,
    ) -> EvaluationResponse:
        service = (
            self.engine
            .evaluation_service
        )

        if service is None:
            raise ControlPlaneError(
                "Evaluation is not "
                "configured."
            )

        evaluation = (
            service.get_evaluation(
                run_id
            )
        )

        if evaluation is None:
            raise RunNotFoundError(
                "Evaluation not found "
                f"for run: {run_id}"
            )

        baseline = service.get_baseline()

        regression_detected = None

        state = self._runs.get(
            run_id
        )

        if state is not None:
            benchmark = (
                state.metadata.get(
                    "evaluation_benchmark"
                )
            )

            if benchmark is not None:
                regression_detected = (
                    benchmark.get(
                        "regression_detected"
                    )
                )

        return EvaluationResponse(
            run_id=evaluation.run_id,
            overall_score=(
                evaluation.overall_score
            ),
            status=(
                evaluation.status.value
            ),
            regression_detected=(
                regression_detected
            ),
            baseline_run_id=(
                baseline.run_id
                if baseline is not None
                else None
            ),
            payload=(
                evaluation.model_dump(
                    mode="json"
                )
            ),
        )

    def pending_approvals(
        self,
    ) -> list[
        ApprovalResponse
    ]:
        manager = (
            self.engine
            .approval_manager
        )

        if manager is None:
            raise ControlPlaneError(
                "Approvals are not "
                "configured."
            )

        responses = []

        for request in (
            manager.pending_requests()
        ):
            responses.append(
                ApprovalResponse(
                    request_id=request.id,
                    run_id=request.run_id,
                    status=(
                        request.status.value
                    ),
                    risk=(
                        request.risk.value
                    ),
                    action_type=(
                        request
                        .action_type
                        .value
                    ),
                    title=request.title,
                    requested_by=(
                        request.requested_by
                    ),
                    proposed_action=(
                        request
                        .proposed_action
                    ),
                    allowed=False,
                )
            )

        return responses

    def approve(
        self,
        request_id: str,
        *,
        reason: str,
        decided_by: str,
        metadata: Optional[
            dict
        ] = None,
    ) -> ApprovalResponse:
        manager = (
            self.engine
            .approval_manager
        )

        if manager is None:
            raise ControlPlaneError(
                "Approvals are not "
                "configured."
            )

        result = manager.approve(
            request_id,
            reason=reason,
            decided_by=decided_by,
            metadata=metadata,
        )

        request = result.request

        return ApprovalResponse(
            request_id=request.id,
            run_id=request.run_id,
            status=(
                request.status.value
            ),
            risk=request.risk.value,
            action_type=(
                request.action_type.value
            ),
            title=request.title,
            requested_by=(
                request.requested_by
            ),
            proposed_action=(
                request.proposed_action
            ),
            allowed=result.allowed,
        )

    def reject(
        self,
        request_id: str,
        *,
        reason: str,
        decided_by: str,
        metadata: Optional[
            dict
        ] = None,
    ) -> ApprovalResponse:
        manager = (
            self.engine
            .approval_manager
        )

        if manager is None:
            raise ControlPlaneError(
                "Approvals are not "
                "configured."
            )

        result = manager.reject(
            request_id,
            reason=reason,
            decided_by=decided_by,
            metadata=metadata,
        )

        request = result.request

        return ApprovalResponse(
            request_id=request.id,
            run_id=request.run_id,
            status=(
                request.status.value
            ),
            risk=request.risk.value,
            action_type=(
                request.action_type.value
            ),
            title=request.title,
            requested_by=(
                request.requested_by
            ),
            proposed_action=(
                request.proposed_action
            ),
            allowed=result.allowed,
        )
