from typing import Optional

from app.checkpointing.models import (
    CheckpointStatus,
    CheckpointType,
    RecoveryInfo,
    RecoveryResult,
    RecoveryStatus,
    WorkflowCheckpoint,
)
from app.checkpointing.store import (
    CheckpointStore,
)
from app.core.state import NexusState


class CheckpointService:
    """
    Production checkpointing and recovery
    interface for NEXUS workflows.

    Responsibilities:
    - serialize NexusState
    - create ordered checkpoints
    - classify recoverability
    - reconstruct saved workflow state
    """

    def __init__(
        self,
        store: CheckpointStore,
    ):
        self.store = store

    def _serialize_state(
        self,
        state: NexusState,
    ) -> dict:
        return state.model_dump(
            mode="json"
        )

    def _restore_state(
        self,
        payload: dict,
    ) -> NexusState:
        return NexusState.model_validate(
            payload
        )

    def create_checkpoint(
        self,
        *,
        state: NexusState,
        checkpoint_type: CheckpointType,
        reason: str,
        task_id: Optional[
            str
        ] = None,
        status: CheckpointStatus = (
            CheckpointStatus.ACTIVE
        ),
        metadata: Optional[
            dict
        ] = None,
    ) -> WorkflowCheckpoint:
        sequence = (
            self.store.next_sequence(
                state.run_id
            )
        )

        checkpoint = WorkflowCheckpoint(
            run_id=state.run_id,
            checkpoint_type=(
                checkpoint_type
            ),
            status=status,
            sequence=sequence,
            state_payload=(
                self._serialize_state(
                    state
                )
            ),
            reason=reason,
            task_id=task_id,
            metadata=(
                metadata
                or {}
            ),
        )

        self.store.save(
            checkpoint
        )

        return checkpoint

    def workflow_started(
        self,
        state: NexusState,
    ) -> WorkflowCheckpoint:
        return self.create_checkpoint(
            state=state,
            checkpoint_type=(
                CheckpointType
                .WORKFLOW_STARTED
            ),
            reason=(
                "Workflow execution started."
            ),
        )

    def task_completed(
        self,
        state: NexusState,
        task_id: str,
    ) -> WorkflowCheckpoint:
        return self.create_checkpoint(
            state=state,
            checkpoint_type=(
                CheckpointType
                .TASK_COMPLETED
            ),
            reason=(
                "Task completed successfully."
            ),
            task_id=task_id,
        )

    def iteration_completed(
        self,
        state: NexusState,
    ) -> WorkflowCheckpoint:
        return self.create_checkpoint(
            state=state,
            checkpoint_type=(
                CheckpointType
                .ITERATION_COMPLETED
            ),
            reason=(
                "Workflow iteration completed."
            ),
            metadata={
                "iteration":
                    state.iteration,
            },
        )

    def repair_completed(
        self,
        state: NexusState,
        *,
        attempts: int,
        passed: bool,
    ) -> WorkflowCheckpoint:
        return self.create_checkpoint(
            state=state,
            checkpoint_type=(
                CheckpointType
                .REPAIR_COMPLETED
            ),
            reason=(
                "Autonomous repair cycle "
                "completed."
            ),
            metadata={
                "attempts": attempts,
                "passed": passed,
            },
        )

    def replan_completed(
        self,
        state: NexusState,
        *,
        action: str,
    ) -> WorkflowCheckpoint:
        return self.create_checkpoint(
            state=state,
            checkpoint_type=(
                CheckpointType
                .REPLAN_COMPLETED
            ),
            reason=(
                "Dynamic workflow replanning "
                "completed."
            ),
            metadata={
                "action": action,
            },
        )

    def approval_pending(
        self,
        state: NexusState,
        *,
        approval_request_id: str,
    ) -> WorkflowCheckpoint:
        return self.create_checkpoint(
            state=state,
            checkpoint_type=(
                CheckpointType
                .APPROVAL_PENDING
            ),
            reason=(
                "Workflow paused pending "
                "human approval."
            ),
            metadata={
                "approval_request_id":
                    approval_request_id,
            },
        )

    def workflow_completed(
        self,
        state: NexusState,
    ) -> WorkflowCheckpoint:
        return self.create_checkpoint(
            state=state,
            checkpoint_type=(
                CheckpointType
                .WORKFLOW_COMPLETED
            ),
            status=(
                CheckpointStatus.COMPLETED
            ),
            reason=(
                "Workflow completed "
                "successfully."
            ),
        )

    def workflow_failed(
        self,
        state: NexusState,
        *,
        reason: str,
    ) -> WorkflowCheckpoint:
        return self.create_checkpoint(
            state=state,
            checkpoint_type=(
                CheckpointType
                .WORKFLOW_FAILED
            ),
            status=(
                CheckpointStatus.FAILED
            ),
            reason=reason,
        )

    def latest_checkpoint(
        self,
        run_id: str,
    ) -> Optional[
        WorkflowCheckpoint
    ]:
        return self.store.latest(
            run_id
        )

    def recovery_info(
        self,
        run_id: str,
    ) -> RecoveryInfo:
        checkpoint = (
            self.latest_checkpoint(
                run_id
            )
        )

        if checkpoint is None:
            return RecoveryInfo(
                run_id=run_id,
                status=(
                    RecoveryStatus.NOT_FOUND
                ),
                reason=(
                    "No checkpoint exists "
                    "for this run."
                ),
            )

        if (
            checkpoint.checkpoint_type
            == CheckpointType.WORKFLOW_COMPLETED
            or checkpoint.status
            == CheckpointStatus.COMPLETED
        ):
            return RecoveryInfo(
                run_id=run_id,
                status=(
                    RecoveryStatus.COMPLETED
                ),
                latest_checkpoint_id=(
                    checkpoint.id
                ),
                sequence=(
                    checkpoint.sequence
                ),
                checkpoint_type=(
                    checkpoint
                    .checkpoint_type
                ),
                reason=(
                    "Workflow has already "
                    "completed."
                ),
            )

        if (
            checkpoint.checkpoint_type
            == CheckpointType.WORKFLOW_FAILED
            or checkpoint.status
            == CheckpointStatus.FAILED
        ):
            return RecoveryInfo(
                run_id=run_id,
                status=(
                    RecoveryStatus.FAILED
                ),
                latest_checkpoint_id=(
                    checkpoint.id
                ),
                sequence=(
                    checkpoint.sequence
                ),
                checkpoint_type=(
                    checkpoint
                    .checkpoint_type
                ),
                reason=(
                    "Latest checkpoint records "
                    "a failed workflow."
                ),
            )

        return RecoveryInfo(
            run_id=run_id,
            status=(
                RecoveryStatus.RECOVERABLE
            ),
            latest_checkpoint_id=(
                checkpoint.id
            ),
            sequence=(
                checkpoint.sequence
            ),
            checkpoint_type=(
                checkpoint
                .checkpoint_type
            ),
            reason=(
                "Workflow has an active "
                "recoverable checkpoint."
            ),
        )

    def recover(
        self,
        run_id: str,
    ) -> RecoveryResult:
        info = self.recovery_info(
            run_id
        )

        if (
            info.status
            == RecoveryStatus.NOT_FOUND
        ):
            return RecoveryResult(
                recovery=info,
                checkpoint=None,
            )

        checkpoint = (
            self.latest_checkpoint(
                run_id
            )
        )

        return RecoveryResult(
            recovery=info,
            checkpoint=checkpoint,
        )

    def restore_state(
        self,
        run_id: str,
    ) -> Optional[
        NexusState
    ]:
        result = self.recover(
            run_id
        )

        if result.checkpoint is None:
            return None

        return self._restore_state(
            result.checkpoint
            .state_payload
        )
