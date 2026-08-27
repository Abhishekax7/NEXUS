from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import (
    BaseModel,
    Field,
)


def utc_now() -> datetime:
    return datetime.now(
        timezone.utc
    )


class CheckpointType(
    str,
    Enum,
):
    WORKFLOW_STARTED = (
        "workflow_started"
    )

    ITERATION_COMPLETED = (
        "iteration_completed"
    )

    TASK_COMPLETED = (
        "task_completed"
    )

    REPAIR_COMPLETED = (
        "repair_completed"
    )

    REPLAN_COMPLETED = (
        "replan_completed"
    )

    APPROVAL_PENDING = (
        "approval_pending"
    )

    WORKFLOW_COMPLETED = (
        "workflow_completed"
    )

    WORKFLOW_FAILED = (
        "workflow_failed"
    )

    MANUAL = "manual"


class CheckpointStatus(
    str,
    Enum,
):
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    SUPERSEDED = "superseded"


class WorkflowCheckpoint(BaseModel):
    id: str = Field(
        default_factory=lambda: str(
            uuid4()
        )
    )

    run_id: str = Field(
        min_length=1
    )

    checkpoint_type: CheckpointType

    status: CheckpointStatus = (
        CheckpointStatus.ACTIVE
    )

    sequence: int = Field(
        ge=0
    )

    created_at: datetime = Field(
        default_factory=utc_now
    )

    state_payload: dict[
        str,
        Any,
    ]

    reason: str = Field(
        min_length=1
    )

    task_id: Optional[
        str
    ] = None

    metadata: dict[
        str,
        Any,
    ] = Field(
        default_factory=dict
    )


class RecoveryStatus(
    str,
    Enum,
):
    RECOVERABLE = "recoverable"
    COMPLETED = "completed"
    FAILED = "failed"
    NOT_FOUND = "not_found"


class RecoveryInfo(BaseModel):
    run_id: str = Field(
        min_length=1
    )

    status: RecoveryStatus

    latest_checkpoint_id: Optional[
        str
    ] = None

    sequence: Optional[
        int
    ] = None

    checkpoint_type: Optional[
        CheckpointType
    ] = None

    reason: str = Field(
        min_length=1
    )


class RecoveryResult(BaseModel):
    recovery: RecoveryInfo

    checkpoint: Optional[
        WorkflowCheckpoint
    ] = None


class CheckpointError(Exception):
    """
    Raised when NEXUS checkpointing or
    workflow recovery cannot be completed
    safely.
    """
