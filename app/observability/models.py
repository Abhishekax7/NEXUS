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


class TraceEventType(
    str,
    Enum,
):
    WORKFLOW_STARTED = (
        "workflow_started"
    )

    WORKFLOW_COMPLETED = (
        "workflow_completed"
    )

    WORKFLOW_FAILED = (
        "workflow_failed"
    )

    TASK_STARTED = (
        "task_started"
    )

    TASK_COMPLETED = (
        "task_completed"
    )

    TASK_FAILED = (
        "task_failed"
    )

    ARTIFACT_CREATED = (
        "artifact_created"
    )

    REPAIR_STARTED = (
        "repair_started"
    )

    REPAIR_COMPLETED = (
        "repair_completed"
    )

    REPAIR_FAILED = (
        "repair_failed"
    )

    REPLAN_STARTED = (
        "replan_started"
    )

    REPLAN_COMPLETED = (
        "replan_completed"
    )

    EVALUATION_COMPLETED = (
        "evaluation_completed"
    )


class TraceStatus(
    str,
    Enum,
):
    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"
    INFO = "info"


class TraceEvent(BaseModel):
    id: str = Field(
        default_factory=lambda: str(
            uuid4()
        )
    )

    run_id: str = Field(
        min_length=1
    )

    event_type: TraceEventType

    status: TraceStatus

    timestamp: datetime = Field(
        default_factory=utc_now
    )

    task_id: Optional[str] = None

    agent_role: Optional[str] = None

    artifact_id: Optional[str] = None

    duration_ms: Optional[float] = Field(
        default=None,
        ge=0.0,
    )

    message: Optional[str] = None

    metadata: dict[
        str,
        Any,
    ] = Field(
        default_factory=dict
    )


class WorkflowTrace(BaseModel):
    run_id: str = Field(
        min_length=1
    )

    started_at: datetime = Field(
        default_factory=utc_now
    )

    completed_at: Optional[
        datetime
    ] = None

    status: TraceStatus = (
        TraceStatus.STARTED
    )

    events: list[
        TraceEvent
    ] = Field(
        default_factory=list
    )

    total_duration_ms: Optional[
        float
    ] = Field(
        default=None,
        ge=0.0,
    )

    task_count: int = Field(
        default=0,
        ge=0,
    )

    completed_task_count: int = Field(
        default=0,
        ge=0,
    )

    failed_task_count: int = Field(
        default=0,
        ge=0,
    )

    repair_count: int = Field(
        default=0,
        ge=0,
    )

    replan_count: int = Field(
        default=0,
        ge=0,
    )

    artifact_count: int = Field(
        default=0,
        ge=0,
    )

    def add_event(
        self,
        event: TraceEvent,
    ) -> None:
        if (
            event.run_id
            != self.run_id
        ):
            raise ValueError(
                "Trace event run_id does "
                "not match workflow trace."
            )

        self.events.append(
            event
        )


class TraceSummary(BaseModel):
    run_id: str

    status: TraceStatus

    total_events: int = Field(
        ge=0
    )

    total_duration_ms: Optional[
        float
    ] = None

    task_count: int = Field(
        ge=0
    )

    completed_task_count: int = Field(
        ge=0
    )

    failed_task_count: int = Field(
        ge=0
    )

    repair_count: int = Field(
        ge=0
    )

    replan_count: int = Field(
        ge=0
    )

    artifact_count: int = Field(
        ge=0
    )

    agents_used: list[str] = Field(
        default_factory=list
    )

