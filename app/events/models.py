from datetime import (
    datetime,
    timezone,
)
from enum import Enum
from typing import (
    Any,
    Optional,
)
from uuid import uuid4

from pydantic import (
    BaseModel,
    Field,
)


def utc_now() -> datetime:
    return datetime.now(
        timezone.utc
    )


class EventType(
    str,
    Enum,
):
    RUN_CREATED = "run.created"

    RUN_STARTED = "run.started"

    RUN_COMPLETED = "run.completed"

    RUN_FAILED = "run.failed"

    JOB_QUEUED = "job.queued"

    JOB_STARTED = "job.started"

    JOB_COMPLETED = "job.completed"

    JOB_FAILED = "job.failed"

    JOB_CANCELLED = "job.cancelled"

    JOB_RETRIED = "job.retried"

    TASK_STARTED = "task.started"

    TASK_COMPLETED = "task.completed"

    TASK_FAILED = "task.failed"

    TOOL_SELECTED = "tool.selected"

    TOOL_STARTED = "tool.started"

    TOOL_COMPLETED = "tool.completed"

    TOOL_FAILED = "tool.failed"

    APPROVAL_REQUIRED = (
        "approval.required"
    )

    APPROVAL_APPROVED = (
        "approval.approved"
    )

    APPROVAL_REJECTED = (
        "approval.rejected"
    )

    REPLAN_STARTED = (
        "replan.started"
    )

    REPLAN_COMPLETED = (
        "replan.completed"
    )

    REPAIR_STARTED = (
        "repair.started"
    )

    REPAIR_COMPLETED = (
        "repair.completed"
    )

    GOVERNANCE_ALLOWED = (
        "governance.allowed"
    )

    GOVERNANCE_BLOCKED = (
        "governance.blocked"
    )


class EventSeverity(
    str,
    Enum,
):
    DEBUG = "debug"

    INFO = "info"

    WARNING = "warning"

    ERROR = "error"

    CRITICAL = "critical"


class NexusEvent(
    BaseModel
):
    id: str = Field(
        default_factory=lambda: str(
            uuid4()
        )
    )

    type: EventType

    severity: EventSeverity = (
        EventSeverity.INFO
    )

    timestamp: datetime = Field(
        default_factory=utc_now
    )

    run_id: Optional[
        str
    ] = None

    job_id: Optional[
        str
    ] = None

    task_id: Optional[
        str
    ] = None

    agent_role: Optional[
        str
    ] = None

    source: str = Field(
        min_length=1
    )

    message: str = Field(
        min_length=1
    )

    payload: dict[
        str,
        Any,
    ] = Field(
        default_factory=dict
    )


class EventFilter(
    BaseModel
):
    run_id: Optional[
        str
    ] = None

    job_id: Optional[
        str
    ] = None

    event_types: list[
        EventType
    ] = Field(
        default_factory=list
    )

    severities: list[
        EventSeverity
    ] = Field(
        default_factory=list
    )


class EventPage(
    BaseModel
):
    events: list[
        NexusEvent
    ]

    count: int = Field(
        ge=0
    )

    total: int = Field(
        ge=0
    )


class EventSubscription(
    BaseModel
):
    id: str = Field(
        default_factory=lambda: str(
            uuid4()
        )
    )

    run_id: Optional[
        str
    ] = None

    job_id: Optional[
        str
    ] = None

    created_at: datetime = Field(
        default_factory=utc_now
    )

    active: bool = True


class EventError(Exception):
    """
    Base error for the NEXUS
    real-time event subsystem.
    """


class SubscriptionNotFoundError(
    EventError
):
    """
    Raised when an event subscription
    cannot be resolved.
    """
