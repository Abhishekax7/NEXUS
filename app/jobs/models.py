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


class JobStatus(
    str,
    Enum,
):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobPriority(
    int,
    Enum,
):
    LOW = 30
    NORMAL = 20
    HIGH = 10
    CRITICAL = 0


class WorkflowJob(BaseModel):
    id: str = Field(
        default_factory=lambda: str(
            uuid4()
        )
    )

    run_id: str = Field(
        min_length=1
    )

    status: JobStatus = (
        JobStatus.QUEUED
    )

    priority: JobPriority = (
        JobPriority.NORMAL
    )

    created_at: datetime = Field(
        default_factory=utc_now
    )

    started_at: Optional[
        datetime
    ] = None

    completed_at: Optional[
        datetime
    ] = None

    attempt: int = Field(
        default=0,
        ge=0,
    )

    max_attempts: int = Field(
        default=1,
        ge=1,
    )

    metadata: dict[
        str,
        Any,
    ] = Field(
        default_factory=dict
    )


class JobExecutionResult(
    BaseModel
):
    job_id: str = Field(
        min_length=1
    )

    run_id: str = Field(
        min_length=1
    )

    status: JobStatus

    success: bool

    started_at: Optional[
        datetime
    ] = None

    completed_at: Optional[
        datetime
    ] = None

    error: Optional[
        str
    ] = None

    metadata: dict[
        str,
        Any,
    ] = Field(
        default_factory=dict
    )


class JobSnapshot(
    BaseModel
):
    job_id: str

    run_id: str

    status: JobStatus

    priority: JobPriority

    attempt: int

    max_attempts: int

    queued: bool

    running: bool

    terminal: bool


class JobError(Exception):
    """
    Base exception for the NEXUS
    asynchronous job subsystem.
    """


class JobNotFoundError(
    JobError
):
    """
    Raised when a requested job
    cannot be found.
    """


class JobStateError(
    JobError
):
    """
    Raised when an invalid job
    lifecycle transition is attempted.
    """
