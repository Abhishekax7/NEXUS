from enum import Enum
from typing import Any, Optional

from pydantic import (
    BaseModel,
    Field,
)


class RunStatus(
    str,
    Enum,
):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RECOVERABLE = "recoverable"


class CreateRunRequest(BaseModel):
    user_request: str = Field(
        min_length=1
    )

    metadata: dict[
        str,
        Any,
    ] = Field(
        default_factory=dict
    )


class RunResponse(BaseModel):
    run_id: str = Field(
        min_length=1
    )

    status: RunStatus

    user_request: str

    completed: bool

    failed: bool

    iteration: int = Field(
        ge=0
    )

    task_count: int = Field(
        ge=0
    )

    artifact_count: int = Field(
        ge=0
    )

    metadata: dict[
        str,
        Any,
    ] = Field(
        default_factory=dict
    )


class RunSummaryResponse(BaseModel):
    run_id: str

    status: RunStatus

    completed: bool

    failed: bool

    iteration: int = Field(
        ge=0
    )

    task_count: int = Field(
        ge=0
    )

    completed_task_count: int = Field(
        ge=0
    )

    failed_task_count: int = Field(
        ge=0
    )

    artifact_count: int = Field(
        ge=0
    )


class ResumeRunRequest(BaseModel):
    allow_failed: bool = False


class ResumeRunResponse(BaseModel):
    run_id: str

    status: RunStatus

    resumed: bool

    recovered_from_checkpoint: Optional[
        dict[str, Any]
    ] = None


class ApprovalDecisionRequest(BaseModel):
    reason: str = Field(
        min_length=1
    )

    decided_by: str = Field(
        min_length=1
    )

    metadata: dict[
        str,
        Any,
    ] = Field(
        default_factory=dict
    )


class ApprovalResponse(BaseModel):
    request_id: str

    run_id: str

    status: str

    risk: str

    action_type: str

    title: str

    requested_by: Optional[
        str
    ] = None

    proposed_action: dict[
        str,
        Any,
    ]

    allowed: bool


class EvaluationResponse(BaseModel):
    run_id: str

    overall_score: float = Field(
        ge=0.0,
        le=100.0,
    )

    status: str

    regression_detected: Optional[
        bool
    ] = None

    baseline_run_id: Optional[
        str
    ] = None

    payload: dict[
        str,
        Any,
    ]


class TraceResponse(BaseModel):
    run_id: str

    status: str

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

    agents_used: list[
        str
    ] = Field(
        default_factory=list
    )


class RecoveryResponse(BaseModel):
    run_id: str

    status: str

    recoverable: bool

    latest_checkpoint_id: Optional[
        str
    ] = None

    checkpoint_sequence: Optional[
        int
    ] = None

    checkpoint_type: Optional[
        str
    ] = None

    reason: str


class HealthResponse(BaseModel):
    status: str = "ok"

    service: str = "nexus"

    version: str = "1.0"

