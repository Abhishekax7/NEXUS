from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import (
    BaseModel,
    Field,
)


class ApprovalStatus(
    str,
    Enum,
):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


class ApprovalRisk(
    str,
    Enum,
):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ApprovalActionType(
    str,
    Enum,
):
    TOOL_EXECUTION = "tool_execution"
    CODE_CHANGE = "code_change"
    PLAN_MUTATION = "plan_mutation"
    SECURITY_OVERRIDE = "security_override"
    FILE_WRITE = "file_write"
    OTHER = "other"


class ApprovalRequest(BaseModel):
    id: str = Field(
        default_factory=lambda: str(
            uuid4()
        )
    )

    run_id: str = Field(
        min_length=1
    )

    action_type: ApprovalActionType

    risk: ApprovalRisk

    title: str = Field(
        min_length=1
    )

    description: str = Field(
        min_length=1
    )

    proposed_action: dict[
        str,
        Any,
    ]

    reason: str = Field(
        min_length=1
    )

    status: ApprovalStatus = (
        ApprovalStatus.PENDING
    )

    requested_by: Optional[
        str
    ] = None

    metadata: dict[
        str,
        Any,
    ] = Field(
        default_factory=dict
    )


class ApprovalDecision(BaseModel):
    request_id: str = Field(
        min_length=1
    )

    approved: bool

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


class ApprovalResult(BaseModel):
    request: ApprovalRequest

    decision: Optional[
        ApprovalDecision
    ] = None

    allowed: bool


class ApprovalError(Exception):
    """
    Raised when an approval workflow
    cannot be processed safely.
    """
