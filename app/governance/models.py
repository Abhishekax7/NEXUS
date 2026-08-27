from enum import Enum
from typing import Any, Optional

from pydantic import (
    BaseModel,
    Field,
)


class PolicyEffect(
    str,
    Enum,
):
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = (
        "require_approval"
    )


class ResourceType(
    str,
    Enum,
):
    LLM_CALLS = "llm_calls"
    TOOL_CALLS = "tool_calls"
    TASKS = "tasks"
    REPLANS = "replans"
    REPAIRS = "repairs"
    WALL_TIME_SECONDS = (
        "wall_time_seconds"
    )


class ResourceBudget(
    BaseModel
):
    max_llm_calls: Optional[
        int
    ] = Field(
        default=None,
        ge=0,
    )

    max_tool_calls: Optional[
        int
    ] = Field(
        default=None,
        ge=0,
    )

    max_tasks: Optional[
        int
    ] = Field(
        default=None,
        ge=0,
    )

    max_replans: Optional[
        int
    ] = Field(
        default=None,
        ge=0,
    )

    max_repairs: Optional[
        int
    ] = Field(
        default=None,
        ge=0,
    )

    max_wall_time_seconds: Optional[
        float
    ] = Field(
        default=None,
        ge=0.0,
    )


class ResourceUsage(
    BaseModel
):
    llm_calls: int = Field(
        default=0,
        ge=0,
    )

    tool_calls: int = Field(
        default=0,
        ge=0,
    )

    tasks: int = Field(
        default=0,
        ge=0,
    )

    replans: int = Field(
        default=0,
        ge=0,
    )

    repairs: int = Field(
        default=0,
        ge=0,
    )

    wall_time_seconds: float = Field(
        default=0.0,
        ge=0.0,
    )


class PolicyRule(
    BaseModel
):
    id: str = Field(
        min_length=1
    )

    action: str = Field(
        min_length=1
    )

    effect: PolicyEffect

    reason: str = Field(
        min_length=1
    )

    metadata: dict[
        str,
        Any,
    ] = Field(
        default_factory=dict
    )


class PolicyDecision(
    BaseModel
):
    action: str = Field(
        min_length=1
    )

    effect: PolicyEffect

    allowed: bool

    requires_approval: bool = False

    matched_rule_id: Optional[
        str
    ] = None

    reason: str = Field(
        min_length=1
    )

    metadata: dict[
        str,
        Any,
    ] = Field(
        default_factory=dict
    )


class BudgetViolation(
    BaseModel
):
    resource: ResourceType

    limit: float

    observed: float

    message: str = Field(
        min_length=1
    )


class GovernanceError(Exception):
    """
    Base error for NEXUS execution
    governance and resource controls.
    """


class PolicyDeniedError(
    GovernanceError
):
    """
    Raised when an action is explicitly
    denied by execution policy.
    """


class ResourceLimitExceeded(
    GovernanceError
):
    """
    Raised when a workflow exceeds an
    allowed resource budget.
    """
