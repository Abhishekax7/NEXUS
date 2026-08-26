from enum import Enum
from typing import Optional

from pydantic import (
    BaseModel,
    Field,
)

from app.core.models import AgentRole


class ReplanAction(str, Enum):
    KEEP_PLAN = "keep_plan"
    ADD_TASK = "add_task"
    REMOVE_TASK = "remove_task"
    REPLACE_TASK = "replace_task"


class ProposedTask(BaseModel):
    title: str = Field(
        min_length=1
    )

    description: str = Field(
        min_length=1
    )

    assigned_agent: AgentRole

    depends_on_roles: list[
        AgentRole
    ] = []

    metadata: dict = {}


class ReplanningDecision(BaseModel):
    should_replan: bool

    reason: str = Field(
        min_length=1
    )

    action: ReplanAction

    proposed_task: Optional[
        ProposedTask
    ] = None

    target_task_id: Optional[
        str
    ] = None

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    evidence: list[str] = Field(
        min_length=1
    )


class ReplanningError(Exception):
    """Raised when a replanning decision is invalid."""


def validate_replanning_decision(
    decision: ReplanningDecision,
) -> None:
    if not decision.should_replan:
        if (
            decision.action
            != ReplanAction.KEEP_PLAN
        ):
            raise ReplanningError(
                "Non-replanning decision must "
                "use KEEP_PLAN."
            )

        return

    if (
        decision.action
        == ReplanAction.KEEP_PLAN
    ):
        raise ReplanningError(
            "Replanning decision cannot "
            "use KEEP_PLAN."
        )

    if (
        decision.action
        in {
            ReplanAction.ADD_TASK,
            ReplanAction.REPLACE_TASK,
        }
        and decision.proposed_task is None
    ):
        raise ReplanningError(
            "ADD_TASK and REPLACE_TASK "
            "require proposed_task."
        )

    if (
        decision.action
        in {
            ReplanAction.REMOVE_TASK,
            ReplanAction.REPLACE_TASK,
        }
        and not decision.target_task_id
    ):
        raise ReplanningError(
            "REMOVE_TASK and REPLACE_TASK "
            "require target_task_id."
        )
