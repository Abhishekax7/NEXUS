from dataclasses import dataclass

from app.approval.models import (
    ApprovalActionType,
    ApprovalRequest,
    ApprovalRisk,
)


@dataclass
class ApprovalPolicyDecision:
    requires_approval: bool
    auto_allowed: bool
    reason: str


class ApprovalPolicy:
    """
    Deterministic policy deciding whether
    an action requires explicit approval.
    """

    def __init__(
        self,
        require_medium_risk_approval: bool = False,
    ):
        self.require_medium_risk_approval = (
            require_medium_risk_approval
        )

    def evaluate(
        self,
        request: ApprovalRequest,
    ) -> ApprovalPolicyDecision:
        if request.risk == ApprovalRisk.LOW:
            return ApprovalPolicyDecision(
                requires_approval=False,
                auto_allowed=True,
                reason=(
                    "Low-risk action may be "
                    "executed automatically."
                ),
            )

        if request.risk == ApprovalRisk.MEDIUM:
            if self.require_medium_risk_approval:
                return ApprovalPolicyDecision(
                    requires_approval=True,
                    auto_allowed=False,
                    reason=(
                        "Medium-risk actions require "
                        "approval under current policy."
                    ),
                )

            return ApprovalPolicyDecision(
                requires_approval=False,
                auto_allowed=True,
                reason=(
                    "Medium-risk action is allowed "
                    "automatically under current policy."
                ),
            )

        if request.risk == ApprovalRisk.HIGH:
            return ApprovalPolicyDecision(
                requires_approval=True,
                auto_allowed=False,
                reason=(
                    "High-risk action requires "
                    "explicit approval."
                ),
            )

        if request.risk == ApprovalRisk.CRITICAL:
            return ApprovalPolicyDecision(
                requires_approval=True,
                auto_allowed=False,
                reason=(
                    "Critical-risk action requires "
                    "explicit approval."
                ),
            )

        return ApprovalPolicyDecision(
            requires_approval=True,
            auto_allowed=False,
            reason=(
                "Unknown risk level requires "
                "explicit approval."
            ),
        )

    def allows_automatic_execution(
        self,
        request: ApprovalRequest,
    ) -> bool:
        return (
            self.evaluate(
                request
            ).auto_allowed
        )

    def requires_approval(
        self,
        request: ApprovalRequest,
    ) -> bool:
        return (
            self.evaluate(
                request
            ).requires_approval
        )
