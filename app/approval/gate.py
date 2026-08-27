from dataclasses import dataclass
from typing import Any, Optional

from app.approval.manager import (
    ApprovalManager,
)
from app.approval.models import (
    ApprovalActionType,
    ApprovalRequest,
    ApprovalRisk,
    ApprovalStatus,
)


class ApprovalRequired(Exception):
    """
    Raised when an action cannot proceed
    until explicit approval is provided.
    """

    def __init__(
        self,
        request: ApprovalRequest,
    ):
        self.request = request

        super().__init__(
            "Approval required for action: "
            f"{request.id}"
        )


class ApprovalRejected(Exception):
    """
    Raised when an action was explicitly
    rejected and therefore cannot execute.
    """

    def __init__(
        self,
        request: ApprovalRequest,
    ):
        self.request = request

        super().__init__(
            "Approval rejected for action: "
            f"{request.id}"
        )


@dataclass
class ApprovalGateResult:
    allowed: bool
    request: ApprovalRequest
    automatic: bool


class ApprovalGate:
    """
    Execution gate for actions that may
    require human approval.

    The gate does not execute actions.
    It only determines whether execution
    is currently permitted.
    """

    def __init__(
        self,
        manager: ApprovalManager,
    ):
        self.manager = manager

    def request_execution(
        self,
        *,
        run_id: str,
        action_type: ApprovalActionType,
        risk: ApprovalRisk,
        title: str,
        description: str,
        proposed_action: dict[
            str,
            Any,
        ],
        reason: str,
        requested_by: Optional[
            str
        ] = None,
        metadata: Optional[
            dict[str, Any]
        ] = None,
    ) -> ApprovalGateResult:
        request = ApprovalRequest(
            run_id=run_id,
            action_type=action_type,
            risk=risk,
            title=title,
            description=description,
            proposed_action=(
                proposed_action
            ),
            reason=reason,
            requested_by=requested_by,
            metadata=(
                metadata
                or {}
            ),
        )

        result = (
            self.manager.create_request(
                request
            )
        )

        automatic = bool(
            result.decision
            is not None
            and result.decision.metadata.get(
                "automatic"
            )
            is True
        )

        return ApprovalGateResult(
            allowed=result.allowed,
            request=result.request,
            automatic=automatic,
        )

    def require_execution(
        self,
        *,
        run_id: str,
        action_type: ApprovalActionType,
        risk: ApprovalRisk,
        title: str,
        description: str,
        proposed_action: dict[
            str,
            Any,
        ],
        reason: str,
        requested_by: Optional[
            str
        ] = None,
        metadata: Optional[
            dict[str, Any]
        ] = None,
    ) -> ApprovalGateResult:
        result = self.request_execution(
            run_id=run_id,
            action_type=action_type,
            risk=risk,
            title=title,
            description=description,
            proposed_action=(
                proposed_action
            ),
            reason=reason,
            requested_by=requested_by,
            metadata=metadata,
        )

        if not result.allowed:
            raise ApprovalRequired(
                result.request
            )

        return result

    def resume(
        self,
        request_id: str,
    ) -> ApprovalGateResult:
        request = (
            self.manager.get_request(
                request_id
            )
        )

        if request is None:
            raise ApprovalRequired(
                ApprovalRequest(
                    id=request_id,
                    run_id="unknown",
                    action_type=(
                        ApprovalActionType.OTHER
                    ),
                    risk=(
                        ApprovalRisk.HIGH
                    ),
                    title=(
                        "Unknown approval request"
                    ),
                    description=(
                        "Approval request could "
                        "not be found."
                    ),
                    proposed_action={},
                    reason=(
                        "Execution cannot continue "
                        "without a valid approval."
                    ),
                )
            )

        if (
            request.status
            == ApprovalStatus.PENDING
        ):
            raise ApprovalRequired(
                request
            )

        if (
            request.status
            in {
                ApprovalStatus.REJECTED,
                ApprovalStatus.EXPIRED,
            }
        ):
            raise ApprovalRejected(
                request
            )

        decision = (
            self.manager.get_decision(
                request_id
            )
        )

        automatic = bool(
            decision is not None
            and decision.metadata.get(
                "automatic"
            )
            is True
        )

        return ApprovalGateResult(
            allowed=True,
            request=request,
            automatic=automatic,
        )
