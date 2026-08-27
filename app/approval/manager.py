from typing import Optional

from app.approval.models import (
    ApprovalDecision,
    ApprovalError,
    ApprovalRequest,
    ApprovalResult,
    ApprovalStatus,
)
from app.approval.policy import (
    ApprovalPolicy,
)


class ApprovalManager:
    """
    Manage approval request lifecycle
    for NEXUS human-in-the-loop actions.
    """

    def __init__(
        self,
        policy: Optional[
            ApprovalPolicy
        ] = None,
    ):
        self.policy = (
            policy
            or ApprovalPolicy()
        )

        self._requests: dict[
            str,
            ApprovalRequest,
        ] = {}

        self._decisions: dict[
            str,
            ApprovalDecision,
        ] = {}

    def create_request(
        self,
        request: ApprovalRequest,
    ) -> ApprovalResult:
        if request.id in self._requests:
            raise ApprovalError(
                "Approval request already exists: "
                f"{request.id}"
            )

        policy_decision = (
            self.policy.evaluate(
                request
            )
        )

        if (
            policy_decision.auto_allowed
            and not policy_decision.requires_approval
        ):
            request.status = (
                ApprovalStatus.APPROVED
            )

            decision = ApprovalDecision(
                request_id=request.id,
                approved=True,
                reason=(
                    policy_decision.reason
                ),
                decided_by="policy",
                metadata={
                    "automatic": True,
                },
            )

            self._requests[
                request.id
            ] = request

            self._decisions[
                request.id
            ] = decision

            return ApprovalResult(
                request=request,
                decision=decision,
                allowed=True,
            )

        request.status = (
            ApprovalStatus.PENDING
        )

        self._requests[
            request.id
        ] = request

        return ApprovalResult(
            request=request,
            decision=None,
            allowed=False,
        )

    def approve(
        self,
        request_id: str,
        *,
        reason: str,
        decided_by: str,
        metadata: Optional[
            dict
        ] = None,
    ) -> ApprovalResult:
        request = self._get_request(
            request_id
        )

        self._ensure_pending(
            request
        )

        decision = ApprovalDecision(
            request_id=request.id,
            approved=True,
            reason=reason,
            decided_by=decided_by,
            metadata=(
                metadata
                or {}
            ),
        )

        request.status = (
            ApprovalStatus.APPROVED
        )

        self._decisions[
            request.id
        ] = decision

        return ApprovalResult(
            request=request,
            decision=decision,
            allowed=True,
        )

    def reject(
        self,
        request_id: str,
        *,
        reason: str,
        decided_by: str,
        metadata: Optional[
            dict
        ] = None,
    ) -> ApprovalResult:
        request = self._get_request(
            request_id
        )

        self._ensure_pending(
            request
        )

        decision = ApprovalDecision(
            request_id=request.id,
            approved=False,
            reason=reason,
            decided_by=decided_by,
            metadata=(
                metadata
                or {}
            ),
        )

        request.status = (
            ApprovalStatus.REJECTED
        )

        self._decisions[
            request.id
        ] = decision

        return ApprovalResult(
            request=request,
            decision=decision,
            allowed=False,
        )

    def expire(
        self,
        request_id: str,
    ) -> ApprovalRequest:
        request = self._get_request(
            request_id
        )

        self._ensure_pending(
            request
        )

        request.status = (
            ApprovalStatus.EXPIRED
        )

        return request

    def get_request(
        self,
        request_id: str,
    ) -> Optional[
        ApprovalRequest
    ]:
        return self._requests.get(
            request_id
        )

    def get_decision(
        self,
        request_id: str,
    ) -> Optional[
        ApprovalDecision
    ]:
        return self._decisions.get(
            request_id
        )

    def pending_requests(
        self,
    ) -> list[
        ApprovalRequest
    ]:
        return [
            request
            for request
            in self._requests.values()
            if (
                request.status
                == ApprovalStatus.PENDING
            )
        ]

    def all_requests(
        self,
    ) -> list[
        ApprovalRequest
    ]:
        return list(
            self._requests.values()
        )

    def _get_request(
        self,
        request_id: str,
    ) -> ApprovalRequest:
        request = (
            self._requests.get(
                request_id
            )
        )

        if request is None:
            raise ApprovalError(
                "Approval request not found: "
                f"{request_id}"
            )

        return request

    def _ensure_pending(
        self,
        request: ApprovalRequest,
    ) -> None:
        if (
            request.status
            != ApprovalStatus.PENDING
        ):
            raise ApprovalError(
                "Approval request is no longer "
                "pending: "
                f"{request.id}"
            )
