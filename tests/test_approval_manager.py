import pytest

from app.approval.manager import (
    ApprovalManager,
)
from app.approval.models import (
    ApprovalActionType,
    ApprovalError,
    ApprovalRequest,
    ApprovalRisk,
    ApprovalStatus,
)
from app.approval.policy import (
    ApprovalPolicy,
)


def build_request(
    risk=ApprovalRisk.HIGH,
):
    return ApprovalRequest(
        run_id="run-1",
        action_type=(
            ApprovalActionType.TOOL_EXECUTION
        ),
        risk=risk,
        title="Execute sensitive tool",
        description=(
            "Execute a controlled action."
        ),
        proposed_action={
            "tool": "command_executor",
        },
        reason=(
            "The workflow requires "
            "this capability."
        ),
        requested_by="coder",
    )


def test_low_risk_request_is_auto_approved():
    manager = ApprovalManager()

    request = build_request(
        ApprovalRisk.LOW
    )

    result = manager.create_request(
        request
    )

    assert result.allowed is True

    assert (
        result.request.status
        == ApprovalStatus.APPROVED
    )

    assert (
        result.decision
        is not None
    )

    assert (
        result.decision.approved
        is True
    )

    assert (
        result.decision.decided_by
        == "policy"
    )


def test_high_risk_request_stays_pending():
    manager = ApprovalManager()

    request = build_request(
        ApprovalRisk.HIGH
    )

    result = manager.create_request(
        request
    )

    assert result.allowed is False

    assert (
        result.request.status
        == ApprovalStatus.PENDING
    )

    assert (
        result.decision
        is None
    )


def test_pending_request_can_be_approved():
    manager = ApprovalManager()

    request = build_request()

    manager.create_request(
        request
    )

    result = manager.approve(
        request.id,
        reason="Reviewed and approved.",
        decided_by="human",
    )

    assert result.allowed is True

    assert (
        result.request.status
        == ApprovalStatus.APPROVED
    )

    assert (
        result.decision.approved
        is True
    )

    assert (
        result.decision.decided_by
        == "human"
    )


def test_pending_request_can_be_rejected():
    manager = ApprovalManager()

    request = build_request()

    manager.create_request(
        request
    )

    result = manager.reject(
        request.id,
        reason="Too risky.",
        decided_by="human",
    )

    assert result.allowed is False

    assert (
        result.request.status
        == ApprovalStatus.REJECTED
    )

    assert (
        result.decision.approved
        is False
    )


def test_pending_request_can_expire():
    manager = ApprovalManager()

    request = build_request()

    manager.create_request(
        request
    )

    expired = manager.expire(
        request.id
    )

    assert (
        expired.status
        == ApprovalStatus.EXPIRED
    )


def test_decision_can_be_retrieved():
    manager = ApprovalManager()

    request = build_request()

    manager.create_request(
        request
    )

    manager.approve(
        request.id,
        reason="Approved.",
        decided_by="reviewer",
    )

    decision = manager.get_decision(
        request.id
    )

    assert decision is not None

    assert (
        decision.approved
        is True
    )


def test_pending_requests_are_listed():
    manager = ApprovalManager()

    first = build_request()
    second = build_request()

    manager.create_request(
        first
    )

    manager.create_request(
        second
    )

    pending = (
        manager.pending_requests()
    )

    ids = {
        request.id
        for request in pending
    }

    assert ids == {
        first.id,
        second.id,
    }


def test_approved_request_is_removed_from_pending():
    manager = ApprovalManager()

    request = build_request()

    manager.create_request(
        request
    )

    manager.approve(
        request.id,
        reason="Approved.",
        decided_by="human",
    )

    assert (
        manager.pending_requests()
        == []
    )


def test_request_cannot_be_decided_twice():
    manager = ApprovalManager()

    request = build_request()

    manager.create_request(
        request
    )

    manager.approve(
        request.id,
        reason="Approved.",
        decided_by="human",
    )

    with pytest.raises(
        ApprovalError,
        match="no longer pending",
    ):
        manager.reject(
            request.id,
            reason="Changed mind.",
            decided_by="human",
        )


def test_missing_request_cannot_be_approved():
    manager = ApprovalManager()

    with pytest.raises(
        ApprovalError,
        match="not found",
    ):
        manager.approve(
            "missing",
            reason="Approved.",
            decided_by="human",
        )


def test_duplicate_request_is_rejected():
    manager = ApprovalManager()

    request = build_request()

    manager.create_request(
        request
    )

    with pytest.raises(
        ApprovalError,
        match="already exists",
    ):
        manager.create_request(
            request
        )


def test_medium_risk_can_be_forced_pending():
    manager = ApprovalManager(
        policy=ApprovalPolicy(
            require_medium_risk_approval=True
        )
    )

    request = build_request(
        ApprovalRisk.MEDIUM
    )

    result = manager.create_request(
        request
    )

    assert result.allowed is False

    assert (
        result.request.status
        == ApprovalStatus.PENDING
    )


def test_all_requests_are_exposed():
    manager = ApprovalManager()

    first = build_request(
        ApprovalRisk.LOW
    )

    second = build_request(
        ApprovalRisk.HIGH
    )

    manager.create_request(
        first
    )

    manager.create_request(
        second
    )

    requests = (
        manager.all_requests()
    )

    assert len(requests) == 2
