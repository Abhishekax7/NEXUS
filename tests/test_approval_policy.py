from app.approval.models import (
    ApprovalActionType,
    ApprovalRequest,
    ApprovalRisk,
)
from app.approval.policy import (
    ApprovalPolicy,
)


def build_request(
    risk: ApprovalRisk,
):
    return ApprovalRequest(
        run_id="run-1",
        action_type=(
            ApprovalActionType.TOOL_EXECUTION
        ),
        risk=risk,
        title="Execute action",
        description=(
            "Perform a controlled action."
        ),
        proposed_action={
            "tool": "example_tool",
        },
        reason=(
            "The workflow requires this action."
        ),
        requested_by="research",
    )


def test_low_risk_is_auto_allowed():
    policy = ApprovalPolicy()

    decision = policy.evaluate(
        build_request(
            ApprovalRisk.LOW
        )
    )

    assert (
        decision.auto_allowed
        is True
    )

    assert (
        decision.requires_approval
        is False
    )


def test_medium_risk_is_auto_allowed_by_default():
    policy = ApprovalPolicy()

    decision = policy.evaluate(
        build_request(
            ApprovalRisk.MEDIUM
        )
    )

    assert (
        decision.auto_allowed
        is True
    )

    assert (
        decision.requires_approval
        is False
    )


def test_medium_risk_can_require_approval():
    policy = ApprovalPolicy(
        require_medium_risk_approval=True
    )

    decision = policy.evaluate(
        build_request(
            ApprovalRisk.MEDIUM
        )
    )

    assert (
        decision.auto_allowed
        is False
    )

    assert (
        decision.requires_approval
        is True
    )


def test_high_risk_requires_approval():
    policy = ApprovalPolicy()

    decision = policy.evaluate(
        build_request(
            ApprovalRisk.HIGH
        )
    )

    assert (
        decision.auto_allowed
        is False
    )

    assert (
        decision.requires_approval
        is True
    )


def test_critical_risk_requires_approval():
    policy = ApprovalPolicy()

    decision = policy.evaluate(
        build_request(
            ApprovalRisk.CRITICAL
        )
    )

    assert (
        decision.auto_allowed
        is False
    )

    assert (
        decision.requires_approval
        is True
    )


def test_helper_allows_automatic_execution():
    policy = ApprovalPolicy()

    request = build_request(
        ApprovalRisk.LOW
    )

    assert (
        policy.allows_automatic_execution(
            request
        )
        is True
    )


def test_helper_requires_approval():
    policy = ApprovalPolicy()

    request = build_request(
        ApprovalRisk.HIGH
    )

    assert (
        policy.requires_approval(
            request
        )
        is True
    )


def test_policy_reason_is_present():
    policy = ApprovalPolicy()

    decision = policy.evaluate(
        build_request(
            ApprovalRisk.HIGH
        )
    )

    assert (
        isinstance(
            decision.reason,
            str,
        )
    )

    assert (
        len(
            decision.reason
        )
        > 0
    )
