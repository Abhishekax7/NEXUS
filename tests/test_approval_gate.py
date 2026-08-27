import pytest

from app.approval.gate import (
    ApprovalGate,
    ApprovalRejected,
    ApprovalRequired,
)
from app.approval.manager import (
    ApprovalManager,
)
from app.approval.models import (
    ApprovalActionType,
    ApprovalRisk,
    ApprovalStatus,
)


def build_gate():
    return ApprovalGate(
        manager=ApprovalManager()
    )


def request_kwargs(
    risk,
):
    return {
        "run_id": "run-1",
        "action_type": (
            ApprovalActionType.TOOL_EXECUTION
        ),
        "risk": risk,
        "title": "Execute tool",
        "description": (
            "Execute a controlled tool."
        ),
        "proposed_action": {
            "tool": "example_tool",
            "arguments": {
                "query": "example",
            },
        },
        "reason": (
            "The workflow requires "
            "this tool."
        ),
        "requested_by": "research",
    }


def test_low_risk_execution_is_allowed():
    gate = build_gate()

    result = gate.request_execution(
        **request_kwargs(
            ApprovalRisk.LOW
        )
    )

    assert result.allowed is True

    assert (
        result.request.status
        == ApprovalStatus.APPROVED
    )


def test_low_risk_execution_is_automatic():
    gate = build_gate()

    result = gate.request_execution(
        **request_kwargs(
            ApprovalRisk.LOW
        )
    )

    assert result.automatic is True


def test_high_risk_execution_is_not_allowed():
    gate = build_gate()

    result = gate.request_execution(
        **request_kwargs(
            ApprovalRisk.HIGH
        )
    )

    assert result.allowed is False

    assert (
        result.request.status
        == ApprovalStatus.PENDING
    )


def test_require_execution_returns_for_safe_action():
    gate = build_gate()

    result = gate.require_execution(
        **request_kwargs(
            ApprovalRisk.LOW
        )
    )

    assert result.allowed is True


def test_require_execution_blocks_pending_action():
    gate = build_gate()

    with pytest.raises(
        ApprovalRequired
    ) as exc_info:
        gate.require_execution(
            **request_kwargs(
                ApprovalRisk.HIGH
            )
        )

    request = (
        exc_info.value.request
    )

    assert (
        request.status
        == ApprovalStatus.PENDING
    )


def test_pending_action_can_resume_after_approval():
    gate = build_gate()

    result = gate.request_execution(
        **request_kwargs(
            ApprovalRisk.HIGH
        )
    )

    request_id = (
        result.request.id
    )

    gate.manager.approve(
        request_id,
        reason="Human approved.",
        decided_by="human",
    )

    resumed = gate.resume(
        request_id
    )

    assert resumed.allowed is True

    assert (
        resumed.request.status
        == ApprovalStatus.APPROVED
    )


def test_pending_action_cannot_resume_early():
    gate = build_gate()

    result = gate.request_execution(
        **request_kwargs(
            ApprovalRisk.HIGH
        )
    )

    with pytest.raises(
        ApprovalRequired
    ):
        gate.resume(
            result.request.id
        )


def test_rejected_action_cannot_resume():
    gate = build_gate()

    result = gate.request_execution(
        **request_kwargs(
            ApprovalRisk.HIGH
        )
    )

    gate.manager.reject(
        result.request.id,
        reason="Unsafe action.",
        decided_by="human",
    )

    with pytest.raises(
        ApprovalRejected
    ):
        gate.resume(
            result.request.id
        )


def test_expired_action_cannot_resume():
    gate = build_gate()

    result = gate.request_execution(
        **request_kwargs(
            ApprovalRisk.HIGH
        )
    )

    gate.manager.expire(
        result.request.id
    )

    with pytest.raises(
        ApprovalRejected
    ):
        gate.resume(
            result.request.id
        )


def test_gate_preserves_action_payload():
    gate = build_gate()

    result = gate.request_execution(
        **request_kwargs(
            ApprovalRisk.HIGH
        )
    )

    assert (
        result.request
        .proposed_action[
            "tool"
        ]
        == "example_tool"
    )

    assert (
        result.request
        .proposed_action[
            "arguments"
        ][
            "query"
        ]
        == "example"
    )


def test_gate_preserves_requesting_agent():
    gate = build_gate()

    result = gate.request_execution(
        **request_kwargs(
            ApprovalRisk.HIGH
        )
    )

    assert (
        result.request.requested_by
        == "research"
    )


def test_manual_approval_is_not_automatic():
    gate = build_gate()

    result = gate.request_execution(
        **request_kwargs(
            ApprovalRisk.HIGH
        )
    )

    gate.manager.approve(
        result.request.id,
        reason="Reviewed.",
        decided_by="human",
    )

    resumed = gate.resume(
        result.request.id
    )

    assert (
        resumed.automatic
        is False
    )
