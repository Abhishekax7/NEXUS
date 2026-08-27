import json

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
    ApprovalStatus,
)

from app.tools.contracts import (
    ToolCapability,
    ToolCategory,
    ToolParameter,
    ToolRiskLevel,
)
from app.tools.executor_runtime import (
    ToolExecutor,
)
from app.tools.registry import (
    ToolRegistry,
)
from app.tools.runtime import (
    ToolRuntime,
)
from app.tools.selector import (
    ToolSelector,
)


class FakeLLM:
    def __init__(
        self,
        response,
    ):
        self.response = response

    def generate(
        self,
        system_prompt,
        user_prompt,
        json_mode=False,
    ):
        return self.response


class CallTracker:
    def __init__(
        self,
    ):
        self.calls = 0

    def execute(
        self,
        command,
    ):
        self.calls += 1

        return {
            "command": command,
            "executed": True,
        }


def build_runtime(
    risk_level,
    tracker,
):
    registry = ToolRegistry()

    registry.register(
        ToolCapability(
            name="controlled_tool",
            description=(
                "Execute a controlled "
                "test action."
            ),
            category=(
                ToolCategory.EXECUTION
            ),
            risk_level=risk_level,
            parameters=[
                ToolParameter(
                    name="command",
                    description=(
                        "Command to execute."
                    ),
                    required=True,
                    parameter_type="string",
                )
            ],
        ),
        tracker.execute,
    )

    llm = FakeLLM(
        json.dumps(
            {
                "use_tool": True,
                "tool_name":
                    "controlled_tool",
                "arguments": {
                    "command":
                        "pytest"
                },
                "reason": (
                    "Testing requires "
                    "execution."
                ),
                "confidence": 0.95,
            }
        )
    )

    selector = ToolSelector(
        registry=registry,
        llm_client=llm,
    )

    executor = ToolExecutor(
        registry=registry
    )

    manager = ApprovalManager()

    gate = ApprovalGate(
        manager=manager
    )

    runtime = ToolRuntime(
        selector=selector,
        executor=executor,
        approval_gate=gate,
    )

    return (
        runtime,
        manager,
    )


def test_low_risk_tool_executes_automatically():
    tracker = CallTracker()

    runtime, _ = build_runtime(
        ToolRiskLevel.LOW,
        tracker,
    )

    result = runtime.run(
        "Run the controlled tool.",
        run_id="run-1",
        requested_by="tester",
    )

    assert result.success is True

    assert (
        result.approval_granted
        is True
    )

    assert (
        result.approval_request
        is not None
    )

    assert (
        result.approval_request.status
        == ApprovalStatus.APPROVED
    )

    assert tracker.calls == 1


def test_high_risk_tool_does_not_execute_before_approval():
    tracker = CallTracker()

    runtime, _ = build_runtime(
        ToolRiskLevel.HIGH,
        tracker,
    )

    result = runtime.run(
        "Run the controlled tool.",
        run_id="run-1",
        requested_by="coder",
    )

    assert result.success is False

    assert (
        result.approval_required
        is True
    )

    assert (
        result.approval_granted
        is False
    )

    assert tracker.calls == 0


def test_high_risk_request_is_pending():
    tracker = CallTracker()

    runtime, _ = build_runtime(
        ToolRiskLevel.HIGH,
        tracker,
    )

    result = runtime.run(
        "Execute sensitive action.",
        run_id="run-1",
    )

    assert (
        result.approval_request
        is not None
    )

    assert (
        result.approval_request.status
        == ApprovalStatus.PENDING
    )


def test_high_risk_execution_can_resume_after_approval():
    tracker = CallTracker()

    runtime, manager = build_runtime(
        ToolRiskLevel.HIGH,
        tracker,
    )

    initial = runtime.run(
        "Execute sensitive action.",
        run_id="run-1",
    )

    request_id = (
        initial.approval_request.id
    )

    manager.approve(
        request_id,
        reason="Reviewed and approved.",
        decided_by="human",
    )

    resumed = runtime.resume(
        request_id
    )

    assert resumed.success is True

    assert (
        resumed.approval_granted
        is True
    )

    assert tracker.calls == 1


def test_pending_action_cannot_resume_before_decision():
    tracker = CallTracker()

    runtime, _ = build_runtime(
        ToolRiskLevel.HIGH,
        tracker,
    )

    initial = runtime.run(
        "Execute sensitive action.",
        run_id="run-1",
    )

    with pytest.raises(
        ApprovalRequired
    ):
        runtime.resume(
            initial.approval_request.id
        )

    assert tracker.calls == 0


def test_rejected_action_never_executes():
    tracker = CallTracker()

    runtime, manager = build_runtime(
        ToolRiskLevel.HIGH,
        tracker,
    )

    initial = runtime.run(
        "Execute sensitive action.",
        run_id="run-1",
    )

    request_id = (
        initial.approval_request.id
    )

    manager.reject(
        request_id,
        reason="Unsafe.",
        decided_by="human",
    )

    with pytest.raises(
        ApprovalRejected
    ):
        runtime.resume(
            request_id
        )

    assert tracker.calls == 0


def test_pending_execution_is_removed_after_success():
    tracker = CallTracker()

    runtime, manager = build_runtime(
        ToolRiskLevel.HIGH,
        tracker,
    )

    initial = runtime.run(
        "Execute sensitive action.",
        run_id="run-1",
    )

    request_id = (
        initial.approval_request.id
    )

    assert (
        request_id
        in runtime.pending_approval_ids()
    )

    manager.approve(
        request_id,
        reason="Approved.",
        decided_by="human",
    )

    runtime.resume(
        request_id
    )

    assert (
        request_id
        not in runtime.pending_approval_ids()
    )


def test_request_preserves_tool_payload():
    tracker = CallTracker()

    runtime, _ = build_runtime(
        ToolRiskLevel.HIGH,
        tracker,
    )

    result = runtime.run(
        "Execute sensitive action.",
        run_id="run-abc",
        requested_by="coder",
    )

    request = (
        result.approval_request
    )

    assert request is not None

    assert (
        request.run_id
        == "run-abc"
    )

    assert (
        request.requested_by
        == "coder"
    )

    assert (
        request.proposed_action[
            "tool_name"
        ]
        == "controlled_tool"
    )

    assert (
        request.proposed_action[
            "arguments"
        ][
            "command"
        ]
        == "pytest"
    )


def test_runtime_without_gate_preserves_old_behavior():
    tracker = CallTracker()

    registry = ToolRegistry()

    registry.register(
        ToolCapability(
            name="controlled_tool",
            description="High risk tool.",
            category=(
                ToolCategory.EXECUTION
            ),
            risk_level=(
                ToolRiskLevel.HIGH
            ),
            parameters=[
                ToolParameter(
                    name="command",
                    description="Command.",
                    required=True,
                    parameter_type="string",
                )
            ],
        ),
        tracker.execute,
    )

    selector = ToolSelector(
        registry=registry,
        llm_client=FakeLLM(
            json.dumps(
                {
                    "use_tool": True,
                    "tool_name":
                        "controlled_tool",
                    "arguments": {
                        "command":
                            "pytest"
                    },
                    "reason":
                        "Run test.",
                    "confidence":
                        0.9,
                }
            )
        ),
    )

    runtime = ToolRuntime(
        selector=selector,
        executor=ToolExecutor(
            registry
        ),
        approval_gate=None,
    )

    result = runtime.run(
        "Run tool."
    )

    assert result.success is True

    assert (
        result.approval_request
        is None
    )

    assert (
        result.approval_granted
        is None
    )

    assert tracker.calls == 1
