import json

import pytest

from app.approval.gate import (
    ApprovalRejected,
    ApprovalRequired,
)
from app.core.runtime import (
    build_approval_gate,
    build_approval_manager,
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


class FakeSelectorLLM:
    def __init__(
        self,
        tool_name,
        arguments,
    ):
        self.tool_name = tool_name
        self.arguments = arguments

    def generate(
        self,
        system_prompt,
        user_prompt,
        json_mode=False,
    ):
        return json.dumps(
            {
                "use_tool": True,
                "tool_name": self.tool_name,
                "arguments": self.arguments,
                "reason": (
                    "The workflow requires "
                    "this capability."
                ),
                "confidence": 0.97,
            }
        )


class ExecutionTracker:
    def __init__(self):
        self.calls = []

    def low_handler(
        self,
        query,
    ):
        self.calls.append(
            (
                "low_tool",
                query,
            )
        )

        return {
            "query": query,
            "result": "safe-result",
        }

    def high_handler(
        self,
        command,
    ):
        self.calls.append(
            (
                "high_tool",
                command,
            )
        )

        return {
            "command": command,
            "executed": True,
        }


def build_registry(
    tracker,
):
    registry = ToolRegistry()

    registry.register(
        ToolCapability(
            name="low_tool",
            description=(
                "Read low-risk "
                "information."
            ),
            category=(
                ToolCategory.SEARCH
            ),
            risk_level=(
                ToolRiskLevel.LOW
            ),
            parameters=[
                ToolParameter(
                    name="query",
                    description="Query.",
                    required=True,
                    parameter_type="string",
                )
            ],
        ),
        tracker.low_handler,
    )

    registry.register(
        ToolCapability(
            name="high_tool",
            description=(
                "Execute a sensitive "
                "operation."
            ),
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
        tracker.high_handler,
    )

    return registry


def build_runtime(
    registry,
    *,
    tool_name,
    arguments,
):
    manager = (
        build_approval_manager()
    )

    gate = build_approval_gate(
        manager
    )

    selector = ToolSelector(
        registry=registry,
        llm_client=FakeSelectorLLM(
            tool_name=tool_name,
            arguments=arguments,
        ),
    )

    runtime = ToolRuntime(
        selector=selector,
        executor=ToolExecutor(
            registry=registry
        ),
        approval_gate=gate,
    )

    return (
        runtime,
        manager,
        gate,
    )


def test_low_risk_tool_executes_immediately():
    tracker = ExecutionTracker()

    registry = build_registry(
        tracker
    )

    runtime, manager, _ = (
        build_runtime(
            registry,
            tool_name="low_tool",
            arguments={
                "query": "FastAPI"
            },
        )
    )

    result = runtime.run(
        "Research FastAPI.",
        run_id="run-low",
        requested_by="research",
    )

    assert result.success is True

    assert (
        result.approval_granted
        is True
    )

    assert (
        result.execution
        is not None
    )

    assert tracker.calls == [
        (
            "low_tool",
            "FastAPI",
        )
    ]

    assert (
        manager.pending_requests()
        == []
    )


def test_high_risk_tool_is_blocked_before_approval():
    tracker = ExecutionTracker()

    registry = build_registry(
        tracker
    )

    runtime, manager, _ = (
        build_runtime(
            registry,
            tool_name="high_tool",
            arguments={
                "command": "pytest"
            },
        )
    )

    result = runtime.run(
        "Execute tests.",
        run_id="run-high",
        requested_by="coder",
    )

    assert result.success is False

    assert (
        result.approval_required
        is True
    )

    assert (
        result.execution
        is None
    )

    assert tracker.calls == []

    pending = (
        manager.pending_requests()
    )

    assert len(pending) == 1

    assert (
        pending[0].run_id
        == "run-high"
    )


def test_high_risk_tool_executes_only_after_approval():
    tracker = ExecutionTracker()

    registry = build_registry(
        tracker
    )

    runtime, manager, _ = (
        build_runtime(
            registry,
            tool_name="high_tool",
            arguments={
                "command": "pytest"
            },
        )
    )

    initial = runtime.run(
        "Execute tests.",
        run_id="run-high",
        requested_by="coder",
    )

    request_id = (
        initial.approval_request.id
    )

    assert tracker.calls == []

    manager.approve(
        request_id,
        reason=(
            "Reviewed and approved."
        ),
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

    assert (
        resumed.execution
        is not None
    )

    assert tracker.calls == [
        (
            "high_tool",
            "pytest",
        )
    ]


def test_high_risk_rejection_prevents_execution():
    tracker = ExecutionTracker()

    registry = build_registry(
        tracker
    )

    runtime, manager, _ = (
        build_runtime(
            registry,
            tool_name="high_tool",
            arguments={
                "command": "pytest"
            },
        )
    )

    initial = runtime.run(
        "Execute tests.",
        run_id="run-high",
    )

    request_id = (
        initial.approval_request.id
    )

    manager.reject(
        request_id,
        reason="Action rejected.",
        decided_by="human",
    )

    with pytest.raises(
        ApprovalRejected
    ):
        runtime.resume(
            request_id
        )

    assert tracker.calls == []


def test_pending_high_risk_tool_cannot_resume_early():
    tracker = ExecutionTracker()

    registry = build_registry(
        tracker
    )

    runtime, _, _ = (
        build_runtime(
            registry,
            tool_name="high_tool",
            arguments={
                "command": "pytest"
            },
        )
    )

    initial = runtime.run(
        "Execute tests.",
        run_id="run-high",
    )

    with pytest.raises(
        ApprovalRequired
    ):
        runtime.resume(
            initial.approval_request.id
        )

    assert tracker.calls == []


def test_execution_occurs_exactly_once_after_approval():
    tracker = ExecutionTracker()

    registry = build_registry(
        tracker
    )

    runtime, manager, _ = (
        build_runtime(
            registry,
            tool_name="high_tool",
            arguments={
                "command": "pytest"
            },
        )
    )

    initial = runtime.run(
        "Execute tests.",
        run_id="run-high",
    )

    request_id = (
        initial.approval_request.id
    )

    manager.approve(
        request_id,
        reason="Approved.",
        decided_by="human",
    )

    runtime.resume(
        request_id
    )

    assert len(
        tracker.calls
    ) == 1

    assert (
        request_id
        not in runtime.pending_approval_ids()
    )


def test_approval_request_preserves_context():
    tracker = ExecutionTracker()

    registry = build_registry(
        tracker
    )

    runtime, _, _ = (
        build_runtime(
            registry,
            tool_name="high_tool",
            arguments={
                "command": "pytest"
            },
        )
    )

    result = runtime.run(
        "Execute tests.",
        run_id="run-context",
        requested_by="tester",
    )

    request = (
        result.approval_request
    )

    assert request is not None

    assert (
        request.run_id
        == "run-context"
    )

    assert (
        request.requested_by
        == "tester"
    )

    assert (
        request.proposed_action[
            "tool_name"
        ]
        == "high_tool"
    )

    assert (
        request.proposed_action[
            "arguments"
        ][
            "command"
        ]
        == "pytest"
    )


def test_low_and_high_risk_paths_use_same_gate_architecture():
    tracker = ExecutionTracker()

    registry = build_registry(
        tracker
    )

    low_runtime, low_manager, low_gate = (
        build_runtime(
            registry,
            tool_name="low_tool",
            arguments={
                "query": "docs"
            },
        )
    )

    high_runtime, high_manager, high_gate = (
        build_runtime(
            registry,
            tool_name="high_tool",
            arguments={
                "command": "pytest"
            },
        )
    )

    assert (
        low_runtime.approval_gate
        is low_gate
    )

    assert (
        high_runtime.approval_gate
        is high_gate
    )

    assert (
        low_gate.manager
        is low_manager
    )

    assert (
        high_gate.manager
        is high_manager
    )

