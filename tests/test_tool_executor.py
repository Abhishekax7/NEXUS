import pytest

from app.tools.contracts import (
    ToolCapability,
    ToolCategory,
    ToolExecutionRequest,
    ToolParameter,
    ToolRiskLevel,
)
from app.tools.executor_runtime import (
    ToolExecutionError,
    ToolExecutor,
)
from app.tools.registry import ToolRegistry


def build_registry():
    registry = ToolRegistry()

    registry.register(
        ToolCapability(
            name="web_search",
            description="Search technical information.",
            category=ToolCategory.SEARCH,
            risk_level=ToolRiskLevel.LOW,
            parameters=[
                ToolParameter(
                    name="query",
                    description="Search query",
                    required=True,
                    parameter_type="string",
                ),
                ToolParameter(
                    name="limit",
                    description="Maximum results",
                    required=False,
                    parameter_type="integer",
                    default=3,
                ),
            ],
        ),
        lambda query, limit=3: {
            "query": query,
            "limit": limit,
        },
    )

    registry.register(
        ToolCapability(
            name="failing_tool",
            description="Simulate tool failure.",
            category=ToolCategory.OTHER,
            risk_level=ToolRiskLevel.MEDIUM,
            parameters=[],
        ),
        lambda: (_ for _ in ()).throw(
            RuntimeError(
                "simulated failure"
            )
        ),
    )

    registry.register(
        ToolCapability(
            name="disabled_tool",
            description="Disabled tool.",
            category=ToolCategory.OTHER,
            risk_level=ToolRiskLevel.LOW,
            parameters=[],
            enabled=False,
        ),
        lambda: "should not run",
    )

    return registry


def test_executor_runs_registered_tool():
    executor = ToolExecutor(
        build_registry()
    )

    request = ToolExecutionRequest(
        tool_name="web_search",
        arguments={
            "query": "FastAPI docs",
        },
        reason="Need technical evidence.",
    )

    result = executor.execute(
        request
    )

    assert result.success is True

    assert (
        result.tool_name
        == "web_search"
    )

    assert (
        result.output["query"]
        == "FastAPI docs"
    )


def test_executor_applies_default_argument():
    executor = ToolExecutor(
        build_registry()
    )

    request = ToolExecutionRequest(
        tool_name="web_search",
        arguments={
            "query": "Python testing",
        },
        reason="Need search results.",
    )

    result = executor.execute(
        request
    )

    assert result.success is True

    assert (
        result.output["limit"]
        == 3
    )


def test_executor_accepts_explicit_optional_argument():
    executor = ToolExecutor(
        build_registry()
    )

    request = ToolExecutionRequest(
        tool_name="web_search",
        arguments={
            "query": "FastAPI",
            "limit": 7,
        },
        reason="Need more results.",
    )

    result = executor.execute(
        request
    )

    assert (
        result.output["limit"]
        == 7
    )


def test_executor_rejects_missing_required_argument():
    executor = ToolExecutor(
        build_registry()
    )

    request = ToolExecutionRequest(
        tool_name="web_search",
        arguments={},
        reason="Need search.",
    )

    with pytest.raises(
        ToolExecutionError,
        match="Missing required tool argument",
    ):
        executor.execute(
            request
        )


def test_executor_rejects_unknown_argument():
    executor = ToolExecutor(
        build_registry()
    )

    request = ToolExecutionRequest(
        tool_name="web_search",
        arguments={
            "query": "FastAPI",
            "unknown": True,
        },
        reason="Need search.",
    )

    with pytest.raises(
        ToolExecutionError,
        match="Unknown tool arguments",
    ):
        executor.execute(
            request
        )


def test_executor_rejects_unknown_tool():
    executor = ToolExecutor(
        build_registry()
    )

    request = ToolExecutionRequest(
        tool_name="missing_tool",
        arguments={},
        reason="Try missing tool.",
    )

    with pytest.raises(
        ToolExecutionError,
        match="not registered",
    ):
        executor.execute(
            request
        )


def test_executor_rejects_disabled_tool():
    executor = ToolExecutor(
        build_registry()
    )

    request = ToolExecutionRequest(
        tool_name="disabled_tool",
        arguments={},
        reason="Try disabled tool.",
    )

    with pytest.raises(
        ToolExecutionError,
        match="Tool is disabled",
    ):
        executor.execute(
            request
        )


def test_executor_returns_structured_failure():
    executor = ToolExecutor(
        build_registry()
    )

    request = ToolExecutionRequest(
        tool_name="failing_tool",
        arguments={},
        reason="Test failure handling.",
    )

    result = executor.execute(
        request
    )

    assert result.success is False

    assert (
        result.error
        == "simulated failure"
    )

    assert result.output is None


def test_executor_records_reason_in_metadata():
    executor = ToolExecutor(
        build_registry()
    )

    request = ToolExecutionRequest(
        tool_name="web_search",
        arguments={
            "query": "FastAPI",
        },
        reason="Research API framework.",
    )

    result = executor.execute(
        request
    )

    assert (
        result.metadata["reason"]
        == "Research API framework."
    )


def test_executor_records_risk_level():
    executor = ToolExecutor(
        build_registry()
    )

    request = ToolExecutionRequest(
        tool_name="web_search",
        arguments={
            "query": "FastAPI",
        },
        reason="Research.",
    )

    result = executor.execute(
        request
    )

    assert (
        result.metadata["risk_level"]
        == "low"
    )


def test_executor_records_category():
    executor = ToolExecutor(
        build_registry()
    )

    request = ToolExecutionRequest(
        tool_name="web_search",
        arguments={
            "query": "FastAPI",
        },
        reason="Research.",
    )

    result = executor.execute(
        request
    )

    assert (
        result.metadata["category"]
        == "search"
    )


def test_executor_preserves_arbitrary_handler_output():
    registry = ToolRegistry()

    registry.register(
        ToolCapability(
            name="analysis_tool",
            description="Return structured analysis.",
            category=ToolCategory.ANALYSIS,
            risk_level=ToolRiskLevel.LOW,
            parameters=[],
        ),
        lambda: {
            "score": 0.92,
            "issues": [
                "none"
            ],
        },
    )

    executor = ToolExecutor(
        registry
    )

    result = executor.execute(
        ToolExecutionRequest(
            tool_name="analysis_tool",
            arguments={},
            reason="Analyze output.",
        )
    )

    assert result.success is True

    assert (
        result.output["score"]
        == 0.92
    )

    assert (
        result.output["issues"]
        == [
            "none"
        ]
    )


def test_executor_can_run_no_argument_tool():
    registry = ToolRegistry()

    registry.register(
        ToolCapability(
            name="health_check",
            description="Return system health.",
            category=ToolCategory.ANALYSIS,
            parameters=[],
        ),
        lambda: "healthy",
    )

    executor = ToolExecutor(
        registry
    )

    result = executor.execute(
        ToolExecutionRequest(
            tool_name="health_check",
            arguments={},
            reason="Check health.",
        )
    )

    assert result.success is True

    assert (
        result.output
        == "healthy"
    )


def test_executor_uses_updated_tool_enabled_state():
    registry = build_registry()

    executor = ToolExecutor(
        registry
    )

    registry.disable(
        "web_search"
    )

    request = ToolExecutionRequest(
        tool_name="web_search",
        arguments={
            "query": "FastAPI",
        },
        reason="Search.",
    )

    with pytest.raises(
        ToolExecutionError,
        match="disabled",
    ):
        executor.execute(
            request
        )


def test_executor_can_run_tool_after_reenabled():
    registry = build_registry()

    registry.disable(
        "web_search"
    )

    registry.enable(
        "web_search"
    )

    executor = ToolExecutor(
        registry
    )

    result = executor.execute(
        ToolExecutionRequest(
            tool_name="web_search",
            arguments={
                "query": "FastAPI",
            },
            reason="Search.",
        )
    )

    assert result.success is True
