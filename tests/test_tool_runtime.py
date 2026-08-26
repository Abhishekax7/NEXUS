import json

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
    ToolRuntimeResult,
)
from app.tools.selector import (
    ToolSelector,
)


class FakeLLM:
    def __init__(
        self,
        responses,
    ):
        self.responses = list(
            responses
        )
        self.calls = 0

    def generate(
        self,
        system_prompt,
        user_prompt,
        json_mode=False,
    ):
        index = min(
            self.calls,
            len(self.responses) - 1,
        )

        response = (
            self.responses[index]
        )

        self.calls += 1

        return response


def build_registry():
    registry = ToolRegistry()

    registry.register(
        ToolCapability(
            name="web_search",
            description=(
                "Search for technical "
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
                    description=(
                        "Search query."
                    ),
                    required=True,
                    parameter_type=(
                        "string"
                    ),
                )
            ],
        ),
        lambda query: {
            "query": query,
            "results": [
                "result-1",
                "result-2",
            ],
        },
    )

    registry.register(
        ToolCapability(
            name="failing_tool",
            description=(
                "Tool used to test "
                "execution failure."
            ),
            category=(
                ToolCategory.OTHER
            ),
            risk_level=(
                ToolRiskLevel.MEDIUM
            ),
            parameters=[],
        ),
        lambda: (
            _ for _ in ()
        ).throw(
            RuntimeError(
                "tool exploded"
            )
        ),
    )

    return registry


def build_runtime(
    responses,
):
    registry = build_registry()

    selector = ToolSelector(
        registry=registry,
        llm_client=FakeLLM(
            responses
        ),
    )

    executor = ToolExecutor(
        registry=registry
    )

    return ToolRuntime(
        selector=selector,
        executor=executor,
    )


def search_decision():
    return json.dumps(
        {
            "use_tool": True,
            "tool_name": (
                "web_search"
            ),
            "arguments": {
                "query": (
                    "FastAPI documentation"
                )
            },
            "reason": (
                "External technical "
                "information is required."
            ),
            "confidence": 0.96,
        }
    )


def no_tool_decision():
    return json.dumps(
        {
            "use_tool": False,
            "tool_name": None,
            "arguments": {},
            "reason": (
                "Existing context is "
                "sufficient."
            ),
            "confidence": 0.93,
        }
    )


def failing_decision():
    return json.dumps(
        {
            "use_tool": True,
            "tool_name": (
                "failing_tool"
            ),
            "arguments": {},
            "reason": (
                "Exercise failure path."
            ),
            "confidence": 0.9,
        }
    )


def test_runtime_selects_and_executes_tool():
    runtime = build_runtime(
        [
            search_decision()
        ]
    )

    result = runtime.run(
        "Find current FastAPI docs."
    )

    assert isinstance(
        result,
        ToolRuntimeResult,
    )

    assert (
        result.tool_used
        is True
    )

    assert (
        result.success
        is True
    )

    assert (
        result.decision.tool_name
        == "web_search"
    )

    assert (
        result.request.tool_name
        == "web_search"
    )

    assert (
        result.execution.tool_name
        == "web_search"
    )


def test_runtime_returns_tool_output():
    runtime = build_runtime(
        [
            search_decision()
        ]
    )

    result = runtime.run(
        "Research FastAPI."
    )

    assert (
        result.execution.output[
            "query"
        ]
        == "FastAPI documentation"
    )

    assert (
        result.execution.output[
            "results"
        ]
        == [
            "result-1",
            "result-2",
        ]
    )


def test_runtime_handles_no_tool():
    runtime = build_runtime(
        [
            no_tool_decision()
        ]
    )

    result = runtime.run(
        "Summarize existing context."
    )

    assert (
        result.tool_used
        is False
    )

    assert (
        result.success
        is True
    )

    assert result.request is None

    assert result.execution is None


def test_runtime_preserves_selection_reason():
    runtime = build_runtime(
        [
            search_decision()
        ]
    )

    result = runtime.run(
        "Research framework."
    )

    assert (
        result.request.reason
        == result.decision.reason
    )

    assert (
        result.execution.metadata[
            "reason"
        ]
        == result.decision.reason
    )


def test_runtime_preserves_arguments():
    runtime = build_runtime(
        [
            search_decision()
        ]
    )

    result = runtime.run(
        "Find documentation."
    )

    assert (
        result.request.arguments[
            "query"
        ]
        == "FastAPI documentation"
    )


def test_runtime_reports_execution_failure():
    runtime = build_runtime(
        [
            failing_decision()
        ]
    )

    result = runtime.run(
        "Exercise failing tool."
    )

    assert (
        result.tool_used
        is True
    )

    assert (
        result.success
        is False
    )

    assert (
        result.execution.success
        is False
    )

    assert (
        result.execution.error
        == "tool exploded"
    )


def test_runtime_passes_context_to_selector():
    class ContextAwareLLM:
        def __init__(self):
            self.prompt = None

        def generate(
            self,
            system_prompt,
            user_prompt,
            json_mode=False,
        ):
            self.prompt = user_prompt

            return (
                no_tool_decision()
            )

    registry = build_registry()

    fake_llm = ContextAwareLLM()

    selector = ToolSelector(
        registry=registry,
        llm_client=fake_llm,
    )

    runtime = ToolRuntime(
        selector=selector,
        executor=ToolExecutor(
            registry
        ),
    )

    runtime.run(
        "Analyze framework.",
        context={
            "framework": "FastAPI",
            "source": "requirements",
        },
    )

    assert (
        "FastAPI"
        in fake_llm.prompt
    )

    assert (
        "requirements"
        in fake_llm.prompt
    )


def test_runtime_uses_shared_registry():
    registry = build_registry()

    selector = ToolSelector(
        registry=registry,
        llm_client=FakeLLM(
            [
                search_decision()
            ]
        ),
    )

    executor = ToolExecutor(
        registry=registry
    )

    runtime = ToolRuntime(
        selector=selector,
        executor=executor,
    )

    assert (
        runtime.selector.registry
        is registry
    )

    assert (
        runtime.executor.registry
        is registry
    )


def test_runtime_result_success_for_no_tool():
    runtime = build_runtime(
        [
            no_tool_decision()
        ]
    )

    result = runtime.run(
        "No external capability needed."
    )

    assert result.success is True


def test_runtime_result_failure_when_execution_missing():
    decision = (
        ToolSelector(
            registry=build_registry(),
            llm_client=FakeLLM(
                [
                    search_decision()
                ]
            ),
        )
        .select(
            "Find documentation."
        )
    )

    from app.tools.contracts import (
        ToolExecutionRequest,
    )

    result = ToolRuntimeResult(
        decision=decision,
        request=ToolExecutionRequest(
            tool_name="web_search",
            arguments={
                "query": "FastAPI"
            },
            reason="Research.",
        ),
        execution=None,
    )

    assert (
        result.tool_used
        is True
    )

    assert (
        result.success
        is False
    )
