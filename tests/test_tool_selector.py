import json

import pytest

from app.tools.contracts import (
    ToolCapability,
    ToolCategory,
    ToolParameter,
    ToolRiskLevel,
)
from app.tools.registry import ToolRegistry
from app.tools.selector import (
    ToolSelectionDecision,
    ToolSelectionError,
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
        self.last_user_prompt = None

    def generate(
        self,
        system_prompt,
        user_prompt,
        json_mode=False,
    ):
        self.last_user_prompt = user_prompt

        response = self.responses[
            min(
                self.calls,
                len(self.responses) - 1,
            )
        ]

        self.calls += 1

        return response


def build_registry():
    registry = ToolRegistry()

    registry.register(
        ToolCapability(
            name="web_search",
            description=(
                "Search the web for technical "
                "information."
            ),
            category=ToolCategory.SEARCH,
            risk_level=ToolRiskLevel.LOW,
            parameters=[
                ToolParameter(
                    name="query",
                    description="Search query.",
                    required=True,
                    parameter_type="string",
                ),
                ToolParameter(
                    name="limit",
                    description=(
                        "Maximum number of results."
                    ),
                    required=False,
                    parameter_type="integer",
                    default=3,
                ),
            ],
            tags=[
                "research",
                "web",
            ],
        ),
        lambda query, limit=3: {
            "query": query,
            "limit": limit,
        },
    )

    registry.register(
        ToolCapability(
            name="command_executor",
            description=(
                "Execute an approved local command."
            ),
            category=ToolCategory.EXECUTION,
            risk_level=ToolRiskLevel.HIGH,
            parameters=[
                ToolParameter(
                    name="command",
                    description=(
                        "Command to execute."
                    ),
                    required=True,
                    parameter_type="string",
                ),
            ],
            tags=[
                "execution",
                "testing",
            ],
        ),
        lambda command: {
            "command": command,
        },
    )

    registry.register(
        ToolCapability(
            name="disabled_tool",
            description="Disabled capability.",
            category=ToolCategory.OTHER,
            risk_level=ToolRiskLevel.LOW,
            parameters=[],
            enabled=False,
        ),
        lambda: None,
    )

    return registry


def valid_search_response():
    return json.dumps(
        {
            "use_tool": True,
            "tool_name": "web_search",
            "arguments": {
                "query": (
                    "FastAPI official documentation"
                )
            },
            "reason": (
                "Current technical evidence "
                "is required."
            ),
            "confidence": 0.94,
        }
    )


def valid_no_tool_response():
    return json.dumps(
        {
            "use_tool": False,
            "tool_name": None,
            "arguments": {},
            "reason": (
                "The task can be completed "
                "without external tools."
            ),
            "confidence": 0.9,
        }
    )


def test_selector_selects_registered_tool():
    registry = build_registry()

    selector = ToolSelector(
        registry=registry,
        llm_client=FakeLLM(
            [
                valid_search_response()
            ]
        ),
    )

    decision = selector.select(
        "Find current FastAPI documentation."
    )

    assert decision.use_tool is True

    assert (
        decision.tool_name
        == "web_search"
    )

    assert (
        decision.arguments["query"]
        == "FastAPI official documentation"
    )


def test_selector_accepts_no_tool_decision():
    selector = ToolSelector(
        registry=build_registry(),
        llm_client=FakeLLM(
            [
                valid_no_tool_response()
            ]
        ),
    )

    decision = selector.select(
        "Summarize the provided context.",
        context={
            "text": "Already available."
        },
    )

    assert (
        decision.use_tool
        is False
    )

    assert decision.tool_name is None
    assert decision.arguments == {}


def test_selector_prompt_contains_task():
    fake_llm = FakeLLM(
        [
            valid_search_response()
        ]
    )

    selector = ToolSelector(
        registry=build_registry(),
        llm_client=fake_llm,
    )

    selector.select(
        "Research FastAPI documentation."
    )

    assert (
        "Research FastAPI documentation"
        in fake_llm.last_user_prompt
    )


def test_selector_prompt_contains_enabled_tools():
    fake_llm = FakeLLM(
        [
            valid_search_response()
        ]
    )

    selector = ToolSelector(
        registry=build_registry(),
        llm_client=fake_llm,
    )

    selector.select(
        "Research FastAPI."
    )

    assert (
        "web_search"
        in fake_llm.last_user_prompt
    )

    assert (
        "command_executor"
        in fake_llm.last_user_prompt
    )


def test_selector_prompt_excludes_disabled_tools():
    fake_llm = FakeLLM(
        [
            valid_no_tool_response()
        ]
    )

    selector = ToolSelector(
        registry=build_registry(),
        llm_client=fake_llm,
    )

    selector.select(
        "Perform a simple task."
    )

    assert (
        "disabled_tool"
        not in fake_llm.last_user_prompt
    )


def test_selector_rejects_hallucinated_tool():
    invalid = json.dumps(
        {
            "use_tool": True,
            "tool_name": "delete_everything",
            "arguments": {},
            "reason": "Use invented tool.",
            "confidence": 0.8,
        }
    )

    fake_llm = FakeLLM(
        [
            invalid,
            valid_no_tool_response(),
        ]
    )

    selector = ToolSelector(
        registry=build_registry(),
        llm_client=fake_llm,
        max_validation_retries=1,
    )

    decision = selector.select(
        "Complete task."
    )

    assert fake_llm.calls == 2

    assert (
        decision.use_tool
        is False
    )


def test_selector_rejects_disabled_tool():
    invalid = json.dumps(
        {
            "use_tool": True,
            "tool_name": "disabled_tool",
            "arguments": {},
            "reason": "Use disabled tool.",
            "confidence": 0.8,
        }
    )

    fake_llm = FakeLLM(
        [
            invalid,
            valid_no_tool_response(),
        ]
    )

    selector = ToolSelector(
        registry=build_registry(),
        llm_client=fake_llm,
        max_validation_retries=1,
    )

    decision = selector.select(
        "Complete task."
    )

    assert fake_llm.calls == 2

    assert (
        decision.use_tool
        is False
    )


def test_selector_rejects_missing_required_argument():
    invalid = json.dumps(
        {
            "use_tool": True,
            "tool_name": "web_search",
            "arguments": {},
            "reason": "Search web.",
            "confidence": 0.8,
        }
    )

    fake_llm = FakeLLM(
        [
            invalid,
            valid_search_response(),
        ]
    )

    selector = ToolSelector(
        registry=build_registry(),
        llm_client=fake_llm,
        max_validation_retries=1,
    )

    decision = selector.select(
        "Find documentation."
    )

    assert fake_llm.calls == 2

    assert (
        decision.tool_name
        == "web_search"
    )


def test_selector_rejects_unknown_argument():
    invalid = json.dumps(
        {
            "use_tool": True,
            "tool_name": "web_search",
            "arguments": {
                "query": "FastAPI",
                "unknown": True,
            },
            "reason": "Search.",
            "confidence": 0.8,
        }
    )

    selector = ToolSelector(
        registry=build_registry(),
        llm_client=FakeLLM(
            [
                invalid,
                valid_search_response(),
            ]
        ),
        max_validation_retries=1,
    )

    decision = selector.select(
        "Find docs."
    )

    assert (
        decision.tool_name
        == "web_search"
    )


def test_no_tool_requires_null_tool_name():
    invalid = json.dumps(
        {
            "use_tool": False,
            "tool_name": "web_search",
            "arguments": {},
            "reason": "No tool required.",
            "confidence": 0.9,
        }
    )

    fake_llm = FakeLLM(
        [
            invalid,
            valid_no_tool_response(),
        ]
    )

    selector = ToolSelector(
        registry=build_registry(),
        llm_client=fake_llm,
        max_validation_retries=1,
    )

    decision = selector.select(
        "Simple task."
    )

    assert fake_llm.calls == 2

    assert decision.tool_name is None


def test_no_tool_requires_empty_arguments():
    invalid = json.dumps(
        {
            "use_tool": False,
            "tool_name": None,
            "arguments": {
                "query": "FastAPI"
            },
            "reason": "No tool required.",
            "confidence": 0.9,
        }
    )

    selector = ToolSelector(
        registry=build_registry(),
        llm_client=FakeLLM(
            [
                invalid,
                valid_no_tool_response(),
            ]
        ),
        max_validation_retries=1,
    )

    decision = selector.select(
        "Simple task."
    )

    assert (
        decision.arguments
        == {}
    )


def test_selector_retries_invalid_json():
    fake_llm = FakeLLM(
        [
            "not-json",
            valid_search_response(),
        ]
    )

    selector = ToolSelector(
        registry=build_registry(),
        llm_client=fake_llm,
        max_validation_retries=1,
    )

    decision = selector.select(
        "Research FastAPI."
    )

    assert fake_llm.calls == 2

    assert (
        decision.tool_name
        == "web_search"
    )


def test_selector_fails_after_retry_limit():
    selector = ToolSelector(
        registry=build_registry(),
        llm_client=FakeLLM(
            [
                "{}",
                "{}",
            ]
        ),
        max_validation_retries=1,
    )

    with pytest.raises(
        ToolSelectionError,
        match="could not be validated",
    ):
        selector.select(
            "Research something."
        )


def test_selector_rejects_empty_task_description():
    selector = ToolSelector(
        registry=build_registry(),
        llm_client=FakeLLM(
            [
                valid_no_tool_response()
            ]
        ),
    )

    with pytest.raises(
        ToolSelectionError,
        match="cannot be empty",
    ):
        selector.select(
            "   "
        )


def test_selector_rejects_non_string_task():
    selector = ToolSelector(
        registry=build_registry(),
        llm_client=FakeLLM(
            [
                valid_no_tool_response()
            ]
        ),
    )

    with pytest.raises(
        ToolSelectionError,
        match="must be a string",
    ):
        selector.select(
            123
        )


def test_selector_passes_context_to_prompt():
    fake_llm = FakeLLM(
        [
            valid_search_response()
        ]
    )

    selector = ToolSelector(
        registry=build_registry(),
        llm_client=fake_llm,
    )

    selector.select(
        "Research API framework.",
        context={
            "framework": "FastAPI",
            "requirement": "free tools",
        },
    )

    assert (
        "free tools"
        in fake_llm.last_user_prompt
    )


def test_create_request_from_tool_decision():
    selector = ToolSelector(
        registry=build_registry(),
        llm_client=FakeLLM(
            [
                valid_search_response()
            ]
        ),
    )

    decision = selector.select(
        "Find FastAPI docs."
    )

    request = selector.create_request(
        decision
    )

    assert request is not None

    assert (
        request.tool_name
        == "web_search"
    )

    assert (
        request.arguments["query"]
        == "FastAPI official documentation"
    )

    assert (
        request.reason
        == decision.reason
    )


def test_create_request_returns_none_for_no_tool():
    selector = ToolSelector(
        registry=build_registry(),
        llm_client=FakeLLM(
            [
                valid_no_tool_response()
            ]
        ),
    )

    decision = selector.select(
        "No tool needed."
    )

    request = selector.create_request(
        decision
    )

    assert request is None


def test_create_request_revalidates_decision():
    selector = ToolSelector(
        registry=build_registry(),
        llm_client=FakeLLM(
            [
                valid_no_tool_response()
            ]
        ),
    )

    invalid_decision = (
        ToolSelectionDecision(
            use_tool=True,
            tool_name="missing_tool",
            arguments={},
            reason="Invalid.",
            confidence=0.8,
        )
    )

    with pytest.raises(
        ToolSelectionError,
        match="not registered",
    ):
        selector.create_request(
            invalid_decision
        )
