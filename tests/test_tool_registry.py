import pytest

from app.tools.contracts import (
    ToolCapability,
    ToolCategory,
    ToolParameter,
    ToolRiskLevel,
)
from app.tools.registry import (
    ToolRegistry,
    ToolRegistryError,
)


def fake_search(
    query,
):
    return {
        "query": query,
        "results": [],
    }


def fake_executor(
    command,
):
    return {
        "command": command,
        "exit_code": 0,
    }


def create_search_capability(
    enabled=True,
):
    return ToolCapability(
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
                description=(
                    "Search query."
                ),
                required=True,
                parameter_type="string",
            )
        ],
        tags=[
            "web",
            "research",
        ],
        enabled=enabled,
    )


def create_execution_capability(
    enabled=True,
):
    return ToolCapability(
        name="command_executor",
        description=(
            "Execute an approved command."
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
            )
        ],
        tags=[
            "execution",
            "testing",
        ],
        enabled=enabled,
    )


def test_tool_can_be_registered():
    registry = ToolRegistry()

    registry.register(
        create_search_capability(),
        fake_search,
    )

    assert registry.is_registered(
        "web_search"
    )

    assert registry.count() == 1


def test_registered_tool_can_be_resolved():
    registry = ToolRegistry()

    capability = (
        create_search_capability()
    )

    registry.register(
        capability,
        fake_search,
    )

    registered = registry.get(
        "web_search"
    )

    assert (
        registered.capability
        is capability
    )

    assert (
        registered.handler
        is fake_search
    )


def test_registry_returns_capability():
    registry = ToolRegistry()

    registry.register(
        create_search_capability(),
        fake_search,
    )

    capability = (
        registry.get_capability(
            "web_search"
        )
    )

    assert (
        capability.name
        == "web_search"
    )

    assert (
        capability.category
        == ToolCategory.SEARCH
    )


def test_registry_returns_handler():
    registry = ToolRegistry()

    registry.register(
        create_search_capability(),
        fake_search,
    )

    handler = (
        registry.get_handler(
            "web_search"
        )
    )

    result = handler(
        "FastAPI"
    )

    assert (
        result["query"]
        == "FastAPI"
    )


def test_duplicate_tool_registration_is_rejected():
    registry = ToolRegistry()

    registry.register(
        create_search_capability(),
        fake_search,
    )

    with pytest.raises(
        ToolRegistryError,
        match="already registered",
    ):
        registry.register(
            create_search_capability(),
            fake_search,
        )


def test_non_callable_handler_is_rejected():
    registry = ToolRegistry()

    with pytest.raises(
        ToolRegistryError,
        match="must be callable",
    ):
        registry.register(
            create_search_capability(),
            "not-callable",
        )


def test_missing_tool_raises_error():
    registry = ToolRegistry()

    with pytest.raises(
        ToolRegistryError,
        match="not registered",
    ):
        registry.get(
            "missing_tool"
        )


def test_tool_can_be_unregistered():
    registry = ToolRegistry()

    registry.register(
        create_search_capability(),
        fake_search,
    )

    registry.unregister(
        "web_search"
    )

    assert (
        registry.is_registered(
            "web_search"
        )
        is False
    )

    assert registry.count() == 0


def test_unregister_missing_tool_raises_error():
    registry = ToolRegistry()

    with pytest.raises(
        ToolRegistryError,
        match="not registered",
    ):
        registry.unregister(
            "missing_tool"
        )


def test_enabled_tools_are_listed_by_default():
    registry = ToolRegistry()

    registry.register(
        create_search_capability(
            enabled=True
        ),
        fake_search,
    )

    registry.register(
        create_execution_capability(
            enabled=False
        ),
        fake_executor,
    )

    capabilities = (
        registry.list_capabilities()
    )

    names = [
        capability.name
        for capability
        in capabilities
    ]

    assert names == [
        "web_search"
    ]


def test_disabled_tools_can_be_included():
    registry = ToolRegistry()

    registry.register(
        create_search_capability(
            enabled=True
        ),
        fake_search,
    )

    registry.register(
        create_execution_capability(
            enabled=False
        ),
        fake_executor,
    )

    capabilities = (
        registry.list_capabilities(
            enabled_only=False
        )
    )

    names = {
        capability.name
        for capability
        in capabilities
    }

    assert names == {
        "web_search",
        "command_executor",
    }


def test_registry_returns_enabled_names():
    registry = ToolRegistry()

    registry.register(
        create_search_capability(),
        fake_search,
    )

    registry.register(
        create_execution_capability(
            enabled=False
        ),
        fake_executor,
    )

    assert registry.names() == [
        "web_search"
    ]


def test_tool_can_be_disabled():
    registry = ToolRegistry()

    registry.register(
        create_search_capability(),
        fake_search,
    )

    registry.disable(
        "web_search"
    )

    assert (
        registry.is_enabled(
            "web_search"
        )
        is False
    )

    assert (
        "web_search"
        not in registry.names()
    )


def test_tool_can_be_enabled():
    registry = ToolRegistry()

    registry.register(
        create_search_capability(
            enabled=False
        ),
        fake_search,
    )

    registry.enable(
        "web_search"
    )

    assert (
        registry.is_enabled(
            "web_search"
        )
        is True
    )

    assert (
        "web_search"
        in registry.names()
    )


def test_count_can_filter_enabled_tools():
    registry = ToolRegistry()

    registry.register(
        create_search_capability(),
        fake_search,
    )

    registry.register(
        create_execution_capability(
            enabled=False
        ),
        fake_executor,
    )

    assert (
        registry.count()
        == 2
    )

    assert (
        registry.count(
            enabled_only=True
        )
        == 1
    )


def test_tool_capability_preserves_metadata():
    registry = ToolRegistry()

    capability = (
        create_search_capability()
    )

    capability.metadata = {
        "owner": "research_agent",
        "version": "1.0",
    }

    registry.register(
        capability,
        fake_search,
    )

    stored = (
        registry.get_capability(
            "web_search"
        )
    )

    assert (
        stored.metadata["owner"]
        == "research_agent"
    )

    assert (
        stored.metadata["version"]
        == "1.0"
    )


def test_tool_parameters_are_preserved():
    registry = ToolRegistry()

    registry.register(
        create_search_capability(),
        fake_search,
    )

    capability = (
        registry.get_capability(
            "web_search"
        )
    )

    assert (
        len(capability.parameters)
        == 1
    )

    parameter = (
        capability.parameters[0]
    )

    assert (
        parameter.name
        == "query"
    )

    assert (
        parameter.required
        is True
    )

    assert (
        parameter.parameter_type
        == "string"
    )


def test_risk_level_is_preserved():
    registry = ToolRegistry()

    registry.register(
        create_execution_capability(),
        fake_executor,
    )

    capability = (
        registry.get_capability(
            "command_executor"
        )
    )

    assert (
        capability.risk_level
        == ToolRiskLevel.HIGH
    )
