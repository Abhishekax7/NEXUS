from dataclasses import dataclass

from app.tools.contracts import (
    ToolCallable,
    ToolCapability,
)


class ToolRegistryError(Exception):
    """
    Raised when tool registration
    or resolution fails.
    """


@dataclass
class RegisteredTool:
    capability: ToolCapability

    handler: ToolCallable


class ToolRegistry:
    """
    Registry containing tools that NEXUS
    agents are allowed to discover and use.
    """

    def __init__(
        self,
    ):
        self._tools: dict[
            str,
            RegisteredTool,
        ] = {}

    def register(
        self,
        capability: ToolCapability,
        handler: ToolCallable,
    ) -> None:
        if not callable(
            handler
        ):
            raise ToolRegistryError(
                "Tool handler must be callable."
            )

        name = capability.name.strip()

        if not name:
            raise ToolRegistryError(
                "Tool name cannot be empty."
            )

        if name in self._tools:
            raise ToolRegistryError(
                "Tool already registered: "
                f"{name}"
            )

        self._tools[
            name
        ] = RegisteredTool(
            capability=capability,
            handler=handler,
        )

    def unregister(
        self,
        name: str,
    ) -> None:
        if name not in self._tools:
            raise ToolRegistryError(
                "Tool not registered: "
                f"{name}"
            )

        del self._tools[
            name
        ]

    def get(
        self,
        name: str,
    ) -> RegisteredTool:
        tool = self._tools.get(
            name
        )

        if tool is None:
            raise ToolRegistryError(
                "Tool not registered: "
                f"{name}"
            )

        return tool

    def get_capability(
        self,
        name: str,
    ) -> ToolCapability:
        return self.get(
            name
        ).capability

    def get_handler(
        self,
        name: str,
    ) -> ToolCallable:
        return self.get(
            name
        ).handler

    def list_capabilities(
        self,
        enabled_only: bool = True,
    ) -> list[ToolCapability]:
        capabilities = []

        for registered in (
            self._tools.values()
        ):
            capability = (
                registered.capability
            )

            if (
                enabled_only
                and not capability.enabled
            ):
                continue

            capabilities.append(
                capability
            )

        return capabilities

    def names(
        self,
        enabled_only: bool = True,
    ) -> list[str]:
        return [
            capability.name
            for capability
            in self.list_capabilities(
                enabled_only=enabled_only
            )
        ]

    def is_registered(
        self,
        name: str,
    ) -> bool:
        return (
            name
            in self._tools
        )

    def is_enabled(
        self,
        name: str,
    ) -> bool:
        capability = (
            self.get_capability(
                name
            )
        )

        return capability.enabled

    def enable(
        self,
        name: str,
    ) -> None:
        capability = (
            self.get_capability(
                name
            )
        )

        capability.enabled = True

    def disable(
        self,
        name: str,
    ) -> None:
        capability = (
            self.get_capability(
                name
            )
        )

        capability.enabled = False

    def count(
        self,
        enabled_only: bool = False,
    ) -> int:
        if not enabled_only:
            return len(
                self._tools
            )

        return len(
            self.list_capabilities(
                enabled_only=True
            )
        )
