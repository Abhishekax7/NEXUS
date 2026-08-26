from app.tools.contracts import (
    ToolExecutionRequest,
    ToolExecutionResult,
)
from app.tools.registry import (
    ToolRegistry,
    ToolRegistryError,
)


class ToolExecutionError(Exception):
    """Raised when a registered tool cannot be executed safely."""


class ToolExecutor:
    def __init__(
        self,
        registry: ToolRegistry,
    ):
        self.registry = registry

    def _validate_arguments(
        self,
        request: ToolExecutionRequest,
    ) -> None:
        try:
            capability = (
                self.registry.get_capability(
                    request.tool_name
                )
            )
        except ToolRegistryError as exc:
            raise ToolExecutionError(
                str(exc)
            ) from exc

        if not capability.enabled:
            raise ToolExecutionError(
                "Tool is disabled: "
                f"{request.tool_name}"
            )

        supplied = request.arguments

        expected_names = {
            parameter.name
            for parameter
            in capability.parameters
        }

        unknown_arguments = (
            set(supplied.keys())
            - expected_names
        )

        if unknown_arguments:
            names = ", ".join(
                sorted(
                    unknown_arguments
                )
            )

            raise ToolExecutionError(
                "Unknown tool arguments: "
                f"{names}"
            )

        for parameter in capability.parameters:
            if (
                parameter.required
                and parameter.name
                not in supplied
                and parameter.default
                is None
            ):
                raise ToolExecutionError(
                    "Missing required tool argument: "
                    f"{parameter.name}"
                )

    def _build_arguments(
        self,
        request: ToolExecutionRequest,
    ) -> dict:
        capability = (
            self.registry.get_capability(
                request.tool_name
            )
        )

        arguments = dict(
            request.arguments
        )

        for parameter in capability.parameters:
            if (
                parameter.name
                not in arguments
                and parameter.default
                is not None
            ):
                arguments[
                    parameter.name
                ] = parameter.default

        return arguments

    def execute(
        self,
        request: ToolExecutionRequest,
    ) -> ToolExecutionResult:
        self._validate_arguments(
            request
        )

        try:
            registered = self.registry.get(
                request.tool_name
            )
        except ToolRegistryError as exc:
            raise ToolExecutionError(
                str(exc)
            ) from exc

        arguments = self._build_arguments(
            request
        )

        try:
            output = registered.handler(
                **arguments
            )

            return ToolExecutionResult(
                tool_name=request.tool_name,
                success=True,
                output=output,
                error=None,
                metadata={
                    "reason": request.reason,
                    "risk_level": (
                        registered.capability
                        .risk_level.value
                    ),
                    "category": (
                        registered.capability
                        .category.value
                    ),
                },
            )

        except Exception as exc:
            return ToolExecutionResult(
                tool_name=request.tool_name,
                success=False,
                output=None,
                error=str(exc),
                metadata={
                    "reason": request.reason,
                    "risk_level": (
                        registered.capability
                        .risk_level.value
                    ),
                    "category": (
                        registered.capability
                        .category.value
                    ),
                },
            )
