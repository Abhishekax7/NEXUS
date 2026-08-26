import json
from typing import Optional

from pydantic import (
    BaseModel,
    Field,
    ValidationError,
)

from app.core.llm import LLMClient
from app.tools.contracts import (
    ToolExecutionRequest,
)
from app.tools.registry import (
    ToolRegistry,
)


class ToolSelectionDecision(BaseModel):
    use_tool: bool

    tool_name: Optional[str] = None

    arguments: dict = {}

    reason: str = Field(
        min_length=1
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )


class ToolSelectionError(Exception):
    """
    Raised when a valid tool-selection
    decision cannot be produced.
    """


class ToolSelector:
    def __init__(
        self,
        registry: ToolRegistry,
        llm_client: Optional[
            LLMClient
        ] = None,
        max_validation_retries: int = 2,
    ):
        self.registry = registry

        self.llm = (
            llm_client
            or LLMClient()
        )

        self.max_validation_retries = (
            max_validation_retries
        )

    def _capability_snapshot(
        self,
    ) -> list[dict]:
        capabilities = []

        for capability in (
            self.registry
            .list_capabilities(
                enabled_only=True
            )
        ):
            capabilities.append(
                {
                    "name":
                        capability.name,
                    "description":
                        capability.description,
                    "category":
                        capability.category.value,
                    "risk_level":
                        capability.risk_level.value,
                    "parameters": [
                        {
                            "name":
                                parameter.name,
                            "description":
                                parameter.description,
                            "required":
                                parameter.required,
                            "parameter_type":
                                parameter.parameter_type,
                            "default":
                                parameter.default,
                        }
                        for parameter
                        in capability.parameters
                    ],
                    "tags":
                        list(
                            capability.tags
                        ),
                    "metadata":
                        capability.metadata,
                }
            )

        return capabilities

    def _validate_decision(
        self,
        decision: ToolSelectionDecision,
    ) -> None:
        if not decision.use_tool:
            if decision.tool_name is not None:
                raise ToolSelectionError(
                    "use_tool=false requires "
                    "tool_name to be null."
                )

            if decision.arguments:
                raise ToolSelectionError(
                    "use_tool=false requires "
                    "empty arguments."
                )

            return

        if not decision.tool_name:
            raise ToolSelectionError(
                "use_tool=true requires "
                "tool_name."
            )

        if not self.registry.is_registered(
            decision.tool_name
        ):
            raise ToolSelectionError(
                "Selected tool is not registered: "
                f"{decision.tool_name}"
            )

        if not self.registry.is_enabled(
            decision.tool_name
        ):
            raise ToolSelectionError(
                "Selected tool is disabled: "
                f"{decision.tool_name}"
            )

        capability = (
            self.registry.get_capability(
                decision.tool_name
            )
        )

        expected_names = {
            parameter.name
            for parameter
            in capability.parameters
        }

        unknown = (
            set(
                decision.arguments.keys()
            )
            - expected_names
        )

        if unknown:
            names = ", ".join(
                sorted(
                    unknown
                )
            )

            raise ToolSelectionError(
                "Selected tool received "
                "unknown arguments: "
                f"{names}"
            )

        for parameter in (
            capability.parameters
        ):
            if (
                parameter.required
                and parameter.name
                not in decision.arguments
                and parameter.default
                is None
            ):
                raise ToolSelectionError(
                    "Selected tool is missing "
                    "required argument: "
                    f"{parameter.name}"
                )

    def select(
        self,
        task_description: str,
        context: Optional[
            dict
        ] = None,
    ) -> ToolSelectionDecision:
        if not isinstance(
            task_description,
            str,
        ):
            raise ToolSelectionError(
                "task_description must "
                "be a string."
            )

        task_description = (
            task_description.strip()
        )

        if not task_description:
            raise ToolSelectionError(
                "task_description cannot "
                "be empty."
            )

        context = (
            context
            or {}
        )

        capabilities = (
            self._capability_snapshot()
        )

        system_prompt = (
            "You are the Tool Selector inside "
            "NEXUS, an autonomous AI software "
            "engineering system. Select at most "
            "one tool from the AVAILABLE TOOLS. "
            "Never invent a tool. Prefer not using "
            "a tool when the task can be completed "
            "without one. Return valid JSON only."
        )

        prompt = f"""
TASK:

{task_description}

CONTEXT:

{json.dumps(context, indent=2)}

AVAILABLE TOOLS:

{json.dumps(capabilities, indent=2)}

Choose whether a tool should be used.

Return exactly one JSON object containing:

use_tool
tool_name
arguments
reason
confidence

Rules:

- use only tools listed in AVAILABLE TOOLS

- never invent a tool name

- disabled tools are unavailable

- choose at most one tool

- if no tool is needed:
  use_tool=false
  tool_name=null
  arguments={{}}

- if use_tool=true:
  tool_name must be one available tool

- arguments must match that tool's
  declared parameters

- include all required arguments

- do not add unknown arguments

- prefer LOW risk tools when multiple
  tools can satisfy the task equally well

- MEDIUM or HIGH risk tools should only
  be selected when clearly necessary

- do not execute the tool

- only decide which tool should be used

- confidence must be between 0 and 1

- return JSON only
"""

        last_error = None

        for attempt in range(
            self.max_validation_retries + 1
        ):
            raw_output = (
                self.llm.generate(
                    system_prompt=system_prompt,
                    user_prompt=prompt,
                    json_mode=True,
                )
            )

            try:
                parsed = json.loads(
                    raw_output
                )

                decision = (
                    ToolSelectionDecision
                    .model_validate(
                        parsed
                    )
                )

                self._validate_decision(
                    decision
                )

                return decision

            except (
                json.JSONDecodeError,
                ValidationError,
                ToolSelectionError,
            ) as exc:
                last_error = exc

                prompt = f"""
Your previous tool-selection
response failed validation.

ERROR:

{exc}

PREVIOUS RESPONSE:

{raw_output}

TASK:

{task_description}

AVAILABLE TOOLS:

{json.dumps(capabilities, indent=2)}

Repair the decision.

Return exactly one JSON object containing:

use_tool
tool_name
arguments
reason
confidence

Rules:

- select only an available tool
- never invent tool names
- required arguments must be present
- unknown arguments are forbidden
- false => tool_name=null
- false => arguments={{}}
- return JSON only
"""

        raise ToolSelectionError(
            "Tool selection could not "
            "be validated after retries: "
            f"{last_error}"
        )

    def create_request(
        self,
        decision: ToolSelectionDecision,
    ) -> Optional[
        ToolExecutionRequest
    ]:
        self._validate_decision(
            decision
        )

        if not decision.use_tool:
            return None

        return ToolExecutionRequest(
            tool_name=decision.tool_name,
            arguments=dict(
                decision.arguments
            ),
            reason=decision.reason,
        )
