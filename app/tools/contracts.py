from enum import Enum
from typing import Any, Callable, Optional

from pydantic import (
    BaseModel,
    Field,
)


class ToolRiskLevel(
    str,
    Enum,
):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ToolCategory(
    str,
    Enum,
):
    SEARCH = "search"
    FILESYSTEM = "filesystem"
    EXECUTION = "execution"
    ANALYSIS = "analysis"
    MEMORY = "memory"
    OTHER = "other"


class ToolParameter(BaseModel):
    name: str = Field(
        min_length=1
    )

    description: str = Field(
        min_length=1
    )

    required: bool = True

    parameter_type: str = Field(
        min_length=1
    )

    default: Optional[Any] = None


class ToolCapability(BaseModel):
    name: str = Field(
        min_length=1
    )

    description: str = Field(
        min_length=1
    )

    category: ToolCategory

    risk_level: ToolRiskLevel = (
        ToolRiskLevel.LOW
    )

    parameters: list[
        ToolParameter
    ] = []

    tags: list[str] = []

    enabled: bool = True

    metadata: dict = {}


class ToolExecutionRequest(BaseModel):
    tool_name: str = Field(
        min_length=1
    )

    arguments: dict = {}

    reason: str = Field(
        min_length=1
    )


class ToolExecutionResult(BaseModel):
    tool_name: str = Field(
        min_length=1
    )

    success: bool

    output: Any = None

    error: Optional[str] = None

    metadata: dict = {}


ToolCallable = Callable[
    ...,
    Any,
]
