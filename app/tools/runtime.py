from dataclasses import dataclass
from typing import Optional

from app.tools.contracts import (
    ToolExecutionRequest,
    ToolExecutionResult,
)
from app.tools.executor_runtime import (
    ToolExecutor,
)
from app.tools.selector import (
    ToolSelectionDecision,
    ToolSelector,
)


class ToolRuntimeError(Exception):
    """
    Raised when the dynamic tool runtime
    cannot complete its routing process.
    """


@dataclass
class ToolRuntimeResult:
    """
    Complete record of one dynamic
    tool-routing cycle.
    """

    decision: ToolSelectionDecision

    request: Optional[
        ToolExecutionRequest
    ]

    execution: Optional[
        ToolExecutionResult
    ]

    @property
    def tool_used(
        self,
    ) -> bool:
        return (
            self.request is not None
        )

    @property
    def success(
        self,
    ) -> bool:
        if not self.tool_used:
            return True

        if self.execution is None:
            return False

        return self.execution.success


class ToolRuntime:
    """
    Coordinates tool selection and
    validated execution.

    The selector may recommend a tool,
    but only the ToolExecutor is allowed
    to execute registered capabilities.
    """

    def __init__(
        self,
        selector: ToolSelector,
        executor: ToolExecutor,
    ):
        self.selector = selector
        self.executor = executor

    def run(
        self,
        task_description: str,
        context: Optional[
            dict
        ] = None,
    ) -> ToolRuntimeResult:
        decision = (
            self.selector.select(
                task_description=(
                    task_description
                ),
                context=context,
            )
        )

        request = (
            self.selector.create_request(
                decision
            )
        )

        if request is None:
            return ToolRuntimeResult(
                decision=decision,
                request=None,
                execution=None,
            )

        execution = (
            self.executor.execute(
                request
            )
        )

        return ToolRuntimeResult(
            decision=decision,
            request=request,
            execution=execution,
        )
