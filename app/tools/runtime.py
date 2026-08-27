from dataclasses import dataclass
from typing import Optional

from app.approval.gate import (
    ApprovalGate,
    ApprovalRejected,
    ApprovalRequired,
)
from app.approval.models import (
    ApprovalActionType,
    ApprovalRequest,
    ApprovalRisk,
)

from app.tools.contracts import (
    ToolExecutionRequest,
    ToolExecutionResult,
    ToolRiskLevel,
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

    approval_request: Optional[
        ApprovalRequest
    ] = None

    approval_granted: Optional[
        bool
    ] = None

    @property
    def tool_used(
        self,
    ) -> bool:
        return (
            self.request is not None
        )

    @property
    def approval_required(
        self,
    ) -> bool:
        return (
            self.approval_request
            is not None
            and self.approval_granted
            is False
        )

    @property
    def success(
        self,
    ) -> bool:
        if not self.tool_used:
            return True

        if self.approval_required:
            return False

        if self.execution is None:
            return False

        return self.execution.success


class ToolRuntime:
    """
    Coordinates:

    - AI tool selection
    - capability risk inspection
    - approval policy enforcement
    - validated tool execution

    Approval happens BEFORE execution.
    """

    def __init__(
        self,
        selector: ToolSelector,
        executor: ToolExecutor,
        approval_gate: Optional[
            ApprovalGate
        ] = None,
    ):
        self.selector = selector

        self.executor = executor

        self.approval_gate = (
            approval_gate
        )

        self._pending: dict[
            str,
            tuple[
                ToolSelectionDecision,
                ToolExecutionRequest,
            ],
        ] = {}

    def _approval_risk_for(
        self,
        tool_name: str,
    ) -> ApprovalRisk:
        capability = (
            self.selector
            .registry
            .get_capability(
                tool_name
            )
        )

        configured_risk = (
            capability.metadata.get(
                "approval_risk"
            )
        )

        if configured_risk is not None:
            try:
                return ApprovalRisk(
                    configured_risk
                )

            except ValueError as exc:
                raise ToolRuntimeError(
                    "Invalid approval_risk "
                    f"metadata for tool "
                    f"'{tool_name}': "
                    f"{configured_risk}"
                ) from exc

        mapping = {
            ToolRiskLevel.LOW:
                ApprovalRisk.LOW,

            ToolRiskLevel.MEDIUM:
                ApprovalRisk.MEDIUM,

            ToolRiskLevel.HIGH:
                ApprovalRisk.HIGH,
        }

        try:
            return mapping[
                capability.risk_level
            ]

        except KeyError as exc:
            raise ToolRuntimeError(
                "Unsupported tool risk level: "
                f"{capability.risk_level}"
            ) from exc

    def _request_approval(
        self,
        *,
        request: ToolExecutionRequest,
        run_id: str,
        requested_by: Optional[
            str
        ],
    ):
        if self.approval_gate is None:
            return None

        capability = (
            self.selector
            .registry
            .get_capability(
                request.tool_name
            )
        )

        risk = self._approval_risk_for(
            request.tool_name
        )

        return (
            self.approval_gate
            .request_execution(
                run_id=run_id,
                action_type=(
                    ApprovalActionType
                    .TOOL_EXECUTION
                ),
                risk=risk,
                title=(
                    "Execute NEXUS tool: "
                    f"{request.tool_name}"
                ),
                description=(
                    capability.description
                ),
                proposed_action={
                    "tool_name":
                        request.tool_name,
                    "arguments":
                        dict(
                            request.arguments
                        ),
                },
                reason=request.reason,
                requested_by=(
                    requested_by
                ),
                metadata={
                    "tool_category":
                        capability
                        .category
                        .value,
                    "tool_risk_level":
                        capability
                        .risk_level
                        .value,
                },
            )
        )

    def run(
        self,
        task_description: str,
        context: Optional[
            dict
        ] = None,
        *,
        run_id: Optional[
            str
        ] = None,
        requested_by: Optional[
            str
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
                approval_request=None,
                approval_granted=None,
            )

        if run_id is None:
            if isinstance(
                context,
                dict,
            ):
                context_run_id = (
                    context.get(
                        "run_id"
                    )
                )
            else:
                context_run_id = None

            run_id = str(
                context_run_id
                or "tool-runtime"
            )

        approval_result = (
            self._request_approval(
                request=request,
                run_id=run_id,
                requested_by=requested_by,
            )
        )

        if (
            approval_result is not None
            and not approval_result.allowed
        ):
            approval_request = (
                approval_result.request
            )

            self._pending[
                approval_request.id
            ] = (
                decision,
                request,
            )

            return ToolRuntimeResult(
                decision=decision,
                request=request,
                execution=None,
                approval_request=(
                    approval_request
                ),
                approval_granted=False,
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
            approval_request=(
                approval_result.request
                if approval_result
                is not None
                else None
            ),
            approval_granted=(
                True
                if approval_result
                is not None
                else None
            ),
        )

    def resume(
        self,
        approval_request_id: str,
    ) -> ToolRuntimeResult:
        if self.approval_gate is None:
            raise ToolRuntimeError(
                "Approval gate is not configured."
            )

        pending = self._pending.get(
            approval_request_id
        )

        if pending is None:
            raise ToolRuntimeError(
                "Pending tool execution "
                "not found for approval "
                f"request: "
                f"{approval_request_id}"
            )

        decision, request = pending

        try:
            approval = (
                self.approval_gate
                .resume(
                    approval_request_id
                )
            )

        except ApprovalRequired:
            raise

        except ApprovalRejected:
            del self._pending[
                approval_request_id
            ]

            raise

        execution = (
            self.executor.execute(
                request
            )
        )

        del self._pending[
            approval_request_id
        ]

        return ToolRuntimeResult(
            decision=decision,
            request=request,
            execution=execution,
            approval_request=(
                approval.request
            ),
            approval_granted=True,
        )

    def pending_approval_ids(
        self,
    ) -> list[str]:
        return list(
            self._pending.keys()
        )
