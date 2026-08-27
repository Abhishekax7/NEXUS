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

from app.governance.models import (
    PolicyEffect,
    ResourceUsage,
)
from app.governance.service import (
    GovernanceDecision,
    GovernanceService,
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

    governance: Optional[
        GovernanceDecision
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
    - governance policy enforcement
    - capability risk inspection
    - human approval enforcement
    - rate/concurrency governance
    - validated tool execution

    Governance and approval both happen
    BEFORE actual tool execution.
    """

    def __init__(
        self,
        selector: ToolSelector,
        executor: ToolExecutor,
        approval_gate: Optional[
            ApprovalGate
        ] = None,
        governance_service: Optional[
            GovernanceService
        ] = None,
    ):
        self.selector = selector
        self.executor = executor

        self.approval_gate = (
            approval_gate
        )

        self.governance_service = (
            governance_service
        )

        self._pending: dict[
            str,
            tuple[
                ToolSelectionDecision,
                ToolExecutionRequest,
                str,
                Optional[str],
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

    def _governance_action(
        self,
        request: ToolExecutionRequest,
    ) -> str:
        return (
            f"tool.{request.tool_name}"
        )

    def _evaluate_governance(
        self,
        *,
        request: ToolExecutionRequest,
        run_id: str,
        requested_by: Optional[
            str
        ],
    ) -> Optional[
        GovernanceDecision
    ]:
        if (
            self.governance_service
            is None
        ):
            return None

        context = {
            "run_id": run_id,
            "tool_name":
                request.tool_name,
            "requested_by":
                requested_by,
            "arguments":
                dict(
                    request.arguments
                ),
        }

        decision = (
            self.governance_service
            .evaluate(
                action=(
                    self._governance_action(
                        request
                    )
                ),
                subject=run_id,
                usage=ResourceUsage(
                    tool_calls=1
                ),
                context=context,
            )
        )

        if (
            decision.policy.effect
            == PolicyEffect.DENY
        ):
            self.governance_service \
                .policy_engine \
                .enforce(
                    self._governance_action(
                        request
                    ),
                    context=context,
                )

        return decision

    def _request_approval(
        self,
        *,
        request: ToolExecutionRequest,
        run_id: str,
        requested_by: Optional[
            str
        ],
        governance: Optional[
            GovernanceDecision
        ] = None,
    ):
        if self.approval_gate is None:
            if (
                governance is not None
                and governance
                .requires_approval
            ):
                raise ToolRuntimeError(
                    "Governance policy requires "
                    "human approval but the "
                    "approval gate is not "
                    "configured."
                )

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

        governance_requires_approval = (
            governance is not None
            and governance
            .requires_approval
        )

        # A governance REQUIRE_APPROVAL
        # decision must never be bypassed by
        # a low-risk tool configuration.
        if governance_requires_approval:
            risk = ApprovalRisk.HIGH

        metadata = {
            "tool_category":
                capability.category.value,

            "tool_risk_level":
                capability.risk_level.value,

            "governance_requires_approval":
                governance_requires_approval,
        }

        if governance is not None:
            metadata[
                "governance_policy_effect"
            ] = (
                governance
                .policy
                .effect
                .value
            )

            metadata[
                "governance_rule_id"
            ] = (
                governance
                .policy
                .matched_rule_id
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
                metadata=metadata,
            )
        )

    def _execute_governed(
        self,
        *,
        request: ToolExecutionRequest,
        run_id: str,
        requested_by: Optional[
            str
        ],
    ) -> ToolExecutionResult:
        if (
            self.governance_service
            is None
        ):
            return self.executor.execute(
                request
            )

        context = {
            "run_id": run_id,
            "tool_name":
                request.tool_name,
            "requested_by":
                requested_by,
            "arguments":
                dict(
                    request.arguments
                ),
        }

        self.governance_service.acquire(
            action=(
                self._governance_action(
                    request
                )
            ),
            subject=run_id,
            usage=ResourceUsage(
                tool_calls=1
            ),
            context=context,
        )

        try:
            return self.executor.execute(
                request
            )

        finally:
            self.governance_service.release(
                run_id
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
                governance=None,
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

        governance = (
            self._evaluate_governance(
                request=request,
                run_id=run_id,
                requested_by=requested_by,
            )
        )

        approval_result = (
            self._request_approval(
                request=request,
                run_id=run_id,
                requested_by=requested_by,
                governance=governance,
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
                run_id,
                requested_by,
            )

            return ToolRuntimeResult(
                decision=decision,
                request=request,
                execution=None,
                approval_request=(
                    approval_request
                ),
                approval_granted=False,
                governance=governance,
            )

        execution = (
            self._execute_governed(
                request=request,
                run_id=run_id,
                requested_by=requested_by,
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
            governance=governance,
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

        (
            decision,
            request,
            run_id,
            requested_by,
        ) = pending

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

        governance = (
            self._evaluate_governance(
                request=request,
                run_id=run_id,
                requested_by=requested_by,
            )
        )

        execution = (
            self._execute_governed(
                request=request,
                run_id=run_id,
                requested_by=requested_by,
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
            governance=governance,
        )

    def pending_approval_ids(
        self,
    ) -> list[str]:
        return list(
            self._pending.keys()
        )
