from typing import Optional

from app.governance.models import (
    BudgetViolation,
    ResourceBudget,
    ResourceLimitExceeded,
    ResourceType,
    ResourceUsage,
)


class ResourceBudgetGuard:
    """
    Deterministic resource-governance
    layer for NEXUS workflows.

    The guard checks observed execution
    usage against configured hard limits.
    """

    def __init__(
        self,
        budget: ResourceBudget,
    ):
        self.budget = budget

    def _violation(
        self,
        *,
        resource: ResourceType,
        limit: float,
        observed: float,
    ) -> BudgetViolation:
        return BudgetViolation(
            resource=resource,
            limit=limit,
            observed=observed,
            message=(
                f"Resource budget exceeded "
                f"for {resource.value}: "
                f"observed={observed}, "
                f"limit={limit}."
            ),
        )

    def check(
        self,
        usage: ResourceUsage,
    ) -> list[
        BudgetViolation
    ]:
        violations = []

        if (
            self.budget.max_llm_calls
            is not None
            and usage.llm_calls
            > self.budget.max_llm_calls
        ):
            violations.append(
                self._violation(
                    resource=(
                        ResourceType.LLM_CALLS
                    ),
                    limit=(
                        self.budget
                        .max_llm_calls
                    ),
                    observed=(
                        usage.llm_calls
                    ),
                )
            )

        if (
            self.budget.max_tool_calls
            is not None
            and usage.tool_calls
            > self.budget.max_tool_calls
        ):
            violations.append(
                self._violation(
                    resource=(
                        ResourceType.TOOL_CALLS
                    ),
                    limit=(
                        self.budget
                        .max_tool_calls
                    ),
                    observed=(
                        usage.tool_calls
                    ),
                )
            )

        if (
            self.budget.max_tasks
            is not None
            and usage.tasks
            > self.budget.max_tasks
        ):
            violations.append(
                self._violation(
                    resource=(
                        ResourceType.TASKS
                    ),
                    limit=(
                        self.budget
                        .max_tasks
                    ),
                    observed=(
                        usage.tasks
                    ),
                )
            )

        if (
            self.budget.max_replans
            is not None
            and usage.replans
            > self.budget.max_replans
        ):
            violations.append(
                self._violation(
                    resource=(
                        ResourceType.REPLANS
                    ),
                    limit=(
                        self.budget
                        .max_replans
                    ),
                    observed=(
                        usage.replans
                    ),
                )
            )

        if (
            self.budget.max_repairs
            is not None
            and usage.repairs
            > self.budget.max_repairs
        ):
            violations.append(
                self._violation(
                    resource=(
                        ResourceType.REPAIRS
                    ),
                    limit=(
                        self.budget
                        .max_repairs
                    ),
                    observed=(
                        usage.repairs
                    ),
                )
            )

        if (
            self.budget
            .max_wall_time_seconds
            is not None
            and usage.wall_time_seconds
            > self.budget
            .max_wall_time_seconds
        ):
            violations.append(
                self._violation(
                    resource=(
                        ResourceType
                        .WALL_TIME_SECONDS
                    ),
                    limit=(
                        self.budget
                        .max_wall_time_seconds
                    ),
                    observed=(
                        usage
                        .wall_time_seconds
                    ),
                )
            )

        return violations

    def enforce(
        self,
        usage: ResourceUsage,
    ) -> None:
        violations = self.check(
            usage
        )

        if not violations:
            return

        first = violations[0]

        raise ResourceLimitExceeded(
            first.message
        )

    def allowed(
        self,
        usage: ResourceUsage,
    ) -> bool:
        return (
            len(
                self.check(
                    usage
                )
            )
            == 0
        )

    def remaining(
        self,
        usage: ResourceUsage,
    ) -> dict[
        str,
        Optional[float],
    ]:
        def remaining_value(
            limit,
            observed,
        ):
            if limit is None:
                return None

            return max(
                0,
                limit - observed,
            )

        return {
            ResourceType.LLM_CALLS.value:
                remaining_value(
                    self.budget
                    .max_llm_calls,
                    usage.llm_calls,
                ),

            ResourceType.TOOL_CALLS.value:
                remaining_value(
                    self.budget
                    .max_tool_calls,
                    usage.tool_calls,
                ),

            ResourceType.TASKS.value:
                remaining_value(
                    self.budget
                    .max_tasks,
                    usage.tasks,
                ),

            ResourceType.REPLANS.value:
                remaining_value(
                    self.budget
                    .max_replans,
                    usage.replans,
                ),

            ResourceType.REPAIRS.value:
                remaining_value(
                    self.budget
                    .max_repairs,
                    usage.repairs,
                ),

            ResourceType
            .WALL_TIME_SECONDS
            .value:
                remaining_value(
                    self.budget
                    .max_wall_time_seconds,
                    usage.wall_time_seconds,
                ),
        }
