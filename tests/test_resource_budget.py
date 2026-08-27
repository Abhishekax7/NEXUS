import pytest

from app.governance.budget import (
    ResourceBudgetGuard,
)
from app.governance.models import (
    ResourceBudget,
    ResourceLimitExceeded,
    ResourceType,
    ResourceUsage,
)


def test_empty_budget_allows_usage():
    guard = ResourceBudgetGuard(
        ResourceBudget()
    )

    usage = ResourceUsage(
        llm_calls=100,
        tool_calls=100,
        tasks=100,
        replans=100,
        repairs=100,
        wall_time_seconds=1000,
    )

    assert (
        guard.allowed(
            usage
        )
        is True
    )


def test_usage_at_limit_is_allowed():
    guard = ResourceBudgetGuard(
        ResourceBudget(
            max_tasks=5
        )
    )

    usage = ResourceUsage(
        tasks=5
    )

    assert (
        guard.allowed(
            usage
        )
        is True
    )


def test_task_limit_violation_detected():
    guard = ResourceBudgetGuard(
        ResourceBudget(
            max_tasks=3
        )
    )

    usage = ResourceUsage(
        tasks=4
    )

    violations = guard.check(
        usage
    )

    assert len(
        violations
    ) == 1

    assert (
        violations[0].resource
        == ResourceType.TASKS
    )


def test_llm_limit_violation_detected():
    guard = ResourceBudgetGuard(
        ResourceBudget(
            max_llm_calls=2
        )
    )

    usage = ResourceUsage(
        llm_calls=3
    )

    violations = guard.check(
        usage
    )

    assert (
        violations[0].resource
        == ResourceType.LLM_CALLS
    )


def test_tool_limit_violation_detected():
    guard = ResourceBudgetGuard(
        ResourceBudget(
            max_tool_calls=4
        )
    )

    usage = ResourceUsage(
        tool_calls=5
    )

    assert (
        guard.check(
            usage
        )[0].resource
        == ResourceType.TOOL_CALLS
    )


def test_replan_limit_violation_detected():
    guard = ResourceBudgetGuard(
        ResourceBudget(
            max_replans=1
        )
    )

    usage = ResourceUsage(
        replans=2
    )

    assert (
        guard.check(
            usage
        )[0].resource
        == ResourceType.REPLANS
    )


def test_repair_limit_violation_detected():
    guard = ResourceBudgetGuard(
        ResourceBudget(
            max_repairs=2
        )
    )

    usage = ResourceUsage(
        repairs=3
    )

    assert (
        guard.check(
            usage
        )[0].resource
        == ResourceType.REPAIRS
    )


def test_wall_time_limit_violation_detected():
    guard = ResourceBudgetGuard(
        ResourceBudget(
            max_wall_time_seconds=10.0
        )
    )

    usage = ResourceUsage(
        wall_time_seconds=10.1
    )

    assert (
        guard.check(
            usage
        )[0].resource
        == ResourceType.WALL_TIME_SECONDS
    )


def test_multiple_violations_returned():
    guard = ResourceBudgetGuard(
        ResourceBudget(
            max_tasks=1,
            max_tool_calls=1,
        )
    )

    usage = ResourceUsage(
        tasks=2,
        tool_calls=2,
    )

    violations = guard.check(
        usage
    )

    assert len(
        violations
    ) == 2


def test_enforce_raises_when_budget_exceeded():
    guard = ResourceBudgetGuard(
        ResourceBudget(
            max_tasks=1
        )
    )

    usage = ResourceUsage(
        tasks=2
    )

    with pytest.raises(
        ResourceLimitExceeded,
        match="Resource budget exceeded",
    ):
        guard.enforce(
            usage
        )


def test_enforce_does_not_raise_when_allowed():
    guard = ResourceBudgetGuard(
        ResourceBudget(
            max_tasks=5
        )
    )

    guard.enforce(
        ResourceUsage(
            tasks=3
        )
    )


def test_remaining_reports_capacity():
    guard = ResourceBudgetGuard(
        ResourceBudget(
            max_tasks=10,
            max_tool_calls=5,
        )
    )

    remaining = guard.remaining(
        ResourceUsage(
            tasks=4,
            tool_calls=2,
        )
    )

    assert (
        remaining["tasks"]
        == 6
    )

    assert (
        remaining["tool_calls"]
        == 3
    )


def test_remaining_never_goes_negative():
    guard = ResourceBudgetGuard(
        ResourceBudget(
            max_tasks=1
        )
    )

    remaining = guard.remaining(
        ResourceUsage(
            tasks=5
        )
    )

    assert (
        remaining["tasks"]
        == 0
    )


def test_unlimited_resource_returns_none_remaining():
    guard = ResourceBudgetGuard(
        ResourceBudget()
    )

    remaining = guard.remaining(
        ResourceUsage(
            tasks=50
        )
    )

    assert (
        remaining["tasks"]
        is None
    )
