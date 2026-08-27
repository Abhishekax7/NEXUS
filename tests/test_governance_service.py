import pytest

from app.governance.budget import (
    ResourceBudgetGuard,
)
from app.governance.limits import (
    ConcurrencyGuard,
    RateLimitConfig,
    RateLimitExceeded,
    SlidingWindowRateLimiter,
)
from app.governance.models import (
    PolicyDeniedError,
    PolicyEffect,
    PolicyRule,
    ResourceBudget,
    ResourceLimitExceeded,
    ResourceUsage,
)
from app.governance.policy import (
    PolicyEngine,
)
from app.governance.service import (
    GovernanceService,
)


def build_service(
    *,
    rules=None,
    budget=None,
    rate_limit=None,
    concurrency=None,
):
    policy = PolicyEngine(
        rules=rules or []
    )

    budget_guard = None

    if budget is not None:
        budget_guard = (
            ResourceBudgetGuard(
                budget
            )
        )

    limiter = None

    if rate_limit is not None:
        limiter = (
            SlidingWindowRateLimiter(
                rate_limit
            )
        )

    guard = None

    if concurrency is not None:
        guard = ConcurrencyGuard(
            max_concurrent=(
                concurrency
            )
        )

    return GovernanceService(
        policy_engine=policy,
        budget_guard=budget_guard,
        rate_limiter=limiter,
        concurrency_guard=guard,
    )


def test_default_action_is_allowed():
    service = build_service()

    decision = service.evaluate(
        action="tool.read",
        subject="run-1",
    )

    assert decision.allowed is True


def test_denied_policy_blocks_action():
    service = build_service(
        rules=[
            PolicyRule(
                id="deny-shell",
                action="tool.shell",
                effect=(
                    PolicyEffect.DENY
                ),
                reason="Shell denied.",
            )
        ]
    )

    decision = service.evaluate(
        action="tool.shell",
        subject="run-1",
    )

    assert decision.allowed is False


def test_approval_policy_is_reported():
    service = build_service(
        rules=[
            PolicyRule(
                id="approve-write",
                action="tool.write",
                effect=(
                    PolicyEffect
                    .REQUIRE_APPROVAL
                ),
                reason=(
                    "Write requires review."
                ),
            )
        ]
    )

    decision = service.evaluate(
        action="tool.write",
        subject="run-1",
    )

    assert (
        decision.requires_approval
        is True
    )


def test_budget_can_block_action():
    service = build_service(
        budget=ResourceBudget(
            max_tasks=2
        )
    )

    decision = service.evaluate(
        action="workflow.run",
        subject="run-1",
        usage=ResourceUsage(
            tasks=3
        ),
    )

    assert decision.allowed is False

    assert (
        decision.budget_allowed
        is False
    )


def test_rate_limit_can_block_action():
    service = build_service(
        rate_limit=(
            RateLimitConfig(
                max_requests=1,
                window_seconds=60,
            )
        )
    )

    service.acquire(
        action="workflow.run",
        subject="run-1",
    )

    decision = service.evaluate(
        action="workflow.run",
        subject="run-1",
    )

    assert decision.allowed is False

    assert (
        decision.rate_allowed
        is False
    )


def test_concurrency_can_block_action():
    service = build_service(
        concurrency=1
    )

    service.acquire(
        action="workflow.run",
        subject="run-1",
    )

    decision = service.evaluate(
        action="workflow.run",
        subject="run-2",
    )

    assert decision.allowed is False

    assert (
        decision.concurrency_allowed
        is False
    )


def test_acquire_denied_policy_raises():
    service = build_service(
        rules=[
            PolicyRule(
                id="deny",
                action="tool.delete",
                effect=(
                    PolicyEffect.DENY
                ),
                reason="Delete denied.",
            )
        ]
    )

    with pytest.raises(
        PolicyDeniedError
    ):
        service.acquire(
            action="tool.delete",
            subject="run-1",
        )


def test_acquire_budget_violation_raises():
    service = build_service(
        budget=ResourceBudget(
            max_tasks=1
        )
    )

    with pytest.raises(
        ResourceLimitExceeded
    ):
        service.acquire(
            action="workflow.run",
            subject="run-1",
            usage=ResourceUsage(
                tasks=2
            ),
        )


def test_acquire_rate_limit_raises():
    service = build_service(
        rate_limit=(
            RateLimitConfig(
                max_requests=1,
                window_seconds=60,
            )
        )
    )

    service.acquire(
        action="workflow.run",
        subject="run-1",
    )

    with pytest.raises(
        RateLimitExceeded
    ):
        service.acquire(
            action="workflow.run",
            subject="run-1",
        )


def test_release_frees_concurrency():
    service = build_service(
        concurrency=1
    )

    service.acquire(
        action="workflow.run",
        subject="run-1",
    )

    assert (
        service.concurrency_guard
        .active_total()
        == 1
    )

    service.release(
        "run-1"
    )

    assert (
        service.concurrency_guard
        .active_total()
        == 0
    )


def test_context_reaches_policy():
    service = build_service()

    decision = service.evaluate(
        action="tool.read",
        subject="run-1",
        context={
            "agent": "research"
        },
    )

    assert (
        decision.policy.metadata[
            "context"
        ][
            "agent"
        ]
        == "research"
    )


def test_unconfigured_guards_are_optional():
    service = build_service()

    service.acquire(
        action="workflow.run",
        subject="run-1",
    )

    service.release(
        "run-1"
    )
