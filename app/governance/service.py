from dataclasses import dataclass
from typing import Optional

from app.governance.budget import (
    ResourceBudgetGuard,
)
from app.governance.limits import (
    ConcurrencyGuard,
    SlidingWindowRateLimiter,
)
from app.governance.models import (
    PolicyDecision,
    PolicyEffect,
    ResourceUsage,
)
from app.governance.policy import (
    PolicyEngine,
)


@dataclass
class GovernanceDecision:
    """
    Complete deterministic governance
    result for one NEXUS action.
    """

    action: str

    policy: PolicyDecision

    rate_allowed: bool

    concurrency_allowed: bool

    budget_allowed: bool

    allowed: bool

    requires_approval: bool


class GovernanceService:
    """
    Unified NEXUS production governance
    layer.

    Combines:
    - execution policy
    - resource budgets
    - rate limiting
    - concurrency limits
    """

    def __init__(
        self,
        *,
        policy_engine: PolicyEngine,
        budget_guard: Optional[
            ResourceBudgetGuard
        ] = None,
        rate_limiter: Optional[
            SlidingWindowRateLimiter
        ] = None,
        concurrency_guard: Optional[
            ConcurrencyGuard
        ] = None,
    ):
        self.policy_engine = (
            policy_engine
        )

        self.budget_guard = (
            budget_guard
        )

        self.rate_limiter = (
            rate_limiter
        )

        self.concurrency_guard = (
            concurrency_guard
        )

    def evaluate(
        self,
        *,
        action: str,
        subject: str,
        usage: Optional[
            ResourceUsage
        ] = None,
        context: Optional[
            dict
        ] = None,
    ) -> GovernanceDecision:
        policy = (
            self.policy_engine
            .evaluate(
                action,
                context=context,
            )
        )

        rate_allowed = True

        if self.rate_limiter is not None:
            rate_allowed = (
                self.rate_limiter
                .allowed(
                    subject
                )
            )

        concurrency_allowed = True

        if (
            self.concurrency_guard
            is not None
        ):
            concurrency_allowed = (
                self.concurrency_guard
                .available(
                    subject
                )
            )

        budget_allowed = True

        if (
            self.budget_guard
            is not None
            and usage is not None
        ):
            budget_allowed = (
                self.budget_guard
                .allowed(
                    usage
                )
            )

        allowed = (
            policy.effect
            != PolicyEffect.DENY
            and rate_allowed
            and concurrency_allowed
            and budget_allowed
        )

        requires_approval = (
            policy.requires_approval
            and rate_allowed
            and concurrency_allowed
            and budget_allowed
        )

        return GovernanceDecision(
            action=action,
            policy=policy,
            rate_allowed=rate_allowed,
            concurrency_allowed=(
                concurrency_allowed
            ),
            budget_allowed=(
                budget_allowed
            ),
            allowed=allowed,
            requires_approval=(
                requires_approval
            ),
        )

    def acquire(
        self,
        *,
        action: str,
        subject: str,
        usage: Optional[
            ResourceUsage
        ] = None,
        context: Optional[
            dict
        ] = None,
    ) -> GovernanceDecision:
        """
        Enforce governance and acquire
        execution capacity.

        Rate and concurrency slots are
        consumed only after policy and
        budget checks succeed.
        """

        policy = (
            self.policy_engine
            .enforce(
                action,
                context=context,
            )
        )

        if (
            self.budget_guard
            is not None
            and usage is not None
        ):
            self.budget_guard.enforce(
                usage
            )

        if self.rate_limiter is not None:
            self.rate_limiter.acquire(
                subject
            )

        if (
            self.concurrency_guard
            is not None
        ):
            self.concurrency_guard.acquire(
                subject
            )

        return GovernanceDecision(
            action=action,
            policy=policy,
            rate_allowed=True,
            concurrency_allowed=True,
            budget_allowed=True,
            allowed=(
                policy.effect
                != PolicyEffect.DENY
            ),
            requires_approval=(
                policy.requires_approval
            ),
        )

    def release(
        self,
        subject: str,
    ) -> None:
        """
        Release a concurrency slot after
        execution completes.
        """

        if (
            self.concurrency_guard
            is None
        ):
            return

        if (
            self.concurrency_guard
            .active_for(
                subject
            )
            <= 0
        ):
            return

        self.concurrency_guard.release(
            subject
        )
