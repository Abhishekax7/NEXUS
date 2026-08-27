from collections import deque
from dataclasses import dataclass
from time import monotonic
from typing import Optional

from app.governance.models import (
    GovernanceError,
)


class RateLimitExceeded(
    GovernanceError
):
    """
    Raised when an execution subject
    exceeds its configured rate limit.
    """


class ConcurrencyLimitExceeded(
    GovernanceError
):
    """
    Raised when too many concurrent
    executions are active.
    """


@dataclass
class RateLimitConfig:
    max_requests: int
    window_seconds: float

    def __post_init__(
        self,
    ):
        if self.max_requests <= 0:
            raise ValueError(
                "max_requests must be "
                "greater than zero."
            )

        if self.window_seconds <= 0:
            raise ValueError(
                "window_seconds must be "
                "greater than zero."
            )


class SlidingWindowRateLimiter:
    """
    Deterministic sliding-window
    in-memory rate limiter.

    Each subject maintains its own
    timestamp history.
    """

    def __init__(
        self,
        config: RateLimitConfig,
    ):
        self.config = config

        self._events: dict[
            str,
            deque[float],
        ] = {}

    def _now(
        self,
    ) -> float:
        return monotonic()

    def _events_for(
        self,
        subject: str,
    ) -> deque[
        float
    ]:
        if not subject.strip():
            raise ValueError(
                "Rate-limit subject "
                "cannot be empty."
            )

        return self._events.setdefault(
            subject,
            deque(),
        )

    def _prune(
        self,
        events: deque[
            float
        ],
        now: float,
    ) -> None:
        cutoff = (
            now
            - self.config
            .window_seconds
        )

        while (
            events
            and events[0] <= cutoff
        ):
            events.popleft()

    def current_count(
        self,
        subject: str,
    ) -> int:
        events = self._events_for(
            subject
        )

        now = self._now()

        self._prune(
            events,
            now,
        )

        return len(
            events
        )

    def remaining(
        self,
        subject: str,
    ) -> int:
        used = self.current_count(
            subject
        )

        return max(
            0,
            self.config.max_requests
            - used,
        )

    def allowed(
        self,
        subject: str,
    ) -> bool:
        return (
            self.current_count(
                subject
            )
            < self.config
            .max_requests
        )

    def acquire(
        self,
        subject: str,
    ) -> None:
        events = self._events_for(
            subject
        )

        now = self._now()

        self._prune(
            events,
            now,
        )

        if (
            len(events)
            >= self.config
            .max_requests
        ):
            raise RateLimitExceeded(
                "Rate limit exceeded for "
                f"'{subject}': "
                f"maximum "
                f"{self.config.max_requests} "
                "requests per "
                f"{self.config.window_seconds} "
                "seconds."
            )

        events.append(
            now
        )

    def reset(
        self,
        subject: str,
    ) -> int:
        events = self._events.pop(
            subject,
            None,
        )

        if events is None:
            return 0

        return len(
            events
        )


class ConcurrencyGuard:
    """
    Tracks active executions and
    enforces deterministic concurrency
    limits globally and per subject.
    """

    def __init__(
        self,
        *,
        max_concurrent: int,
        max_per_subject: Optional[
            int
        ] = None,
    ):
        if max_concurrent <= 0:
            raise ValueError(
                "max_concurrent must be "
                "greater than zero."
            )

        if (
            max_per_subject
            is not None
            and max_per_subject <= 0
        ):
            raise ValueError(
                "max_per_subject must be "
                "greater than zero."
            )

        self.max_concurrent = (
            max_concurrent
        )

        self.max_per_subject = (
            max_per_subject
        )

        self._active_total = 0

        self._active_by_subject: dict[
            str,
            int,
        ] = {}

    def active_total(
        self,
    ) -> int:
        return self._active_total

    def active_for(
        self,
        subject: str,
    ) -> int:
        return self._active_by_subject.get(
            subject,
            0,
        )

    def available(
        self,
        subject: str,
    ) -> bool:
        if (
            self._active_total
            >= self.max_concurrent
        ):
            return False

        if (
            self.max_per_subject
            is not None
            and self.active_for(
                subject
            )
            >= self.max_per_subject
        ):
            return False

        return True

    def acquire(
        self,
        subject: str,
    ) -> None:
        if not subject.strip():
            raise ValueError(
                "Concurrency subject "
                "cannot be empty."
            )

        if (
            self._active_total
            >= self.max_concurrent
        ):
            raise ConcurrencyLimitExceeded(
                "Global concurrency limit "
                "exceeded."
            )

        active_for_subject = (
            self.active_for(
                subject
            )
        )

        if (
            self.max_per_subject
            is not None
            and active_for_subject
            >= self.max_per_subject
        ):
            raise ConcurrencyLimitExceeded(
                "Concurrency limit exceeded "
                f"for '{subject}'."
            )

        self._active_total += 1

        self._active_by_subject[
            subject
        ] = (
            active_for_subject
            + 1
        )

    def release(
        self,
        subject: str,
    ) -> None:
        active = self.active_for(
            subject
        )

        if active <= 0:
            raise GovernanceError(
                "Cannot release concurrency "
                "slot that is not active "
                f"for '{subject}'."
            )

        self._active_total -= 1

        if active == 1:
            del self._active_by_subject[
                subject
            ]

        else:
            self._active_by_subject[
                subject
            ] = (
                active - 1
            )

    def reset(
        self,
    ) -> None:
        self._active_total = 0
        self._active_by_subject.clear()
