import pytest

from app.governance.limits import (
    ConcurrencyGuard,
    ConcurrencyLimitExceeded,
    RateLimitConfig,
    RateLimitExceeded,
    SlidingWindowRateLimiter,
)
from app.governance.models import (
    GovernanceError,
)


def test_rate_limit_config_validation():
    with pytest.raises(
        ValueError
    ):
        RateLimitConfig(
            max_requests=0,
            window_seconds=10,
        )

    with pytest.raises(
        ValueError
    ):
        RateLimitConfig(
            max_requests=1,
            window_seconds=0,
        )


def test_rate_limiter_initially_allows():
    limiter = (
        SlidingWindowRateLimiter(
            RateLimitConfig(
                max_requests=2,
                window_seconds=60,
            )
        )
    )

    assert (
        limiter.allowed(
            "run-1"
        )
        is True
    )


def test_rate_limiter_tracks_usage():
    limiter = (
        SlidingWindowRateLimiter(
            RateLimitConfig(
                max_requests=2,
                window_seconds=60,
            )
        )
    )

    limiter.acquire(
        "run-1"
    )

    assert (
        limiter.current_count(
            "run-1"
        )
        == 1
    )

    assert (
        limiter.remaining(
            "run-1"
        )
        == 1
    )


def test_rate_limit_is_enforced():
    limiter = (
        SlidingWindowRateLimiter(
            RateLimitConfig(
                max_requests=2,
                window_seconds=60,
            )
        )
    )

    limiter.acquire(
        "run-1"
    )

    limiter.acquire(
        "run-1"
    )

    with pytest.raises(
        RateLimitExceeded,
        match="Rate limit exceeded",
    ):
        limiter.acquire(
            "run-1"
        )


def test_rate_limits_are_isolated_by_subject():
    limiter = (
        SlidingWindowRateLimiter(
            RateLimitConfig(
                max_requests=1,
                window_seconds=60,
            )
        )
    )

    limiter.acquire(
        "run-a"
    )

    assert (
        limiter.allowed(
            "run-b"
        )
        is True
    )


def test_rate_limiter_reset():
    limiter = (
        SlidingWindowRateLimiter(
            RateLimitConfig(
                max_requests=2,
                window_seconds=60,
            )
        )
    )

    limiter.acquire(
        "run-1"
    )

    removed = limiter.reset(
        "run-1"
    )

    assert removed == 1

    assert (
        limiter.current_count(
            "run-1"
        )
        == 0
    )


def test_empty_rate_limit_subject_rejected():
    limiter = (
        SlidingWindowRateLimiter(
            RateLimitConfig(
                max_requests=1,
                window_seconds=60,
            )
        )
    )

    with pytest.raises(
        ValueError
    ):
        limiter.acquire(
            "   "
        )


def test_concurrency_guard_starts_empty():
    guard = ConcurrencyGuard(
        max_concurrent=2
    )

    assert (
        guard.active_total()
        == 0
    )


def test_concurrency_acquire_and_release():
    guard = ConcurrencyGuard(
        max_concurrent=2
    )

    guard.acquire(
        "run-1"
    )

    assert (
        guard.active_total()
        == 1
    )

    assert (
        guard.active_for(
            "run-1"
        )
        == 1
    )

    guard.release(
        "run-1"
    )

    assert (
        guard.active_total()
        == 0
    )


def test_global_concurrency_limit_enforced():
    guard = ConcurrencyGuard(
        max_concurrent=1
    )

    guard.acquire(
        "run-a"
    )

    with pytest.raises(
        ConcurrencyLimitExceeded,
        match="Global concurrency",
    ):
        guard.acquire(
            "run-b"
        )


def test_per_subject_concurrency_limit_enforced():
    guard = ConcurrencyGuard(
        max_concurrent=5,
        max_per_subject=1,
    )

    guard.acquire(
        "run-a"
    )

    with pytest.raises(
        ConcurrencyLimitExceeded,
        match="run-a",
    ):
        guard.acquire(
            "run-a"
        )


def test_different_subjects_can_execute():
    guard = ConcurrencyGuard(
        max_concurrent=2,
        max_per_subject=1,
    )

    guard.acquire(
        "run-a"
    )

    guard.acquire(
        "run-b"
    )

    assert (
        guard.active_total()
        == 2
    )


def test_available_reports_capacity():
    guard = ConcurrencyGuard(
        max_concurrent=2,
        max_per_subject=1,
    )

    assert (
        guard.available(
            "run-a"
        )
        is True
    )

    guard.acquire(
        "run-a"
    )

    assert (
        guard.available(
            "run-a"
        )
        is False
    )

    assert (
        guard.available(
            "run-b"
        )
        is True
    )


def test_invalid_release_rejected():
    guard = ConcurrencyGuard(
        max_concurrent=2
    )

    with pytest.raises(
        GovernanceError,
        match="Cannot release",
    ):
        guard.release(
            "run-1"
        )


def test_reset_clears_concurrency():
    guard = ConcurrencyGuard(
        max_concurrent=3
    )

    guard.acquire(
        "run-a"
    )

    guard.acquire(
        "run-b"
    )

    guard.reset()

    assert (
        guard.active_total()
        == 0
    )

    assert (
        guard.active_for(
            "run-a"
        )
        == 0
    )


def test_invalid_concurrency_configuration():
    with pytest.raises(
        ValueError
    ):
        ConcurrencyGuard(
            max_concurrent=0
        )

    with pytest.raises(
        ValueError
    ):
        ConcurrencyGuard(
            max_concurrent=2,
            max_per_subject=0,
        )
