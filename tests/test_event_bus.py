import threading

import pytest

from app.events.bus import (
    EventBus,
)
from app.events.models import (
    EventFilter,
    EventSeverity,
    EventType,
    NexusEvent,
    SubscriptionNotFoundError,
)


def make_event(
    *,
    event_type=(
        EventType.JOB_STARTED
    ),
    run_id="run-1",
    job_id="job-1",
    severity=(
        EventSeverity.INFO
    ),
):
    return NexusEvent(
        type=event_type,
        severity=severity,
        run_id=run_id,
        job_id=job_id,
        source="test",
        message="Test event.",
    )


def test_publish_stores_event():
    bus = EventBus()

    event = make_event()

    bus.publish(
        event
    )

    assert (
        bus.event_count()
        == 1
    )

    page = bus.history()

    assert page.count == 1

    assert (
        page.events[0].id
        == event.id
    )


def test_history_is_bounded():
    bus = EventBus(
        max_history=2
    )

    first = make_event()
    second = make_event()
    third = make_event()

    bus.publish(first)
    bus.publish(second)
    bus.publish(third)

    page = bus.history()

    assert page.count == 2

    assert [
        event.id
        for event in page.events
    ] == [
        second.id,
        third.id,
    ]


def test_history_filters_run():
    bus = EventBus()

    bus.publish(
        make_event(
            run_id="run-1"
        )
    )

    bus.publish(
        make_event(
            run_id="run-2"
        )
    )

    page = bus.history(
        EventFilter(
            run_id="run-2"
        )
    )

    assert page.count == 1

    assert (
        page.events[0].run_id
        == "run-2"
    )


def test_history_filters_job():
    bus = EventBus()

    bus.publish(
        make_event(
            job_id="job-1"
        )
    )

    bus.publish(
        make_event(
            job_id="job-2"
        )
    )

    page = bus.history(
        EventFilter(
            job_id="job-2"
        )
    )

    assert page.count == 1

    assert (
        page.events[0].job_id
        == "job-2"
    )


def test_history_filters_type():
    bus = EventBus()

    bus.publish(
        make_event(
            event_type=(
                EventType.JOB_STARTED
            )
        )
    )

    bus.publish(
        make_event(
            event_type=(
                EventType.JOB_COMPLETED
            )
        )
    )

    page = bus.history(
        EventFilter(
            event_types=[
                EventType.JOB_COMPLETED
            ]
        )
    )

    assert page.count == 1

    assert (
        page.events[0].type
        == EventType.JOB_COMPLETED
    )


def test_history_filters_severity():
    bus = EventBus()

    bus.publish(
        make_event(
            severity=(
                EventSeverity.INFO
            )
        )
    )

    bus.publish(
        make_event(
            severity=(
                EventSeverity.ERROR
            )
        )
    )

    page = bus.history(
        EventFilter(
            severities=[
                EventSeverity.ERROR
            ]
        )
    )

    assert page.count == 1

    assert (
        page.events[0].severity
        == EventSeverity.ERROR
    )


def test_history_limit_returns_latest():
    bus = EventBus()

    events = [
        make_event()
        for _ in range(5)
    ]

    for event in events:
        bus.publish(
            event
        )

    page = bus.history(
        limit=2
    )

    assert page.count == 2

    assert page.total == 5

    assert [
        event.id
        for event in page.events
    ] == [
        events[-2].id,
        events[-1].id,
    ]


def test_subscription_receives_event():
    bus = EventBus()

    subscription = bus.subscribe(
        run_id="run-1"
    )

    event = make_event(
        run_id="run-1"
    )

    bus.publish(
        event
    )

    received = bus.get_event(
        subscription.id,
        timeout=0.1,
    )

    assert received is not None

    assert (
        received.id
        == event.id
    )


def test_subscription_ignores_other_run():
    bus = EventBus()

    subscription = bus.subscribe(
        run_id="run-1"
    )

    bus.publish(
        make_event(
            run_id="run-2"
        )
    )

    received = bus.get_event(
        subscription.id,
        timeout=0.01,
    )

    assert received is None


def test_job_subscription_filters_job():
    bus = EventBus()

    subscription = bus.subscribe(
        job_id="job-2"
    )

    bus.publish(
        make_event(
            job_id="job-1"
        )
    )

    expected = make_event(
        job_id="job-2"
    )

    bus.publish(
        expected
    )

    received = bus.get_event(
        subscription.id,
        timeout=0.1,
    )

    assert (
        received.id
        == expected.id
    )


def test_unsubscribe_removes_subscription():
    bus = EventBus()

    subscription = (
        bus.subscribe()
    )

    removed = bus.unsubscribe(
        subscription.id
    )

    assert (
        removed.active
        is False
    )

    assert (
        bus.subscription_count()
        == 0
    )


def test_missing_subscription_raises():
    bus = EventBus()

    with pytest.raises(
        SubscriptionNotFoundError
    ):
        bus.get_event(
            "missing",
            timeout=0,
        )


def test_slow_consumer_drops_oldest():
    bus = EventBus(
        subscriber_queue_size=2
    )

    subscription = (
        bus.subscribe()
    )

    first = make_event()
    second = make_event()
    third = make_event()

    bus.publish(first)
    bus.publish(second)
    bus.publish(third)

    received_one = (
        bus.get_event(
            subscription.id,
            timeout=0.1,
        )
    )

    received_two = (
        bus.get_event(
            subscription.id,
            timeout=0.1,
        )
    )

    assert [
        received_one.id,
        received_two.id,
    ] == [
        second.id,
        third.id,
    ]


def test_clear_only_clears_history():
    bus = EventBus()

    subscription = (
        bus.subscribe()
    )

    bus.publish(
        make_event()
    )

    bus.clear()

    assert (
        bus.event_count()
        == 0
    )

    assert (
        bus.subscription_count()
        == 1
    )

    assert (
        bus.pending_count(
            subscription.id
        )
        == 1
    )


def test_concurrent_publish_is_safe():
    bus = EventBus(
        max_history=500
    )

    def publisher(
        prefix,
    ):
        for index in range(50):
            bus.publish(
                NexusEvent(
                    type=(
                        EventType
                        .TASK_COMPLETED
                    ),
                    run_id=prefix,
                    source="thread",
                    message=(
                        f"{prefix}-{index}"
                    ),
                )
            )

    threads = [
        threading.Thread(
            target=publisher,
            args=(f"run-{index}",),
        )
        for index in range(4)
    ]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    assert (
        bus.event_count()
        == 200
    )


def test_invalid_history_limit_raises():
    bus = EventBus()

    with pytest.raises(
        ValueError
    ):
        bus.history(
            limit=0
        )
