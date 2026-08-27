from datetime import timezone

import pytest

from app.events.models import (
    EventFilter,
    EventPage,
    EventSeverity,
    EventSubscription,
    EventType,
    NexusEvent,
)


def build_event():
    return NexusEvent(
        type=(
            EventType.JOB_STARTED
        ),
        run_id="run-1",
        job_id="job-1",
        source="job-manager",
        message="Job started.",
    )


def test_event_has_unique_id():
    first = build_event()

    second = build_event()

    assert (
        first.id
        != second.id
    )


def test_event_has_timezone_aware_timestamp():
    event = build_event()

    assert (
        event.timestamp.tzinfo
        is not None
    )

    assert (
        event.timestamp.utcoffset()
        is not None
    )


def test_default_severity_is_info():
    event = build_event()

    assert (
        event.severity
        == EventSeverity.INFO
    )


def test_event_tracks_run_and_job():
    event = build_event()

    assert (
        event.run_id
        == "run-1"
    )

    assert (
        event.job_id
        == "job-1"
    )


def test_event_payload_defaults_empty():
    first = build_event()
    second = build_event()

    first.payload[
        "status"
    ] = "running"

    assert (
        second.payload
        == {}
    )


def test_event_type_values_are_stable():
    assert (
        EventType.RUN_CREATED.value
        == "run.created"
    )

    assert (
        EventType.JOB_COMPLETED.value
        == "job.completed"
    )

    assert (
        EventType.GOVERNANCE_BLOCKED.value
        == "governance.blocked"
    )


def test_event_requires_source():
    with pytest.raises(
        Exception
    ):
        NexusEvent(
            type=EventType.RUN_CREATED,
            source="",
            message="Created.",
        )


def test_event_requires_message():
    with pytest.raises(
        Exception
    ):
        NexusEvent(
            type=EventType.RUN_CREATED,
            source="control-plane",
            message="",
        )


def test_filter_defaults_empty():
    event_filter = (
        EventFilter()
    )

    assert (
        event_filter.run_id
        is None
    )

    assert (
        event_filter.event_types
        == []
    )

    assert (
        event_filter.severities
        == []
    )


def test_filter_accepts_multiple_types():
    event_filter = EventFilter(
        event_types=[
            EventType.JOB_STARTED,
            EventType.JOB_COMPLETED,
        ]
    )

    assert (
        len(
            event_filter.event_types
        )
        == 2
    )


def test_event_page_tracks_counts():
    event = build_event()

    page = EventPage(
        events=[
            event
        ],
        count=1,
        total=1,
    )

    assert page.count == 1

    assert page.total == 1


def test_subscription_defaults_active():
    subscription = (
        EventSubscription(
            run_id="run-1"
        )
    )

    assert (
        subscription.active
        is True
    )


def test_subscription_has_unique_id():
    first = EventSubscription(
        run_id="run-1"
    )

    second = EventSubscription(
        run_id="run-1"
    )

    assert (
        first.id
        != second.id
    )


def test_subscription_timestamp_is_aware():
    subscription = (
        EventSubscription(
            run_id="run-1"
        )
    )

    assert (
        subscription.created_at
        .tzinfo
        is not None
    )
