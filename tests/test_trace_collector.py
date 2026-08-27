import pytest

from app.observability.collector import (
    TraceCollector,
)
from app.observability.models import (
    TraceEventType,
    TraceStatus,
)


def test_collector_requires_run_id():
    with pytest.raises(
        ValueError,
        match="run_id",
    ):
        TraceCollector("")


def test_collector_uses_run_id():
    collector = TraceCollector(
        "run-123"
    )

    assert (
        collector.run_id
        == "run-123"
    )

    assert (
        collector.trace.run_id
        == "run-123"
    )


def test_workflow_started_records_event():
    collector = TraceCollector(
        "run-1"
    )

    event = (
        collector.workflow_started(
            task_count=5
        )
    )

    assert (
        event.event_type
        == TraceEventType.WORKFLOW_STARTED
    )

    assert (
        event.status
        == TraceStatus.STARTED
    )

    assert (
        collector.trace.task_count
        == 5
    )

    assert len(
        collector.trace.events
    ) == 1


def test_negative_task_count_rejected():
    collector = TraceCollector(
        "run-1"
    )

    with pytest.raises(
        ValueError,
        match="task_count",
    ):
        collector.workflow_started(
            task_count=-1
        )


def test_workflow_completed_updates_trace():
    collector = TraceCollector(
        "run-1"
    )

    collector.workflow_started(
        task_count=1
    )

    event = (
        collector.workflow_completed()
    )

    assert (
        collector.trace.status
        == TraceStatus.COMPLETED
    )

    assert (
        collector.trace.completed_at
        is not None
    )

    assert (
        collector.trace.total_duration_ms
        is not None
    )

    assert (
        event.event_type
        == TraceEventType.WORKFLOW_COMPLETED
    )


def test_workflow_failed_updates_trace():
    collector = TraceCollector(
        "run-1"
    )

    collector.workflow_started()

    event = collector.workflow_failed(
        "Something failed."
    )

    assert (
        collector.trace.status
        == TraceStatus.FAILED
    )

    assert (
        event.message
        == "Something failed."
    )


def test_task_lifecycle_is_recorded():
    collector = TraceCollector(
        "run-1"
    )

    collector.workflow_started(
        task_count=1
    )

    collector.task_started(
        task_id="task-1",
        agent_role="coder",
    )

    event = collector.task_completed(
        task_id="task-1",
        agent_role="coder",
        artifact_id="artifact-1",
    )

    assert (
        event.event_type
        == TraceEventType.TASK_COMPLETED
    )

    assert (
        event.duration_ms
        is not None
    )

    assert (
        collector.trace
        .completed_task_count
        == 1
    )


def test_failed_task_is_counted():
    collector = TraceCollector(
        "run-1"
    )

    collector.task_started(
        task_id="task-1",
        agent_role="tester",
    )

    event = collector.task_failed(
        task_id="task-1",
        agent_role="tester",
        message="Tests failed.",
    )

    assert (
        collector.trace.failed_task_count
        == 1
    )

    assert (
        event.status
        == TraceStatus.FAILED
    )


def test_artifact_creation_is_counted():
    collector = TraceCollector(
        "run-1"
    )

    event = collector.artifact_created(
        artifact_id="artifact-1",
        agent_role="coder",
        artifact_type="code",
    )

    assert (
        collector.trace.artifact_count
        == 1
    )

    assert (
        event.metadata[
            "artifact_type"
        ]
        == "code"
    )


def test_repair_lifecycle_is_recorded():
    collector = TraceCollector(
        "run-1"
    )

    collector.repair_started()

    event = collector.repair_completed(
        passed=True,
        attempts=2,
    )

    assert (
        collector.trace.repair_count
        == 1
    )

    assert (
        event.metadata["attempts"]
        == 2
    )

    assert (
        event.metadata["passed"]
        is True
    )


def test_failed_repair_is_recorded():
    collector = TraceCollector(
        "run-1"
    )

    collector.repair_started()

    event = collector.repair_failed(
        "Repair budget exhausted."
    )

    assert (
        collector.trace.repair_count
        == 1
    )

    assert (
        event.event_type
        == TraceEventType.REPAIR_FAILED
    )


def test_replan_lifecycle_is_recorded():
    collector = TraceCollector(
        "run-1"
    )

    collector.replan_started()

    event = collector.replan_completed(
        action="insert_task"
    )

    assert (
        collector.trace.replan_count
        == 1
    )

    assert (
        event.metadata["action"]
        == "insert_task"
    )


def test_evaluation_event_is_recorded():
    collector = TraceCollector(
        "run-1"
    )

    event = (
        collector.evaluation_completed(
            overall_score=92.5,
            regression_detected=False,
        )
    )

    assert (
        event.event_type
        == TraceEventType.EVALUATION_COMPLETED
    )

    assert (
        event.metadata[
            "overall_score"
        ]
        == 92.5
    )

    assert (
        event.metadata[
            "regression_detected"
        ]
        is False
    )


def test_summary_contains_agents_used():
    collector = TraceCollector(
        "run-1"
    )

    collector.workflow_started(
        task_count=2
    )

    collector.task_started(
        task_id="task-1",
        agent_role="coder",
    )

    collector.task_completed(
        task_id="task-1",
        agent_role="coder",
    )

    collector.task_started(
        task_id="task-2",
        agent_role="tester",
    )

    collector.task_completed(
        task_id="task-2",
        agent_role="tester",
    )

    collector.workflow_completed()

    summary = collector.summary()

    assert summary.agents_used == [
        "coder",
        "tester",
    ]

    assert (
        summary.completed_task_count
        == 2
    )

    assert (
        summary.total_events
        == 6
    )


def test_all_events_use_same_run_id():
    collector = TraceCollector(
        "run-xyz"
    )

    collector.workflow_started()

    collector.task_started(
        task_id="task-1",
        agent_role="architect",
    )

    collector.task_completed(
        task_id="task-1",
        agent_role="architect",
    )

    collector.workflow_completed()

    assert all(
        event.run_id == "run-xyz"
        for event
        in collector.trace.events
    )
