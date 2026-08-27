from app.observability.models import (
    TraceStatus,
)
from app.observability.service import (
    ObservabilityService,
)
from app.observability.store import (
    TraceStore,
)


def build_service(
    tmp_path,
):
    return ObservabilityService(
        store=TraceStore(
            db_path=str(
                tmp_path
                / "traces.db"
            )
        )
    )


def test_service_starts_run(
    tmp_path,
):
    service = build_service(
        tmp_path
    )

    collector = service.start_run(
        "run-1",
        task_count=3,
    )

    assert (
        collector.run_id
        == "run-1"
    )

    assert (
        collector.trace.status
        == TraceStatus.STARTED
    )

    assert (
        collector.trace.task_count
        == 3
    )


def test_start_run_is_persisted(
    tmp_path,
):
    service = build_service(
        tmp_path
    )

    service.start_run(
        "run-1",
        task_count=2,
    )

    stored = service.get_trace(
        "run-1"
    )

    assert stored is not None

    assert (
        stored.status
        == TraceStatus.STARTED
    )


def test_complete_run_persists_completed_trace(
    tmp_path,
):
    service = build_service(
        tmp_path
    )

    collector = service.start_run(
        "run-1",
        task_count=1,
    )

    collector.task_started(
        task_id="task-1",
        agent_role="coder",
    )

    collector.task_completed(
        task_id="task-1",
        agent_role="coder",
    )

    trace = service.complete_run(
        collector
    )

    assert (
        trace.status
        == TraceStatus.COMPLETED
    )

    stored = service.get_trace(
        "run-1"
    )

    assert stored is not None

    assert (
        stored.status
        == TraceStatus.COMPLETED
    )


def test_fail_run_persists_failed_trace(
    tmp_path,
):
    service = build_service(
        tmp_path
    )

    collector = service.start_run(
        "run-1"
    )

    trace = service.fail_run(
        collector,
        "Workflow crashed.",
    )

    assert (
        trace.status
        == TraceStatus.FAILED
    )

    stored = service.get_trace(
        "run-1"
    )

    assert stored is not None

    assert (
        stored.status
        == TraceStatus.FAILED
    )


def test_save_updates_trace(
    tmp_path,
):
    service = build_service(
        tmp_path
    )

    collector = service.start_run(
        "run-1",
        task_count=1,
    )

    collector.task_started(
        task_id="task-1",
        agent_role="tester",
    )

    service.save(
        collector.trace
    )

    stored = service.get_trace(
        "run-1"
    )

    assert stored is not None

    assert (
        len(stored.events)
        == 2
    )


def test_get_missing_trace_returns_none(
    tmp_path,
):
    service = build_service(
        tmp_path
    )

    assert (
        service.get_trace(
            "missing"
        )
        is None
    )


def test_summary_is_available(
    tmp_path,
):
    service = build_service(
        tmp_path
    )

    collector = service.start_run(
        "run-1",
        task_count=1,
    )

    collector.task_started(
        task_id="task-1",
        agent_role="architect",
    )

    collector.task_completed(
        task_id="task-1",
        agent_role="architect",
    )

    service.complete_run(
        collector
    )

    summary = service.get_summary(
        "run-1"
    )

    assert summary is not None

    assert (
        summary.completed_task_count
        == 1
    )

    assert (
        summary.agents_used
        == ["architect"]
    )


def test_recent_traces_are_exposed(
    tmp_path,
):
    service = build_service(
        tmp_path
    )

    for index in range(3):
        collector = service.start_run(
            f"run-{index}"
        )

        service.complete_run(
            collector
        )

    recent = service.recent_traces(
        limit=2
    )

    assert len(recent) == 2


def test_trace_can_be_deleted(
    tmp_path,
):
    service = build_service(
        tmp_path
    )

    collector = service.start_run(
        "run-1"
    )

    service.complete_run(
        collector
    )

    assert (
        service.delete_trace(
            "run-1"
        )
        is True
    )

    assert (
        service.get_trace(
            "run-1"
        )
        is None
    )


def test_clear_traces_returns_removed_count(
    tmp_path,
):
    service = build_service(
        tmp_path
    )

    for index in range(3):
        service.start_run(
            f"run-{index}"
        )

    removed = service.clear_traces()

    assert removed == 3

    assert (
        service.store.count()
        == 0
    )


def test_service_persists_between_instances(
    tmp_path,
):
    db_path = (
        tmp_path
        / "traces.db"
    )

    first = ObservabilityService(
        store=TraceStore(
            db_path=str(
                db_path
            )
        )
    )

    collector = first.start_run(
        "run-1"
    )

    first.complete_run(
        collector
    )

    second = ObservabilityService(
        store=TraceStore(
            db_path=str(
                db_path
            )
        )
    )

    stored = second.get_trace(
        "run-1"
    )

    assert stored is not None

    assert (
        stored.status
        == TraceStatus.COMPLETED
    )
