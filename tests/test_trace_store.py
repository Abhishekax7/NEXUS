import pytest

from app.observability.collector import (
    TraceCollector,
)
from app.observability.models import (
    TraceStatus,
)
from app.observability.store import (
    TraceStore,
)


def build_trace(
    run_id: str,
    *,
    completed: bool = True,
):
    collector = TraceCollector(
        run_id
    )

    collector.workflow_started(
        task_count=1
    )

    collector.task_started(
        task_id="task-1",
        agent_role="coder",
    )

    collector.artifact_created(
        artifact_id="artifact-1",
        agent_role="coder",
        artifact_type="code",
    )

    collector.task_completed(
        task_id="task-1",
        agent_role="coder",
        artifact_id="artifact-1",
    )

    if completed:
        collector.workflow_completed()

    return collector.trace


def test_store_starts_empty(
    tmp_path,
):
    store = TraceStore(
        db_path=str(
            tmp_path / "traces.db"
        )
    )

    assert store.count() == 0


def test_trace_can_be_saved(
    tmp_path,
):
    store = TraceStore(
        db_path=str(
            tmp_path / "traces.db"
        )
    )

    trace = build_trace(
        "run-1"
    )

    store.save(
        trace
    )

    assert store.count() == 1
    assert store.exists("run-1")


def test_trace_can_be_loaded(
    tmp_path,
):
    store = TraceStore(
        db_path=str(
            tmp_path / "traces.db"
        )
    )

    store.save(
        build_trace(
            "run-1"
        )
    )

    loaded = store.get(
        "run-1"
    )

    assert loaded is not None
    assert loaded.run_id == "run-1"

    assert (
        loaded.status
        == TraceStatus.COMPLETED
    )

    assert len(
        loaded.events
    ) == 5


def test_missing_trace_returns_none(
    tmp_path,
):
    store = TraceStore(
        db_path=str(
            tmp_path / "traces.db"
        )
    )

    assert (
        store.get("missing")
        is None
    )


def test_same_run_updates_existing_trace(
    tmp_path,
):
    store = TraceStore(
        db_path=str(
            tmp_path / "traces.db"
        )
    )

    collector = TraceCollector(
        "run-1"
    )

    collector.workflow_started(
        task_count=1
    )

    store.save(
        collector.trace
    )

    collector.workflow_completed()

    store.save(
        collector.trace
    )

    assert store.count() == 1

    loaded = store.get(
        "run-1"
    )

    assert loaded is not None

    assert (
        loaded.status
        == TraceStatus.COMPLETED
    )


def test_recent_limit_is_respected(
    tmp_path,
):
    store = TraceStore(
        db_path=str(
            tmp_path / "traces.db"
        )
    )

    for index in range(5):
        store.save(
            build_trace(
                f"run-{index}"
            )
        )

    recent = store.list_recent(
        limit=2
    )

    assert len(recent) == 2


def test_invalid_recent_limit_rejected(
    tmp_path,
):
    store = TraceStore(
        db_path=str(
            tmp_path / "traces.db"
        )
    )

    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        store.list_recent(
            limit=0
        )


def test_trace_can_be_deleted(
    tmp_path,
):
    store = TraceStore(
        db_path=str(
            tmp_path / "traces.db"
        )
    )

    store.save(
        build_trace(
            "run-1"
        )
    )

    assert (
        store.delete("run-1")
        is True
    )

    assert (
        store.exists("run-1")
        is False
    )


def test_delete_missing_trace_returns_false(
    tmp_path,
):
    store = TraceStore(
        db_path=str(
            tmp_path / "traces.db"
        )
    )

    assert (
        store.delete("missing")
        is False
    )


def test_clear_removes_all_traces(
    tmp_path,
):
    store = TraceStore(
        db_path=str(
            tmp_path / "traces.db"
        )
    )

    for index in range(3):
        store.save(
            build_trace(
                f"run-{index}"
            )
        )

    removed = store.clear()

    assert removed == 3
    assert store.count() == 0


def test_trace_persists_between_instances(
    tmp_path,
):
    db_path = (
        tmp_path / "traces.db"
    )

    first = TraceStore(
        db_path=str(
            db_path
        )
    )

    first.save(
        build_trace(
            "run-1"
        )
    )

    second = TraceStore(
        db_path=str(
            db_path
        )
    )

    loaded = second.get(
        "run-1"
    )

    assert loaded is not None
    assert loaded.run_id == "run-1"


def test_summary_is_generated(
    tmp_path,
):
    store = TraceStore(
        db_path=str(
            tmp_path / "traces.db"
        )
    )

    store.save(
        build_trace(
            "run-1"
        )
    )

    summary = store.summary(
        "run-1"
    )

    assert summary is not None

    assert (
        summary.status
        == TraceStatus.COMPLETED
    )

    assert (
        summary.task_count
        == 1
    )

    assert (
        summary.completed_task_count
        == 1
    )

    assert (
        summary.artifact_count
        == 1
    )

    assert (
        summary.agents_used
        == ["coder"]
    )


def test_missing_summary_returns_none(
    tmp_path,
):
    store = TraceStore(
        db_path=str(
            tmp_path / "traces.db"
        )
    )

    assert (
        store.summary("missing")
        is None
    )


def test_parent_directory_is_created(
    tmp_path,
):
    db_path = (
        tmp_path
        / "nested"
        / "observability"
        / "traces.db"
    )

    TraceStore(
        db_path=str(
            db_path
        )
    )

    assert db_path.exists()


def test_event_data_survives_roundtrip(
    tmp_path,
):
    store = TraceStore(
        db_path=str(
            tmp_path / "traces.db"
        )
    )

    trace = build_trace(
        "run-1"
    )

    store.save(
        trace
    )

    loaded = store.get(
        "run-1"
    )

    assert loaded is not None

    task_event = next(
        event
        for event in loaded.events
        if event.task_id == "task-1"
        and event.agent_role == "coder"
    )

    assert (
        task_event.run_id
        == "run-1"
    )
