import pytest

from app.checkpointing.models import (
    CheckpointError,
    CheckpointStatus,
    CheckpointType,
    WorkflowCheckpoint,
)
from app.checkpointing.store import (
    CheckpointStore,
)


def build_checkpoint(
    run_id: str,
    sequence: int,
    checkpoint_type=(
        CheckpointType.TASK_COMPLETED
    ),
):
    return WorkflowCheckpoint(
        run_id=run_id,
        checkpoint_type=(
            checkpoint_type
        ),
        sequence=sequence,
        state_payload={
            "run_id": run_id,
            "iteration": sequence,
        },
        reason=(
            "Deterministic test "
            "checkpoint."
        ),
    )


def test_store_starts_empty(
    tmp_path,
):
    store = CheckpointStore(
        db_path=str(
            tmp_path
            / "checkpoints.db"
        )
    )

    assert store.count() == 0


def test_checkpoint_can_be_saved(
    tmp_path,
):
    store = CheckpointStore(
        db_path=str(
            tmp_path
            / "checkpoints.db"
        )
    )

    checkpoint = (
        build_checkpoint(
            "run-1",
            0,
        )
    )

    store.save(
        checkpoint
    )

    assert store.count() == 1

    assert (
        store.count(
            "run-1"
        )
        == 1
    )


def test_checkpoint_can_be_loaded(
    tmp_path,
):
    store = CheckpointStore(
        db_path=str(
            tmp_path
            / "checkpoints.db"
        )
    )

    checkpoint = (
        build_checkpoint(
            "run-1",
            0,
        )
    )

    store.save(
        checkpoint
    )

    loaded = store.get(
        checkpoint.id
    )

    assert loaded is not None

    assert (
        loaded.id
        == checkpoint.id
    )

    assert (
        loaded.run_id
        == "run-1"
    )


def test_missing_checkpoint_returns_none(
    tmp_path,
):
    store = CheckpointStore(
        db_path=str(
            tmp_path
            / "checkpoints.db"
        )
    )

    assert (
        store.get(
            "missing"
        )
        is None
    )


def test_latest_returns_highest_sequence(
    tmp_path,
):
    store = CheckpointStore(
        db_path=str(
            tmp_path
            / "checkpoints.db"
        )
    )

    for sequence in range(3):
        store.save(
            build_checkpoint(
                "run-1",
                sequence,
            )
        )

    latest = store.latest(
        "run-1"
    )

    assert latest is not None

    assert (
        latest.sequence
        == 2
    )


def test_latest_missing_run_returns_none(
    tmp_path,
):
    store = CheckpointStore(
        db_path=str(
            tmp_path
            / "checkpoints.db"
        )
    )

    assert (
        store.latest(
            "missing"
        )
        is None
    )


def test_run_checkpoints_are_ordered(
    tmp_path,
):
    store = CheckpointStore(
        db_path=str(
            tmp_path
            / "checkpoints.db"
        )
    )

    store.save(
        build_checkpoint(
            "run-1",
            2,
        )
    )

    store.save(
        build_checkpoint(
            "run-1",
            0,
        )
    )

    store.save(
        build_checkpoint(
            "run-1",
            1,
        )
    )

    checkpoints = (
        store.list_run(
            "run-1"
        )
    )

    assert [
        checkpoint.sequence
        for checkpoint
        in checkpoints
    ] == [
        0,
        1,
        2,
    ]


def test_runs_are_isolated(
    tmp_path,
):
    store = CheckpointStore(
        db_path=str(
            tmp_path
            / "checkpoints.db"
        )
    )

    store.save(
        build_checkpoint(
            "run-a",
            0,
        )
    )

    store.save(
        build_checkpoint(
            "run-b",
            0,
        )
    )

    assert (
        store.count(
            "run-a"
        )
        == 1
    )

    assert (
        store.count(
            "run-b"
        )
        == 1
    )

    assert store.count() == 2


def test_next_sequence_starts_at_zero(
    tmp_path,
):
    store = CheckpointStore(
        db_path=str(
            tmp_path
            / "checkpoints.db"
        )
    )

    assert (
        store.next_sequence(
            "run-1"
        )
        == 0
    )


def test_next_sequence_increments_latest(
    tmp_path,
):
    store = CheckpointStore(
        db_path=str(
            tmp_path
            / "checkpoints.db"
        )
    )

    store.save(
        build_checkpoint(
            "run-1",
            0,
        )
    )

    store.save(
        build_checkpoint(
            "run-1",
            1,
        )
    )

    assert (
        store.next_sequence(
            "run-1"
        )
        == 2
    )


def test_duplicate_sequence_is_rejected(
    tmp_path,
):
    store = CheckpointStore(
        db_path=str(
            tmp_path
            / "checkpoints.db"
        )
    )

    store.save(
        build_checkpoint(
            "run-1",
            0,
        )
    )

    with pytest.raises(
        CheckpointError,
        match="sequence already exists",
    ):
        store.save(
            build_checkpoint(
                "run-1",
                0,
            )
        )


def test_same_sequence_allowed_for_different_runs(
    tmp_path,
):
    store = CheckpointStore(
        db_path=str(
            tmp_path
            / "checkpoints.db"
        )
    )

    store.save(
        build_checkpoint(
            "run-a",
            0,
        )
    )

    store.save(
        build_checkpoint(
            "run-b",
            0,
        )
    )

    assert store.count() == 2


def test_checkpoint_status_can_be_updated(
    tmp_path,
):
    store = CheckpointStore(
        db_path=str(
            tmp_path
            / "checkpoints.db"
        )
    )

    checkpoint = (
        build_checkpoint(
            "run-1",
            0,
        )
    )

    store.save(
        checkpoint
    )

    store.mark_status(
        checkpoint.id,
        CheckpointStatus.COMPLETED,
    )

    loaded = store.get(
        checkpoint.id
    )

    assert loaded is not None

    assert (
        loaded.status
        == CheckpointStatus.COMPLETED
    )


def test_missing_checkpoint_status_update_fails(
    tmp_path,
):
    store = CheckpointStore(
        db_path=str(
            tmp_path
            / "checkpoints.db"
        )
    )

    with pytest.raises(
        CheckpointError,
        match="not found",
    ):
        store.mark_status(
            "missing",
            CheckpointStatus.FAILED,
        )


def test_run_can_be_deleted(
    tmp_path,
):
    store = CheckpointStore(
        db_path=str(
            tmp_path
            / "checkpoints.db"
        )
    )

    for sequence in range(3):
        store.save(
            build_checkpoint(
                "run-1",
                sequence,
            )
        )

    removed = store.delete_run(
        "run-1"
    )

    assert removed == 3

    assert (
        store.count(
            "run-1"
        )
        == 0
    )


def test_delete_run_does_not_affect_others(
    tmp_path,
):
    store = CheckpointStore(
        db_path=str(
            tmp_path
            / "checkpoints.db"
        )
    )

    store.save(
        build_checkpoint(
            "run-a",
            0,
        )
    )

    store.save(
        build_checkpoint(
            "run-b",
            0,
        )
    )

    store.delete_run(
        "run-a"
    )

    assert (
        store.count(
            "run-a"
        )
        == 0
    )

    assert (
        store.count(
            "run-b"
        )
        == 1
    )


def test_checkpoint_payload_survives_roundtrip(
    tmp_path,
):
    store = CheckpointStore(
        db_path=str(
            tmp_path
            / "checkpoints.db"
        )
    )

    checkpoint = (
        build_checkpoint(
            "run-1",
            4,
        )
    )

    checkpoint.state_payload[
        "custom"
    ] = {
        "value": 42,
        "items": [
            "a",
            "b",
        ],
    }

    store.save(
        checkpoint
    )

    loaded = store.get(
        checkpoint.id
    )

    assert loaded is not None

    assert (
        loaded.state_payload[
            "custom"
        ][
            "value"
        ]
        == 42
    )


def test_store_persists_between_instances(
    tmp_path,
):
    db_path = (
        tmp_path
        / "checkpoints.db"
    )

    first = CheckpointStore(
        db_path=str(
            db_path
        )
    )

    checkpoint = (
        build_checkpoint(
            "run-1",
            0,
        )
    )

    first.save(
        checkpoint
    )

    second = CheckpointStore(
        db_path=str(
            db_path
        )
    )

    loaded = second.get(
        checkpoint.id
    )

    assert loaded is not None

    assert (
        loaded.run_id
        == "run-1"
    )


def test_parent_directory_is_created(
    tmp_path,
):
    db_path = (
        tmp_path
        / "nested"
        / "checkpointing"
        / "checkpoints.db"
    )

    CheckpointStore(
        db_path=str(
            db_path
        )
    )

    assert db_path.exists()
