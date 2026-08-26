import pytest

from app.memory.store import (
    MemoryStore,
    MemoryStoreError,
)


def test_memory_store_saves_and_counts(
    tmp_path,
):
    db_path = (
        tmp_path
        / "memory.db"
    )

    store = MemoryStore(
        db_path=str(db_path)
    )

    memory_id = store.save(
        run_id="run-1",
        memory_type="episodic",
        key="failure",
        value={
            "message": "Test failed"
        },
    )

    assert memory_id > 0
    assert store.count() == 1


def test_memory_store_retrieves_by_run(
    tmp_path,
):
    db_path = (
        tmp_path
        / "memory.db"
    )

    store = MemoryStore(
        db_path=str(db_path)
    )

    store.save(
        run_id="run-1",
        memory_type="episodic",
        key="event",
        value={
            "step": 1
        },
    )

    store.save(
        run_id="run-1",
        memory_type="episodic",
        key="event",
        value={
            "step": 2
        },
    )

    store.save(
        run_id="run-2",
        memory_type="episodic",
        key="event",
        value={
            "step": 3
        },
    )

    memories = store.get_by_run(
        "run-1"
    )

    assert len(memories) == 2

    assert (
        memories[0]["value"]["step"]
        == 1
    )

    assert (
        memories[1]["value"]["step"]
        == 2
    )


def test_memory_store_retrieves_by_type(
    tmp_path,
):
    db_path = (
        tmp_path
        / "memory.db"
    )

    store = MemoryStore(
        db_path=str(db_path)
    )

    store.save(
        run_id="run-1",
        memory_type="repair",
        key="patch",
        value={
            "file": "app.py"
        },
    )

    store.save(
        run_id="run-2",
        memory_type="architecture",
        key="decision",
        value={
            "style": "modular"
        },
    )

    store.save(
        run_id="run-3",
        memory_type="repair",
        key="patch",
        value={
            "file": "service.py"
        },
    )

    memories = store.get_by_type(
        "repair"
    )

    assert len(memories) == 2

    assert all(
        memory["memory_type"]
        == "repair"
        for memory in memories
    )


def test_memory_store_returns_latest(
    tmp_path,
):
    db_path = (
        tmp_path
        / "memory.db"
    )

    store = MemoryStore(
        db_path=str(db_path)
    )

    store.save(
        run_id="run-1",
        memory_type="decision",
        key="architecture_style",
        value="layered",
    )

    store.save(
        run_id="run-2",
        memory_type="decision",
        key="architecture_style",
        value="event-driven",
    )

    latest = store.get_latest(
        key="architecture_style"
    )

    assert latest is not None

    assert (
        latest["value"]
        == "event-driven"
    )


def test_memory_store_filters_latest_by_type(
    tmp_path,
):
    db_path = (
        tmp_path
        / "memory.db"
    )

    store = MemoryStore(
        db_path=str(db_path)
    )

    store.save(
        run_id="run-1",
        memory_type="architecture",
        key="strategy",
        value="modular",
    )

    store.save(
        run_id="run-2",
        memory_type="repair",
        key="strategy",
        value="retry",
    )

    latest = store.get_latest(
        key="strategy",
        memory_type="architecture",
    )

    assert latest is not None

    assert (
        latest["memory_type"]
        == "architecture"
    )

    assert (
        latest["value"]
        == "modular"
    )


def test_memory_store_preserves_metadata(
    tmp_path,
):
    db_path = (
        tmp_path
        / "memory.db"
    )

    store = MemoryStore(
        db_path=str(db_path)
    )

    store.save(
        run_id="run-1",
        memory_type="repair",
        key="patch",
        value={
            "path": "app.py"
        },
        metadata={
            "confidence": 0.95,
            "agent": "debugger",
        },
    )

    memories = store.get_by_run(
        "run-1"
    )

    assert (
        memories[0]["metadata"][
            "confidence"
        ]
        == 0.95
    )

    assert (
        memories[0]["metadata"][
            "agent"
        ]
        == "debugger"
    )


def test_memory_store_persists_across_instances(
    tmp_path,
):
    db_path = (
        tmp_path
        / "memory.db"
    )

    first_store = MemoryStore(
        db_path=str(db_path)
    )

    first_store.save(
        run_id="run-1",
        memory_type="episodic",
        key="event",
        value={
            "status": "completed"
        },
    )

    second_store = MemoryStore(
        db_path=str(db_path)
    )

    memories = (
        second_store.get_by_run(
            "run-1"
        )
    )

    assert len(memories) == 1

    assert (
        memories[0]["value"][
            "status"
        ]
        == "completed"
    )


def test_memory_store_returns_none_for_missing_latest(
    tmp_path,
):
    db_path = (
        tmp_path
        / "memory.db"
    )

    store = MemoryStore(
        db_path=str(db_path)
    )

    result = store.get_latest(
        key="missing"
    )

    assert result is None


def test_memory_store_rejects_empty_run_id(
    tmp_path,
):
    db_path = (
        tmp_path
        / "memory.db"
    )

    store = MemoryStore(
        db_path=str(db_path)
    )

    with pytest.raises(
        MemoryStoreError,
        match="run_id cannot be empty",
    ):
        store.save(
            run_id="",
            memory_type="episodic",
            key="event",
            value={},
        )


def test_memory_store_rejects_empty_memory_type(
    tmp_path,
):
    db_path = (
        tmp_path
        / "memory.db"
    )

    store = MemoryStore(
        db_path=str(db_path)
    )

    with pytest.raises(
        MemoryStoreError,
        match="memory_type cannot be empty",
    ):
        store.save(
            run_id="run-1",
            memory_type="",
            key="event",
            value={},
        )


def test_memory_store_rejects_empty_key(
    tmp_path,
):
    db_path = (
        tmp_path
        / "memory.db"
    )

    store = MemoryStore(
        db_path=str(db_path)
    )

    with pytest.raises(
        MemoryStoreError,
        match="key cannot be empty",
    ):
        store.save(
            run_id="run-1",
            memory_type="episodic",
            key="",
            value={},
        )


def test_memory_store_rejects_non_serializable_value(
    tmp_path,
):
    db_path = (
        tmp_path
        / "memory.db"
    )

    store = MemoryStore(
        db_path=str(db_path)
    )

    with pytest.raises(
        MemoryStoreError,
        match="not JSON serializable",
    ):
        store.save(
            run_id="run-1",
            memory_type="episodic",
            key="bad-value",
            value={
                "bad": object()
            },
        )


def test_memory_store_rejects_invalid_limit(
    tmp_path,
):
    db_path = (
        tmp_path
        / "memory.db"
    )

    store = MemoryStore(
        db_path=str(db_path)
    )

    with pytest.raises(
        MemoryStoreError,
        match="limit must be greater than zero",
    ):
        store.get_by_type(
            "episodic",
            limit=0,
        )
