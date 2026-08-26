import pytest

from app.memory.retriever import MemoryRetriever
from app.memory.store import MemoryStore


def build_store(
    tmp_path,
):
    return MemoryStore(
        db_path=str(
            tmp_path
            / "memory.db"
        )
    )


def seed_memories(
    store,
):
    store.save(
        run_id="run-1",
        memory_type="failure",
        key="fastapi_import_failure",
        value={
            "error": (
                "FastAPI import failed because "
                "the dependency was missing."
            ),
            "task": "Run API tests",
        },
    )

    store.save(
        run_id="run-1",
        memory_type="repair",
        key="fastapi_dependency_fix",
        value={
            "root_cause": (
                "FastAPI dependency was missing."
            ),
            "patches": [
                {
                    "path": "requirements.txt",
                    "reason": (
                        "Add fastapi dependency."
                    ),
                }
            ],
        },
        metadata={
            "confidence": 0.95
        },
    )

    store.save(
        run_id="run-2",
        memory_type="critic",
        key="api_quality_feedback",
        value={
            "verdict": "revise",
            "summary": (
                "API implementation needs "
                "better validation."
            ),
            "required_improvements": [
                "Add input validation."
            ],
        },
    )

    store.save(
        run_id="run-2",
        memory_type="security",
        key="api_security_review",
        value={
            "risk_score": 60,
            "summary": (
                "API lacks sufficient "
                "input validation."
            ),
        },
    )

    store.save(
        run_id="run-3",
        memory_type="artifact",
        key="architecture",
        value={
            "architecture_style": (
                "Modular FastAPI architecture"
            ),
            "technology_stack": [
                "FastAPI",
                "Python",
            ],
        },
    )

    store.save(
        run_id="run-4",
        memory_type="task_event",
        key="generic_event",
        value={
            "title": "Unrelated task",
            "status": "completed",
        },
    )


def test_retriever_returns_relevant_memory(
    tmp_path,
):
    store = build_store(
        tmp_path
    )

    seed_memories(
        store
    )

    retriever = MemoryRetriever(
        store
    )

    results = retriever.retrieve(
        "FastAPI dependency failure",
        limit=5,
    )

    assert len(results) > 0

    returned_types = {
        result.memory["memory_type"]
        for result in results
    }

    assert (
        "repair" in returned_types
        or "failure" in returned_types
    )


def test_repair_is_weighted_highly(
    tmp_path,
):
    store = build_store(
        tmp_path
    )

    seed_memories(
        store
    )

    retriever = MemoryRetriever(
        store
    )

    results = retriever.retrieve(
        "FastAPI dependency missing",
        limit=5,
    )

    assert len(results) > 0

    assert (
        results[0].memory[
            "memory_type"
        ]
        == "repair"
    )


def test_retriever_returns_scores(
    tmp_path,
):
    store = build_store(
        tmp_path
    )

    seed_memories(
        store
    )

    retriever = MemoryRetriever(
        store
    )

    results = retriever.retrieve(
        "FastAPI API dependency",
        limit=5,
    )

    assert len(results) > 0

    for result in results:
        assert result.score > 0


def test_retriever_orders_by_score(
    tmp_path,
):
    store = build_store(
        tmp_path
    )

    seed_memories(
        store
    )

    retriever = MemoryRetriever(
        store
    )

    results = retriever.retrieve(
        "FastAPI dependency API",
        limit=5,
    )

    scores = [
        result.score
        for result in results
    ]

    assert scores == sorted(
        scores,
        reverse=True,
    )


def test_retriever_respects_limit(
    tmp_path,
):
    store = build_store(
        tmp_path
    )

    seed_memories(
        store
    )

    retriever = MemoryRetriever(
        store
    )

    results = retriever.retrieve(
        "API FastAPI validation",
        limit=2,
    )

    assert len(results) <= 2


def test_retriever_filters_memory_types(
    tmp_path,
):
    store = build_store(
        tmp_path
    )

    seed_memories(
        store
    )

    retriever = MemoryRetriever(
        store
    )

    results = retriever.retrieve(
        "API validation",
        memory_types=[
            "critic",
            "security",
        ],
        limit=10,
    )

    assert len(results) > 0

    assert all(
        result.memory["memory_type"]
        in {
            "critic",
            "security",
        }
        for result in results
    )


def test_retriever_excludes_current_run(
    tmp_path,
):
    store = build_store(
        tmp_path
    )

    seed_memories(
        store
    )

    retriever = MemoryRetriever(
        store
    )

    results = retriever.retrieve(
        "FastAPI dependency",
        exclude_run_id="run-1",
        limit=10,
    )

    assert all(
        result.memory["run_id"]
        != "run-1"
        for result in results
    )


def test_retrieve_failures_only(
    tmp_path,
):
    store = build_store(
        tmp_path
    )

    seed_memories(
        store
    )

    retriever = MemoryRetriever(
        store
    )

    results = (
        retriever.retrieve_failures(
            "FastAPI failure",
            limit=5,
        )
    )

    assert len(results) > 0

    assert all(
        result.memory[
            "memory_type"
        ]
        == "failure"
        for result in results
    )


def test_retrieve_repairs_only(
    tmp_path,
):
    store = build_store(
        tmp_path
    )

    seed_memories(
        store
    )

    retriever = MemoryRetriever(
        store
    )

    results = (
        retriever.retrieve_repairs(
            "FastAPI dependency",
            limit=5,
        )
    )

    assert len(results) > 0

    assert all(
        result.memory[
            "memory_type"
        ]
        == "repair"
        for result in results
    )


def test_retrieve_critic_feedback_only(
    tmp_path,
):
    store = build_store(
        tmp_path
    )

    seed_memories(
        store
    )

    retriever = MemoryRetriever(
        store
    )

    results = (
        retriever.retrieve_critic_feedback(
            "API validation",
            limit=5,
        )
    )

    assert len(results) > 0

    assert all(
        result.memory[
            "memory_type"
        ]
        == "critic"
        for result in results
    )


def test_retriever_returns_empty_for_unrelated_query(
    tmp_path,
):
    store = build_store(
        tmp_path
    )

    seed_memories(
        store
    )

    retriever = MemoryRetriever(
        store
    )

    results = retriever.retrieve(
        "quantum astrophysics telescope",
        limit=5,
    )

    assert results == []


def test_retriever_returns_empty_when_store_empty(
    tmp_path,
):
    store = build_store(
        tmp_path
    )

    retriever = MemoryRetriever(
        store
    )

    results = retriever.retrieve(
        "FastAPI",
        limit=5,
    )

    assert results == []


def test_retriever_rejects_empty_query(
    tmp_path,
):
    store = build_store(
        tmp_path
    )

    retriever = MemoryRetriever(
        store
    )

    with pytest.raises(
        ValueError,
        match="query cannot be empty",
    ):
        retriever.retrieve(
            "",
            limit=5,
        )


def test_retriever_rejects_invalid_limit(
    tmp_path,
):
    store = build_store(
        tmp_path
    )

    retriever = MemoryRetriever(
        store
    )

    with pytest.raises(
        ValueError,
        match="limit must be greater than zero",
    ):
        retriever.retrieve(
            "FastAPI",
            limit=0,
        )


def test_custom_type_weights_change_ranking(
    tmp_path,
):
    store = build_store(
        tmp_path
    )

    seed_memories(
        store
    )

    retriever = MemoryRetriever(
        store,
        type_weights={
            "artifact": 5.0,
            "repair": 0.5,
        },
    )

    results = retriever.retrieve(
        "FastAPI architecture",
        limit=5,
    )

    assert len(results) > 0

    assert (
        results[0].memory[
            "memory_type"
        ]
        == "artifact"
    )
