import json

import pytest

from app.evaluation.history import (
    EvaluationHistoryError,
    EvaluationHistoryStore,
)
from app.evaluation.models import (
    EvaluationDimension,
    EvaluationStatus,
    MetricScore,
    WorkflowEvaluation,
)


def build_evaluation(
    run_id: str,
    score: float = 90.0,
):
    metric = MetricScore(
        dimension=(
            EvaluationDimension.TASK_COMPLETION
        ),
        score=score,
        status=(
            EvaluationStatus.PASS
            if score >= 80
            else (
                EvaluationStatus.WARN
                if score >= 60
                else EvaluationStatus.FAIL
            )
        ),
        reason="Synthetic test metric.",
        evidence=[
            "Deterministic test evidence."
        ],
    )

    return WorkflowEvaluation(
        run_id=run_id,
        overall_score=score,
        status=metric.status,
        metrics=[
            metric
        ],
        agent_evaluations=[],
        strengths=[],
        weaknesses=[],
        recommendations=[],
        regression_risk=0.0,
    )


def test_store_starts_empty(
    tmp_path,
):
    store = EvaluationHistoryStore(
        db_path=str(
            tmp_path
            / "evaluations.db"
        )
    )

    assert store.count() == 0

    assert (
        store.get_baseline()
        is None
    )


def test_evaluation_can_be_saved(
    tmp_path,
):
    store = EvaluationHistoryStore(
        db_path=str(
            tmp_path
            / "evaluations.db"
        )
    )

    evaluation = build_evaluation(
        "run-1",
        92.0,
    )

    store.save(
        evaluation
    )

    assert store.count() == 1

    assert store.exists(
        "run-1"
    )


def test_saved_evaluation_can_be_loaded(
    tmp_path,
):
    store = EvaluationHistoryStore(
        db_path=str(
            tmp_path
            / "evaluations.db"
        )
    )

    evaluation = build_evaluation(
        "run-1",
        88.0,
    )

    store.save(
        evaluation
    )

    loaded = store.get(
        "run-1"
    )

    assert loaded is not None

    assert loaded.run_id == "run-1"

    assert (
        loaded.overall_score
        == 88.0
    )


def test_missing_evaluation_returns_none(
    tmp_path,
):
    store = EvaluationHistoryStore(
        db_path=str(
            tmp_path
            / "evaluations.db"
        )
    )

    assert (
        store.get(
            "missing-run"
        )
        is None
    )


def test_saving_same_run_updates_record(
    tmp_path,
):
    store = EvaluationHistoryStore(
        db_path=str(
            tmp_path
            / "evaluations.db"
        )
    )

    store.save(
        build_evaluation(
            "run-1",
            80.0,
        )
    )

    store.save(
        build_evaluation(
            "run-1",
            95.0,
        )
    )

    assert store.count() == 1

    loaded = store.get(
        "run-1"
    )

    assert loaded is not None

    assert (
        loaded.overall_score
        == 95.0
    )


def test_recent_evaluations_are_listed(
    tmp_path,
):
    store = EvaluationHistoryStore(
        db_path=str(
            tmp_path
            / "evaluations.db"
        )
    )

    for index in range(3):
        store.save(
            build_evaluation(
                f"run-{index}",
                80.0 + index,
            )
        )

    recent = store.list_recent(
        limit=3
    )

    assert len(recent) == 3

    run_ids = {
        evaluation.run_id
        for evaluation in recent
    }

    assert run_ids == {
        "run-0",
        "run-1",
        "run-2",
    }


def test_recent_limit_is_respected(
    tmp_path,
):
    store = EvaluationHistoryStore(
        db_path=str(
            tmp_path
            / "evaluations.db"
        )
    )

    for index in range(5):
        store.save(
            build_evaluation(
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
    store = EvaluationHistoryStore(
        db_path=str(
            tmp_path
            / "evaluations.db"
        )
    )

    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        store.list_recent(
            limit=0
        )


def test_baseline_can_be_set(
    tmp_path,
):
    store = EvaluationHistoryStore(
        db_path=str(
            tmp_path
            / "evaluations.db"
        )
    )

    store.save(
        build_evaluation(
            "baseline-run",
            94.0,
        )
    )

    store.set_baseline(
        "baseline-run"
    )

    baseline = (
        store.get_baseline()
    )

    assert baseline is not None

    assert (
        baseline.run_id
        == "baseline-run"
    )


def test_baseline_can_be_changed(
    tmp_path,
):
    store = EvaluationHistoryStore(
        db_path=str(
            tmp_path
            / "evaluations.db"
        )
    )

    store.save(
        build_evaluation(
            "run-a",
            85.0,
        )
    )

    store.save(
        build_evaluation(
            "run-b",
            96.0,
        )
    )

    store.set_baseline(
        "run-a"
    )

    store.set_baseline(
        "run-b"
    )

    baseline = (
        store.get_baseline()
    )

    assert baseline is not None

    assert (
        baseline.run_id
        == "run-b"
    )


def test_missing_run_cannot_be_baseline(
    tmp_path,
):
    store = EvaluationHistoryStore(
        db_path=str(
            tmp_path
            / "evaluations.db"
        )
    )

    with pytest.raises(
        EvaluationHistoryError,
        match="does not exist",
    ):
        store.set_baseline(
            "missing-run"
        )


def test_baseline_can_be_cleared(
    tmp_path,
):
    store = EvaluationHistoryStore(
        db_path=str(
            tmp_path
            / "evaluations.db"
        )
    )

    store.save(
        build_evaluation(
            "run-1"
        )
    )

    store.set_baseline(
        "run-1"
    )

    store.clear_baseline()

    assert (
        store.get_baseline()
        is None
    )


def test_non_baseline_evaluation_can_be_deleted(
    tmp_path,
):
    store = EvaluationHistoryStore(
        db_path=str(
            tmp_path
            / "evaluations.db"
        )
    )

    store.save(
        build_evaluation(
            "run-1"
        )
    )

    deleted = store.delete(
        "run-1"
    )

    assert deleted is True

    assert (
        store.exists(
            "run-1"
        )
        is False
    )


def test_delete_missing_run_returns_false(
    tmp_path,
):
    store = EvaluationHistoryStore(
        db_path=str(
            tmp_path
            / "evaluations.db"
        )
    )

    deleted = store.delete(
        "missing-run"
    )

    assert deleted is False


def test_active_baseline_cannot_be_deleted(
    tmp_path,
):
    store = EvaluationHistoryStore(
        db_path=str(
            tmp_path
            / "evaluations.db"
        )
    )

    store.save(
        build_evaluation(
            "baseline-run"
        )
    )

    store.set_baseline(
        "baseline-run"
    )

    with pytest.raises(
        EvaluationHistoryError,
        match="active baseline",
    ):
        store.delete(
            "baseline-run"
        )


def test_baseline_can_be_deleted_after_clear(
    tmp_path,
):
    store = EvaluationHistoryStore(
        db_path=str(
            tmp_path
            / "evaluations.db"
        )
    )

    store.save(
        build_evaluation(
            "baseline-run"
        )
    )

    store.set_baseline(
        "baseline-run"
    )

    store.clear_baseline()

    deleted = store.delete(
        "baseline-run"
    )

    assert deleted is True


def test_history_persists_across_store_instances(
    tmp_path,
):
    db_path = (
        tmp_path
        / "evaluations.db"
    )

    first = EvaluationHistoryStore(
        db_path=str(
            db_path
        )
    )

    first.save(
        build_evaluation(
            "run-1",
            91.0,
        )
    )

    second = EvaluationHistoryStore(
        db_path=str(
            db_path
        )
    )

    loaded = second.get(
        "run-1"
    )

    assert loaded is not None

    assert (
        loaded.overall_score
        == 91.0
    )


def test_baseline_persists_across_store_instances(
    tmp_path,
):
    db_path = (
        tmp_path
        / "evaluations.db"
    )

    first = EvaluationHistoryStore(
        db_path=str(
            db_path
        )
    )

    first.save(
        build_evaluation(
            "baseline-run"
        )
    )

    first.set_baseline(
        "baseline-run"
    )

    second = EvaluationHistoryStore(
        db_path=str(
            db_path
        )
    )

    baseline = (
        second.get_baseline()
    )

    assert baseline is not None

    assert (
        baseline.run_id
        == "baseline-run"
    )


def test_export_json_contains_evaluations(
    tmp_path,
):
    store = EvaluationHistoryStore(
        db_path=str(
            tmp_path
            / "evaluations.db"
        )
    )

    store.save(
        build_evaluation(
            "run-1",
            90.0,
        )
    )

    exported = json.loads(
        store.export_json()
    )

    assert (
        len(
            exported[
                "evaluations"
            ]
        )
        == 1
    )

    assert (
        exported[
            "evaluations"
        ][0][
            "run_id"
        ]
        == "run-1"
    )


def test_export_json_contains_baseline(
    tmp_path,
):
    store = EvaluationHistoryStore(
        db_path=str(
            tmp_path
            / "evaluations.db"
        )
    )

    store.save(
        build_evaluation(
            "baseline-run"
        )
    )

    store.set_baseline(
        "baseline-run"
    )

    exported = json.loads(
        store.export_json()
    )

    assert (
        exported[
            "baseline_run_id"
        ]
        == "baseline-run"
    )


def test_export_json_has_null_baseline_when_unset(
    tmp_path,
):
    store = EvaluationHistoryStore(
        db_path=str(
            tmp_path
            / "evaluations.db"
        )
    )

    store.save(
        build_evaluation(
            "run-1"
        )
    )

    exported = json.loads(
        store.export_json()
    )

    assert (
        exported[
            "baseline_run_id"
        ]
        is None
    )


def test_database_parent_directory_is_created(
    tmp_path,
):
    db_path = (
        tmp_path
        / "nested"
        / "evaluation"
        / "history.db"
    )

    store = EvaluationHistoryStore(
        db_path=str(
            db_path
        )
    )

    store.save(
        build_evaluation(
            "run-1"
        )
    )

    assert db_path.exists()


def test_evaluation_status_survives_roundtrip(
    tmp_path,
):
    store = EvaluationHistoryStore(
        db_path=str(
            tmp_path
            / "evaluations.db"
        )
    )

    evaluation = build_evaluation(
        "run-1",
        50.0,
    )

    store.save(
        evaluation
    )

    loaded = store.get(
        "run-1"
    )

    assert loaded is not None

    assert (
        loaded.status
        == EvaluationStatus.FAIL
    )


def test_evaluation_metrics_survive_roundtrip(
    tmp_path,
):
    store = EvaluationHistoryStore(
        db_path=str(
            tmp_path
            / "evaluations.db"
        )
    )

    evaluation = build_evaluation(
        "run-1",
        92.0,
    )

    store.save(
        evaluation
    )

    loaded = store.get(
        "run-1"
    )

    assert loaded is not None

    assert len(
        loaded.metrics
    ) == 1

    assert (
        loaded.metrics[0].dimension
        == EvaluationDimension.TASK_COMPLETION
    )

    assert (
        loaded.metrics[0].score
        == 92.0
    )
