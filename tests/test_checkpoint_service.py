from app.checkpointing.models import (
    CheckpointStatus,
    CheckpointType,
    RecoveryStatus,
)
from app.checkpointing.service import (
    CheckpointService,
)
from app.checkpointing.store import (
    CheckpointStore,
)
from app.core.models import (
    AgentRole,
    AgentTask,
    Artifact,
    ArtifactType,
    TaskStatus,
)
from app.core.state import NexusState


def build_service(
    tmp_path,
):
    return CheckpointService(
        store=CheckpointStore(
            db_path=str(
                tmp_path
                / "checkpoints.db"
            )
        )
    )


def build_state():
    state = NexusState(
        user_request=(
            "Build a recoverable "
            "NEXUS workflow."
        )
    )

    task = AgentTask(
        title="Implement application",
        description="Generate code.",
        assigned_agent=(
            AgentRole.CODER
        ),
    )

    state.add_task(
        task
    )

    artifact = Artifact(
        type=ArtifactType.CODE,
        name="code",
        content={
            "files": [
                {
                    "path": "app.py",
                    "content": "print('ok')",
                }
            ]
        },
        created_by=(
            AgentRole.CODER
        ),
    )

    state.add_artifact(
        artifact
    )

    return state, task, artifact


def test_workflow_started_creates_checkpoint(
    tmp_path,
):
    service = build_service(
        tmp_path
    )

    state, _, _ = build_state()

    checkpoint = (
        service.workflow_started(
            state
        )
    )

    assert (
        checkpoint.checkpoint_type
        == CheckpointType.WORKFLOW_STARTED
    )

    assert checkpoint.sequence == 0

    assert (
        checkpoint.status
        == CheckpointStatus.ACTIVE
    )


def test_task_completed_creates_next_checkpoint(
    tmp_path,
):
    service = build_service(
        tmp_path
    )

    state, task, _ = build_state()

    service.workflow_started(
        state
    )

    task.status = (
        TaskStatus.COMPLETED
    )

    checkpoint = (
        service.task_completed(
            state,
            task.id,
        )
    )

    assert checkpoint.sequence == 1

    assert (
        checkpoint.task_id
        == task.id
    )


def test_iteration_checkpoint_records_iteration(
    tmp_path,
):
    service = build_service(
        tmp_path
    )

    state, _, _ = build_state()

    state.iteration = 3

    checkpoint = (
        service.iteration_completed(
            state
        )
    )

    assert (
        checkpoint.metadata[
            "iteration"
        ]
        == 3
    )


def test_repair_checkpoint_records_metadata(
    tmp_path,
):
    service = build_service(
        tmp_path
    )

    state, _, _ = build_state()

    checkpoint = (
        service.repair_completed(
            state,
            attempts=2,
            passed=True,
        )
    )

    assert (
        checkpoint.checkpoint_type
        == CheckpointType.REPAIR_COMPLETED
    )

    assert (
        checkpoint.metadata[
            "attempts"
        ]
        == 2
    )

    assert (
        checkpoint.metadata[
            "passed"
        ]
        is True
    )


def test_replan_checkpoint_records_action(
    tmp_path,
):
    service = build_service(
        tmp_path
    )

    state, _, _ = build_state()

    checkpoint = (
        service.replan_completed(
            state,
            action="insert_task",
        )
    )

    assert (
        checkpoint.metadata[
            "action"
        ]
        == "insert_task"
    )


def test_approval_pending_checkpoint_records_request(
    tmp_path,
):
    service = build_service(
        tmp_path
    )

    state, _, _ = build_state()

    checkpoint = (
        service.approval_pending(
            state,
            approval_request_id=(
                "approval-123"
            ),
        )
    )

    assert (
        checkpoint.checkpoint_type
        == CheckpointType.APPROVAL_PENDING
    )

    assert (
        checkpoint.metadata[
            "approval_request_id"
        ]
        == "approval-123"
    )


def test_completed_workflow_is_not_recoverable(
    tmp_path,
):
    service = build_service(
        tmp_path
    )

    state, _, _ = build_state()

    state.completed = True
    state.failed = False

    service.workflow_completed(
        state
    )

    info = service.recovery_info(
        state.run_id
    )

    assert (
        info.status
        == RecoveryStatus.COMPLETED
    )


def test_failed_workflow_reports_failed_status(
    tmp_path,
):
    service = build_service(
        tmp_path
    )

    state, _, _ = build_state()

    state.failed = True

    service.workflow_failed(
        state,
        reason="Workflow crashed.",
    )

    info = service.recovery_info(
        state.run_id
    )

    assert (
        info.status
        == RecoveryStatus.FAILED
    )


def test_active_checkpoint_is_recoverable(
    tmp_path,
):
    service = build_service(
        tmp_path
    )

    state, _, _ = build_state()

    service.workflow_started(
        state
    )

    info = service.recovery_info(
        state.run_id
    )

    assert (
        info.status
        == RecoveryStatus.RECOVERABLE
    )


def test_missing_run_reports_not_found(
    tmp_path,
):
    service = build_service(
        tmp_path
    )

    info = service.recovery_info(
        "missing-run"
    )

    assert (
        info.status
        == RecoveryStatus.NOT_FOUND
    )


def test_recover_returns_latest_checkpoint(
    tmp_path,
):
    service = build_service(
        tmp_path
    )

    state, task, _ = build_state()

    service.workflow_started(
        state
    )

    task.status = (
        TaskStatus.COMPLETED
    )

    latest = service.task_completed(
        state,
        task.id,
    )

    result = service.recover(
        state.run_id
    )

    assert (
        result.checkpoint
        is not None
    )

    assert (
        result.checkpoint.id
        == latest.id
    )


def test_restore_state_returns_saved_state(
    tmp_path,
):
    service = build_service(
        tmp_path
    )

    state, _, _ = build_state()

    state.iteration = 4

    service.iteration_completed(
        state
    )

    restored = service.restore_state(
        state.run_id
    )

    assert restored is not None

    assert (
        restored.run_id
        == state.run_id
    )

    assert (
        restored.iteration
        == 4
    )


def test_restore_state_preserves_tasks(
    tmp_path,
):
    service = build_service(
        tmp_path
    )

    state, task, _ = build_state()

    service.workflow_started(
        state
    )

    restored = service.restore_state(
        state.run_id
    )

    assert restored is not None

    assert (
        task.id
        in restored.tasks
    )

    assert (
        restored.tasks[
            task.id
        ].title
        == task.title
    )


def test_restore_state_preserves_artifacts(
    tmp_path,
):
    service = build_service(
        tmp_path
    )

    state, _, artifact = build_state()

    service.workflow_started(
        state
    )

    restored = service.restore_state(
        state.run_id
    )

    assert restored is not None

    assert (
        artifact.id
        in restored.artifacts
    )

    assert (
        restored.artifacts[
            artifact.id
        ].content
        == artifact.content
    )


def test_restore_state_preserves_metadata(
    tmp_path,
):
    service = build_service(
        tmp_path
    )

    state, _, _ = build_state()

    state.metadata[
        "replan_count"
    ] = 2

    service.workflow_started(
        state
    )

    restored = service.restore_state(
        state.run_id
    )

    assert restored is not None

    assert (
        restored.metadata[
            "replan_count"
        ]
        == 2
    )


def test_missing_state_restore_returns_none(
    tmp_path,
):
    service = build_service(
        tmp_path
    )

    assert (
        service.restore_state(
            "missing"
        )
        is None
    )


def test_sequences_are_monotonic(
    tmp_path,
):
    service = build_service(
        tmp_path
    )

    state, task, _ = build_state()

    first = service.workflow_started(
        state
    )

    task.status = (
        TaskStatus.COMPLETED
    )

    second = service.task_completed(
        state,
        task.id,
    )

    state.iteration = 1

    third = service.iteration_completed(
        state
    )

    assert [
        first.sequence,
        second.sequence,
        third.sequence,
    ] == [
        0,
        1,
        2,
    ]


def test_checkpoint_state_is_snapshot_not_live_reference(
    tmp_path,
):
    service = build_service(
        tmp_path
    )

    state, _, _ = build_state()

    checkpoint = (
        service.workflow_started(
            state
        )
    )

    state.iteration = 99

    assert (
        checkpoint.state_payload[
            "iteration"
        ]
        != 99
    )


def test_service_persists_between_instances(
    tmp_path,
):
    db_path = (
        tmp_path
        / "checkpoints.db"
    )

    first = CheckpointService(
        store=CheckpointStore(
            db_path=str(
                db_path
            )
        )
    )

    state, _, _ = build_state()

    first.workflow_started(
        state
    )

    second = CheckpointService(
        store=CheckpointStore(
            db_path=str(
                db_path
            )
        )
    )

    restored = (
        second.restore_state(
            state.run_id
        )
    )

    assert restored is not None

    assert (
        restored.run_id
        == state.run_id
    )
