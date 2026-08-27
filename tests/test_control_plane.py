from app.agents.registry import (
    AgentRegistry,
)

from app.api.control_plane import (
    NexusControlPlane,
)
from app.api.schemas import (
    RunStatus,
)

from app.checkpointing.service import (
    CheckpointService,
)
from app.checkpointing.store import (
    CheckpointStore,
)

from app.core.engine import (
    NexusEngine,
)
from app.core.models import (
    AgentRole,
    AgentTask,
    Artifact,
    ArtifactType,
    TaskStatus,
)
from app.core.state import (
    NexusState,
)


def build_engine(
    tmp_path,
):
    return NexusEngine(
        registry=AgentRegistry(),
        checkpoint_service=(
            CheckpointService(
                store=CheckpointStore(
                    db_path=str(
                        tmp_path
                        / "checkpoints.db"
                    )
                )
            )
        ),
    )


def build_state():
    state = NexusState(
        user_request=(
            "Build an API."
        )
    )

    task = AgentTask(
        title="Implement API",
        description="Build service.",
        assigned_agent=(
            AgentRole.CODER
        ),
    )

    state.add_task(
        task
    )

    return state, task


def test_state_can_be_registered(
    tmp_path,
):
    plane = NexusControlPlane(
        build_engine(
            tmp_path
        )
    )

    state, _ = build_state()

    plane.register_state(
        state
    )

    loaded = plane.get_state(
        state.run_id
    )

    assert loaded is state


def test_created_run_has_created_status(
    tmp_path,
):
    plane = NexusControlPlane(
        build_engine(
            tmp_path
        )
    )

    state, _ = build_state()

    plane.register_state(
        state
    )

    response = (
        plane.run_response(
            state
        )
    )

    assert (
        response.status
        == RunStatus.CREATED
    )


def test_completed_run_has_completed_status(
    tmp_path,
):
    plane = NexusControlPlane(
        build_engine(
            tmp_path
        )
    )

    state, task = build_state()

    task.status = (
        TaskStatus.COMPLETED
    )

    state.completed = True

    plane.register_state(
        state
    )

    response = (
        plane.run_response(
            state
        )
    )

    assert (
        response.status
        == RunStatus.COMPLETED
    )


def test_run_response_contains_counts(
    tmp_path,
):
    plane = NexusControlPlane(
        build_engine(
            tmp_path
        )
    )

    state, _ = build_state()

    artifact = Artifact(
        type=ArtifactType.CODE,
        name="code",
        content={
            "files": []
        },
        created_by=(
            AgentRole.CODER
        ),
    )

    state.add_artifact(
        artifact
    )

    response = (
        plane.run_response(
            state
        )
    )

    assert response.task_count == 1

    assert (
        response.artifact_count
        == 1
    )


def test_summary_counts_completed_tasks(
    tmp_path,
):
    plane = NexusControlPlane(
        build_engine(
            tmp_path
        )
    )

    state, task = build_state()

    task.status = (
        TaskStatus.COMPLETED
    )

    plane.register_state(
        state
    )

    summary = plane.run_summary(
        state.run_id
    )

    assert (
        summary.completed_task_count
        == 1
    )

    assert (
        summary.failed_task_count
        == 0
    )


def test_checkpointed_state_can_be_discovered(
    tmp_path,
):
    engine = build_engine(
        tmp_path
    )

    plane = NexusControlPlane(
        engine
    )

    state, _ = build_state()

    engine.checkpoint_service.workflow_started(
        state
    )

    restored = plane.get_state(
        state.run_id
    )

    assert (
        restored.run_id
        == state.run_id
    )


def test_recovery_response_for_active_run(
    tmp_path,
):
    engine = build_engine(
        tmp_path
    )

    plane = NexusControlPlane(
        engine
    )

    state, _ = build_state()

    engine.checkpoint_service.workflow_started(
        state
    )

    recovery = plane.recovery(
        state.run_id
    )

    assert (
        recovery.recoverable
        is True
    )

    assert (
        recovery.status
        == "recoverable"
    )


def test_failed_state_reports_failed_status(
    tmp_path,
):
    plane = NexusControlPlane(
        build_engine(
            tmp_path
        )
    )

    state, task = build_state()

    state.failed = True
    task.status = (
        TaskStatus.FAILED
    )

    plane.register_state(
        state
    )

    response = plane.run_response(
        state
    )

    assert (
        response.status
        == RunStatus.FAILED
    )
