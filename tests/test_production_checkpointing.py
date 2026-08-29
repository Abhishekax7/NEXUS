from pathlib import Path

import pytest

from app.agents.base import BaseAgent
from app.agents.registry import AgentRegistry
from app.checkpointing.models import (
    RecoveryStatus,
)
from app.checkpointing.service import (
    CheckpointService,
)
from app.checkpointing.store import (
    CheckpointStore,
)
from app.core.engine import NexusEngine
from app.core.models import (
    AgentRole,
    AgentTask,
    Artifact,
    ArtifactType,
    TaskStatus,
)
from app.core.state import NexusState


class CountingAgent(BaseAgent):
    role = AgentRole.CODER

    def __init__(
        self,
        counter: dict,
    ):
        self.counter = counter

    def execute(
        self,
        task: AgentTask,
        state: NexusState,
    ) -> Artifact:
        self.counter[
            task.id
        ] = (
            self.counter.get(
                task.id,
                0,
            )
            + 1
        )

        return Artifact(
            type=ArtifactType.CODE,
            name=(
                f"artifact_{task.id}"
            ),
            content={
                "task_id": task.id,
                "execution_count": (
                    self.counter[
                        task.id
                    ]
                ),
            },
            created_by=self.role,
        )


class CrashOnSecondAgent(
    BaseAgent
):
    role = AgentRole.TESTER

    def __init__(
        self,
        counter: dict,
    ):
        self.counter = counter

    def execute(
        self,
        task: AgentTask,
        state: NexusState,
    ) -> Artifact:
        self.counter[
            task.id
        ] = (
            self.counter.get(
                task.id,
                0,
            )
            + 1
        )

        raise RuntimeError(
            "Simulated process crash."
        )


class SuccessfulTesterAgent(
    BaseAgent
):
    role = AgentRole.TESTER

    def __init__(
        self,
        counter: dict,
    ):
        self.counter = counter

    def execute(
        self,
        task: AgentTask,
        state: NexusState,
    ) -> Artifact:
        self.counter[
            task.id
        ] = (
            self.counter.get(
                task.id,
                0,
            )
            + 1
        )

        return Artifact(
            type=ArtifactType.TEST_RESULT,
            name="recovered_test_report",
            content={
                "passed": True,
                "task_id": task.id,
            },
            created_by=self.role,
        )


class UnsupportedCommandCodingAgent(
    BaseAgent
):
    """
    Simulates an older CODE artifact that was
    valid when generated but is incompatible
    with the current NEXUS runtime policy.
    """

    role = AgentRole.CODER

    def __init__(
        self,
        counter: dict,
    ):
        self.counter = counter

    def execute(
        self,
        task: AgentTask,
        state: NexusState,
    ) -> Artifact:
        self.counter[
            task.id
        ] = (
            self.counter.get(
                task.id,
                0,
            )
            + 1
        )

        return Artifact(
            type=ArtifactType.CODE,
            name="legacy_code_bundle",
            content={
                "project_name": (
                    "legacy-app"
                ),
                "summary": (
                    "Legacy generated code"
                ),
                "files": [],
                "dependencies": [],
                "run_commands": [
                    "python app.py",
                ],
                "test_commands": [
                    "mvn test",
                ],
                "implementation_notes": [],
            },
            created_by=self.role,
        )


class SupportedCommandCodingAgent(
    BaseAgent
):
    """
    Produces a CODE artifact that remains
    compatible with the current runtime
    command policy.
    """

    role = AgentRole.CODER

    def __init__(
        self,
        counter: dict,
    ):
        self.counter = counter

    def execute(
        self,
        task: AgentTask,
        state: NexusState,
    ) -> Artifact:
        self.counter[
            task.id
        ] = (
            self.counter.get(
                task.id,
                0,
            )
            + 1
        )

        return Artifact(
            type=ArtifactType.CODE,
            name="supported_code_bundle",
            content={
                "project_name": (
                    "supported-app"
                ),
                "summary": (
                    "Supported generated code"
                ),
                "files": [],
                "dependencies": [],
                "run_commands": [
                    "python app.py",
                ],
                "test_commands": [
                    "pytest -q",
                ],
                "implementation_notes": [],
            },
            created_by=self.role,
        )


def build_state():
    state = NexusState(
        user_request=(
            "Build and test a recoverable "
            "application."
        )
    )

    coding_task = AgentTask(
        title="Implement application",
        description=(
            "Generate application code."
        ),
        assigned_agent=(
            AgentRole.CODER
        ),
    )

    testing_task = AgentTask(
        title="Test application",
        description=(
            "Run application tests."
        ),
        assigned_agent=(
            AgentRole.TESTER
        ),
        dependencies=[
            coding_task.id,
        ],
    )

    state.add_task(
        coding_task
    )

    state.add_task(
        testing_task
    )

    return (
        state,
        coding_task,
        testing_task,
    )


def build_checkpoint_service(
    db_path: Path,
):
    return CheckpointService(
        store=CheckpointStore(
            db_path=str(
                db_path
            )
        )
    )


def build_first_engine(
    checkpoint_service,
    counter,
):
    registry = AgentRegistry()

    registry.register(
        AgentRole.CODER,
        lambda: CountingAgent(
            counter
        ),
    )

    registry.register(
        AgentRole.TESTER,
        lambda: CrashOnSecondAgent(
            counter
        ),
    )

    return NexusEngine(
        registry=registry,
        checkpoint_service=(
            checkpoint_service
        ),
        replanner=None,
        repair_loop=None,
        memory_manager=None,
        evaluation_service=None,
        observability_service=None,
    )


def build_recovery_engine(
    checkpoint_service,
    counter,
):
    registry = AgentRegistry()

    registry.register(
        AgentRole.CODER,
        lambda: CountingAgent(
            counter
        ),
    )

    registry.register(
        AgentRole.TESTER,
        lambda: SuccessfulTesterAgent(
            counter
        ),
    )

    return NexusEngine(
        registry=registry,
        checkpoint_service=(
            checkpoint_service
        ),
        replanner=None,
        repair_loop=None,
        memory_manager=None,
        evaluation_service=None,
        observability_service=None,
    )


def build_legacy_runtime_engine(
    checkpoint_service,
    counter,
):
    registry = AgentRegistry()

    registry.register(
        AgentRole.CODER,
        lambda: (
            UnsupportedCommandCodingAgent(
                counter
            )
        ),
    )

    registry.register(
        AgentRole.TESTER,
        lambda: CrashOnSecondAgent(
            counter
        ),
    )

    return NexusEngine(
        registry=registry,
        checkpoint_service=(
            checkpoint_service
        ),
        replanner=None,
        repair_loop=None,
        memory_manager=None,
        evaluation_service=None,
        observability_service=None,
    )


def build_supported_runtime_engine(
    checkpoint_service,
    counter,
):
    registry = AgentRegistry()

    registry.register(
        AgentRole.CODER,
        lambda: (
            SupportedCommandCodingAgent(
                counter
            )
        ),
    )

    registry.register(
        AgentRole.TESTER,
        lambda: CrashOnSecondAgent(
            counter
        ),
    )

    return NexusEngine(
        registry=registry,
        checkpoint_service=(
            checkpoint_service
        ),
        replanner=None,
        repair_loop=None,
        memory_manager=None,
        evaluation_service=None,
        observability_service=None,
    )


def test_completed_task_is_checkpointed_before_later_crash(
    tmp_path,
):
    db_path = (
        tmp_path
        / "recovery.db"
    )

    service = (
        build_checkpoint_service(
            db_path
        )
    )

    counter = {}

    state, coding_task, _ = (
        build_state()
    )

    engine = build_first_engine(
        service,
        counter,
    )

    with pytest.raises(
        RuntimeError,
        match="Simulated process crash",
    ):
        engine.run(
            state
        )

    assert (
        counter[
            coding_task.id
        ]
        == 1
    )

    checkpoints = (
        service.store.list_run(
            state.run_id
        )
    )

    completed_task_checkpoints = [
        checkpoint
        for checkpoint in checkpoints
        if (
            checkpoint.task_id
            == coding_task.id
        )
    ]

    assert (
        len(
            completed_task_checkpoints
        )
        == 1
    )


def test_failed_run_can_be_restored_explicitly(
    tmp_path,
):
    db_path = (
        tmp_path
        / "recovery.db"
    )

    service = (
        build_checkpoint_service(
            db_path
        )
    )

    counter = {}

    state, coding_task, testing_task = (
        build_state()
    )

    first_engine = (
        build_first_engine(
            service,
            counter,
        )
    )

    with pytest.raises(
        RuntimeError
    ):
        first_engine.run(
            state
        )

    fresh_service = (
        build_checkpoint_service(
            db_path
        )
    )

    recovery_engine = (
        build_recovery_engine(
            fresh_service,
            counter,
        )
    )

    info = (
        fresh_service.recovery_info(
            state.run_id
        )
    )

    assert (
        info.status
        == RecoveryStatus.FAILED
    )

    recovered = (
        recovery_engine.restore_run(
            state.run_id,
            allow_failed=True,
        )
    )

    assert (
        recovered.run_id
        == state.run_id
    )

    assert (
        recovered.tasks[
            coding_task.id
        ].status
        == TaskStatus.COMPLETED
    )

    assert (
        recovered.tasks[
            testing_task.id
        ].status
        != TaskStatus.COMPLETED
    )

    assert (
        recovered.failed
        is False
    )


def test_resume_does_not_execute_completed_task_again(
    tmp_path,
):
    db_path = (
        tmp_path
        / "recovery.db"
    )

    service = (
        build_checkpoint_service(
            db_path
        )
    )

    counter = {}

    state, coding_task, testing_task = (
        build_state()
    )

    first_engine = (
        build_first_engine(
            service,
            counter,
        )
    )

    with pytest.raises(
        RuntimeError
    ):
        first_engine.run(
            state
        )

    assert (
        counter[
            coding_task.id
        ]
        == 1
    )

    assert (
        counter[
            testing_task.id
        ]
        == 1
    )

    fresh_service = (
        build_checkpoint_service(
            db_path
        )
    )

    recovery_engine = (
        build_recovery_engine(
            fresh_service,
            counter,
        )
    )

    result = (
        recovery_engine.resume_run(
            state.run_id,
            allow_failed=True,
        )
    )

    assert result.completed is True
    assert result.failed is False

    # Critical recovery proof:
    # completed coding work was NOT
    # executed a second time.
    assert (
        counter[
            coding_task.id
        ]
        == 1
    )

    # Tester crashed once, then was
    # executed once after recovery.
    assert (
        counter[
            testing_task.id
        ]
        == 2
    )

    assert (
        result.tasks[
            coding_task.id
        ].status
        == TaskStatus.COMPLETED
    )

    assert (
        result.tasks[
            testing_task.id
        ].status
        == TaskStatus.COMPLETED
    )


def test_successful_resume_creates_completed_checkpoint(
    tmp_path,
):
    db_path = (
        tmp_path
        / "recovery.db"
    )

    service = (
        build_checkpoint_service(
            db_path
        )
    )

    counter = {}

    state, _, _ = build_state()

    first_engine = (
        build_first_engine(
            service,
            counter,
        )
    )

    with pytest.raises(
        RuntimeError
    ):
        first_engine.run(
            state
        )

    fresh_service = (
        build_checkpoint_service(
            db_path
        )
    )

    recovery_engine = (
        build_recovery_engine(
            fresh_service,
            counter,
        )
    )

    result = (
        recovery_engine.resume_run(
            state.run_id,
            allow_failed=True,
        )
    )

    info = (
        fresh_service.recovery_info(
            result.run_id
        )
    )

    assert (
        info.status
        == RecoveryStatus.COMPLETED
    )


def test_completed_run_cannot_be_resumed(
    tmp_path,
):
    db_path = (
        tmp_path
        / "recovery.db"
    )

    service = (
        build_checkpoint_service(
            db_path
        )
    )

    counter = {}

    state, _, _ = build_state()

    first_engine = (
        build_first_engine(
            service,
            counter,
        )
    )

    with pytest.raises(
        RuntimeError
    ):
        first_engine.run(
            state
        )

    recovery_engine = (
        build_recovery_engine(
            service,
            counter,
        )
    )

    recovery_engine.resume_run(
        state.run_id,
        allow_failed=True,
    )

    with pytest.raises(
        Exception,
        match=(
            "Completed workflow cannot "
            "be resumed"
        ),
    ):
        recovery_engine.resume_run(
            state.run_id
        )


def test_restore_invalidates_stale_code_artifact(
    tmp_path,
):
    db_path = (
        tmp_path
        / "stale_runtime.db"
    )

    service = (
        build_checkpoint_service(
            db_path
        )
    )

    counter = {}

    state, coding_task, testing_task = (
        build_state()
    )

    first_engine = (
        build_legacy_runtime_engine(
            service,
            counter,
        )
    )

    with pytest.raises(
        RuntimeError
    ):
        first_engine.run(
            state
        )

    fresh_service = (
        build_checkpoint_service(
            db_path
        )
    )

    recovery_engine = (
        build_recovery_engine(
            fresh_service,
            counter,
        )
    )

    recovered = (
        recovery_engine.restore_run(
            state.run_id,
            allow_failed=True,
        )
    )

    assert (
        recovered.tasks[
            coding_task.id
        ].status
        == TaskStatus.PENDING
    )

    assert (
        recovered.tasks[
            testing_task.id
        ].status
        == TaskStatus.PENDING
    )

    assert (
        recovered.tasks[
            coding_task.id
        ].output_artifact_ids
        == []
    )

    assert (
        recovered.metadata[
            "recovered_from_checkpoint"
        ][
            "runtime_artifacts_invalidated"
        ]
        is True
    )


def test_restore_preserves_runtime_compatible_code(
    tmp_path,
):
    db_path = (
        tmp_path
        / "supported_runtime.db"
    )

    service = (
        build_checkpoint_service(
            db_path
        )
    )

    counter = {}

    state, coding_task, _ = (
        build_state()
    )

    first_engine = (
        build_supported_runtime_engine(
            service,
            counter,
        )
    )

    with pytest.raises(
        RuntimeError
    ):
        first_engine.run(
            state
        )

    fresh_service = (
        build_checkpoint_service(
            db_path
        )
    )

    recovery_engine = (
        build_recovery_engine(
            fresh_service,
            counter,
        )
    )

    recovered = (
        recovery_engine.restore_run(
            state.run_id,
            allow_failed=True,
        )
    )

    assert (
        recovered.tasks[
            coding_task.id
        ].status
        == TaskStatus.COMPLETED
    )

    assert (
        len(
            recovered.tasks[
                coding_task.id
            ].output_artifact_ids
        )
        == 1
    )

    assert (
        recovered.metadata[
            "recovered_from_checkpoint"
        ].get(
            "runtime_artifacts_invalidated",
            False,
        )
        is False
    )


def test_restore_reopens_completed_failed_repair_path(
    tmp_path,
):
    db_path = (
        tmp_path
        / "repair_recovery.db"
    )

    service = build_checkpoint_service(
        db_path
    )

    state = NexusState(
        user_request="Build recoverable app"
    )

    coding_task = AgentTask(
        title="Implement application",
        description="Generate code.",
        assigned_agent=AgentRole.CODER,
    )

    testing_task = AgentTask(
        title="Test implementation",
        description="Run tests.",
        assigned_agent=AgentRole.TESTER,
        dependencies=[
            coding_task.id,
        ],
    )

    critic_task = AgentTask(
        title="Critique implementation",
        description="Review result.",
        assigned_agent=AgentRole.CRITIC,
        dependencies=[
            testing_task.id,
        ],
    )

    state.add_task(coding_task)
    state.add_task(testing_task)
    state.add_task(critic_task)

    code_artifact = Artifact(
        type=ArtifactType.CODE,
        name="generated_code_bundle",
        content={
            "project_name": "demo",
            "summary": "Demo",
            "files": [],
            "dependencies": [],
            "run_commands": [
                "python app.py"
            ],
            "test_commands": [
                "pytest -q"
            ],
            "implementation_notes": [],
        },
        created_by=AgentRole.CODER,
    )

    failed_test = Artifact(
        type=ArtifactType.TEST_RESULT,
        name="failed_test_report",
        content={
            "passed": False,
        },
        created_by=AgentRole.TESTER,
    )

    critic_artifact = Artifact(
        type=ArtifactType.EVALUATION,
        name="final_quality_gate",
        content={
            "verdict": "revise",
        },
        created_by=AgentRole.CRITIC,
    )

    state.add_artifact(code_artifact)
    state.add_artifact(failed_test)
    state.add_artifact(critic_artifact)

    coding_task.output_artifact_ids = [
        code_artifact.id
    ]
    testing_task.output_artifact_ids = [
        failed_test.id
    ]
    critic_task.output_artifact_ids = [
        critic_artifact.id
    ]

    coding_task.status = TaskStatus.COMPLETED
    testing_task.status = TaskStatus.COMPLETED
    critic_task.status = TaskStatus.COMPLETED

    state.failed = True
    state.completed = False
    state.errors.append(
        "Autonomous repair exhausted its "
        "retry budget after 2 repair attempts."
    )

    class FakeRecoveryInfo:
        status = RecoveryStatus.FAILED

    class FakeCheckpoint:
        id = "repair-checkpoint"
        sequence = 7

        class checkpoint_type:
            value = "workflow_failed"

    class FakeCheckpointService:
        def recovery_info(
            self,
            run_id,
        ):
            return FakeRecoveryInfo()

        def restore_state(
            self,
            run_id,
        ):
            return state.model_copy(
                deep=True
            )

        def latest_checkpoint(
            self,
            run_id,
        ):
            return FakeCheckpoint()

    registry = AgentRegistry()

    recovery_engine = NexusEngine(
        registry=registry,
        checkpoint_service=(
            FakeCheckpointService()
        ),
        replanner=None,
        repair_loop=None,
        memory_manager=None,
        evaluation_service=None,
        observability_service=None,
    )

    recovered = recovery_engine.restore_run(
        state.run_id,
        allow_failed=True,
    )

    assert recovered.failed is False
    assert recovered.completed is False

    assert (
        recovered.tasks[
            coding_task.id
        ].status
        == TaskStatus.COMPLETED
    )

    assert (
        recovered.tasks[
            testing_task.id
        ].status
        == TaskStatus.PENDING
    )

    assert (
        recovered.tasks[
            critic_task.id
        ].status
        == TaskStatus.PENDING
    )

    assert (
        recovered.tasks[
            coding_task.id
        ].output_artifact_ids
        == [code_artifact.id]
    )

    assert (
        recovered.tasks[
            testing_task.id
        ].output_artifact_ids
        == []
    )

    assert (
        recovered.tasks[
            critic_task.id
        ].output_artifact_ids
        == []
    )

    assert (
        code_artifact.id
        in recovered.artifacts
    )

    assert (
        failed_test.id
        not in recovered.artifacts
    )

    assert (
        critic_artifact.id
        not in recovered.artifacts
    )

    metadata = recovered.metadata[
        "recovered_from_checkpoint"
    ]

    assert (
        metadata[
            "repair_failure_reopened"
        ]
        is True
    )

    assert (
        testing_task.id
        in metadata[
            "repair_failure_root_task_ids"
        ]
    )

    assert (
        critic_task.id
        in metadata[
            "repair_failure_reset_task_ids"
        ]
    )
