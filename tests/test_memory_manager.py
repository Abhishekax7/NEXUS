import pytest

from app.core.models import (
    AgentRole,
    AgentTask,
    Artifact,
    ArtifactType,
    TaskStatus,
)
from app.core.state import NexusState
from app.memory.manager import MemoryManager
from app.memory.store import MemoryStore


def build_manager(
    tmp_path,
):
    store = MemoryStore(
        db_path=str(
            tmp_path
            / "memory.db"
        )
    )

    return MemoryManager(
        store=store
    )


def test_memory_manager_records_task_event(
    tmp_path,
):
    manager = build_manager(
        tmp_path
    )

    state = NexusState(
        user_request="Build app"
    )

    task = AgentTask(
        title="Analyze requirements",
        description="Analyze request",
        assigned_agent=AgentRole.REQUIREMENTS,
    )

    manager.record_task_event(
        task,
        state,
    )

    history = manager.get_run_history(
        state
    )

    assert len(history) == 1

    assert (
        history[0]["memory_type"]
        == "task_event"
    )

    assert (
        history[0]["value"]["agent"]
        == AgentRole.REQUIREMENTS.value
    )


def test_memory_manager_records_failed_task(
    tmp_path,
):
    manager = build_manager(
        tmp_path
    )

    state = NexusState(
        user_request="Build app"
    )

    task = AgentTask(
        title="Implement",
        description="Generate code",
        assigned_agent=AgentRole.CODER,
        status=TaskStatus.FAILED,
        error="Generation failed",
    )

    memory_ids = manager.remember_task(
        task,
        state,
    )

    assert len(memory_ids) == 2

    failures = (
        manager.get_recent_failures()
    )

    assert len(failures) == 1

    assert (
        failures[0]["value"]["error"]
        == "Generation failed"
    )


def test_memory_manager_records_normal_task_once(
    tmp_path,
):
    manager = build_manager(
        tmp_path
    )

    state = NexusState(
        user_request="Build app"
    )

    task = AgentTask(
        title="Research",
        description="Research solution",
        assigned_agent=AgentRole.RESEARCH,
        status=TaskStatus.COMPLETED,
    )

    memory_ids = manager.remember_task(
        task,
        state,
    )

    assert len(memory_ids) == 1


def test_memory_manager_records_generic_artifact(
    tmp_path,
):
    manager = build_manager(
        tmp_path
    )

    state = NexusState(
        user_request="Build app"
    )

    artifact = Artifact(
        type=ArtifactType.ARCHITECTURE,
        name="architecture",
        content={
            "style": "modular"
        },
        created_by=AgentRole.ARCHITECT,
    )

    manager.record_artifact(
        artifact,
        state,
    )

    history = manager.get_run_history(
        state
    )

    assert len(history) == 1

    assert (
        history[0]["memory_type"]
        == "artifact"
    )

    assert (
        history[0]["value"]["type"]
        == ArtifactType.ARCHITECTURE.value
    )


def test_memory_manager_records_repair(
    tmp_path,
):
    manager = build_manager(
        tmp_path
    )

    state = NexusState(
        user_request="Build app"
    )

    artifact = Artifact(
        type=ArtifactType.DEBUG_REPORT,
        name="debug_report",
        content={
            "root_cause": "Broken logic",
            "failure_summary": "Test failed",
            "patches": [
                {
                    "path": "app.py",
                    "new_content": "print('fixed')",
                    "reason": "Fix logic",
                }
            ],
            "confidence": 0.9,
        },
        created_by=AgentRole.DEBUGGER,
        metadata={
            "patch_count": 1
        },
    )

    manager.record_repair(
        artifact,
        state,
    )

    repairs = (
        manager.get_recent_repairs()
    )

    assert len(repairs) == 1

    assert (
        repairs[0]["value"][
            "root_cause"
        ]
        == "Broken logic"
    )

    assert (
        repairs[0]["metadata"][
            "patch_count"
        ]
        == 1
    )


def test_memory_manager_rejects_wrong_repair_artifact(
    tmp_path,
):
    manager = build_manager(
        tmp_path
    )

    state = NexusState(
        user_request="Build app"
    )

    artifact = Artifact(
        type=ArtifactType.CODE,
        name="code",
        content={},
        created_by=AgentRole.CODER,
    )

    with pytest.raises(
        ValueError,
        match="DEBUG_REPORT",
    ):
        manager.record_repair(
            artifact,
            state,
        )


def test_memory_manager_records_security_review(
    tmp_path,
):
    manager = build_manager(
        tmp_path
    )

    state = NexusState(
        user_request="Build app"
    )

    artifact = Artifact(
        type=ArtifactType.SECURITY_REPORT,
        name="security",
        content={
            "passed": False,
            "risk_score": 70,
            "summary": "High-risk issue found",
            "findings": [
                {
                    "title": "Unsafe input"
                }
            ],
        },
        created_by=AgentRole.SECURITY,
        metadata={
            "finding_count": 1
        },
    )

    manager.record_security_review(
        artifact,
        state,
    )

    memories = manager.store.get_by_type(
        "security"
    )

    assert len(memories) == 1

    assert (
        memories[0]["value"][
            "risk_score"
        ]
        == 70
    )

    assert (
        memories[0]["metadata"][
            "finding_count"
        ]
        == 1
    )


def test_memory_manager_rejects_wrong_security_artifact(
    tmp_path,
):
    manager = build_manager(
        tmp_path
    )

    state = NexusState(
        user_request="Build app"
    )

    artifact = Artifact(
        type=ArtifactType.CODE,
        name="code",
        content={},
        created_by=AgentRole.CODER,
    )

    with pytest.raises(
        ValueError,
        match="SECURITY_REPORT",
    ):
        manager.record_security_review(
            artifact,
            state,
        )


def test_memory_manager_records_critic_verdict(
    tmp_path,
):
    manager = build_manager(
        tmp_path
    )

    state = NexusState(
        user_request="Build app"
    )

    artifact = Artifact(
        type=ArtifactType.EVALUATION,
        name="critic",
        content={
            "verdict": "revise",
            "quality_score": 72,
            "summary": "Needs improvement",
            "required_improvements": [
                "Improve security"
            ],
        },
        created_by=AgentRole.CRITIC,
        metadata={
            "issue_count": 2
        },
    )

    manager.record_critic_verdict(
        artifact,
        state,
    )

    verdicts = (
        manager.get_recent_critic_verdicts()
    )

    assert len(verdicts) == 1

    assert (
        verdicts[0]["value"][
            "verdict"
        ]
        == "revise"
    )

    assert (
        verdicts[0]["value"][
            "quality_score"
        ]
        == 72
    )


def test_memory_manager_rejects_wrong_critic_artifact(
    tmp_path,
):
    manager = build_manager(
        tmp_path
    )

    state = NexusState(
        user_request="Build app"
    )

    artifact = Artifact(
        type=ArtifactType.CODE,
        name="code",
        content={},
        created_by=AgentRole.CODER,
    )

    with pytest.raises(
        ValueError,
        match="EVALUATION",
    ):
        manager.record_critic_verdict(
            artifact,
            state,
        )


def test_memory_manager_routes_important_artifacts(
    tmp_path,
):
    manager = build_manager(
        tmp_path
    )

    state = NexusState(
        user_request="Build app"
    )

    debug_artifact = Artifact(
        type=ArtifactType.DEBUG_REPORT,
        name="debug",
        content={
            "root_cause": "Bug",
            "failure_summary": "Failure",
            "patches": [],
            "confidence": 0.5,
        },
        created_by=AgentRole.DEBUGGER,
    )

    security_artifact = Artifact(
        type=ArtifactType.SECURITY_REPORT,
        name="security",
        content={
            "passed": True,
            "risk_score": 5,
            "summary": "Safe",
            "findings": [],
        },
        created_by=AgentRole.SECURITY,
    )

    critic_artifact = Artifact(
        type=ArtifactType.EVALUATION,
        name="critic",
        content={
            "verdict": "accept",
            "quality_score": 95,
            "summary": "Good",
            "required_improvements": [],
        },
        created_by=AgentRole.CRITIC,
    )

    manager.record_important_artifact(
        debug_artifact,
        state,
    )

    manager.record_important_artifact(
        security_artifact,
        state,
    )

    manager.record_important_artifact(
        critic_artifact,
        state,
    )

    assert len(
        manager.get_recent_repairs()
    ) == 1

    assert len(
        manager.store.get_by_type(
            "security"
        )
    ) == 1

    assert len(
        manager.get_recent_critic_verdicts()
    ) == 1


def test_memory_manager_gets_run_history(
    tmp_path,
):
    manager = build_manager(
        tmp_path
    )

    state = NexusState(
        user_request="Build app"
    )

    task = AgentTask(
        title="Task",
        description="Do work",
        assigned_agent=AgentRole.REQUIREMENTS,
    )

    manager.record_task_event(
        task,
        state,
    )

    artifact = Artifact(
        type=ArtifactType.REQUIREMENTS,
        name="requirements",
        content={
            "objective": "Build app"
        },
        created_by=AgentRole.REQUIREMENTS,
    )

    manager.record_artifact(
        artifact,
        state,
    )

    history = manager.get_run_history(
        state
    )

    assert len(history) == 2

    assert (
        history[0]["run_id"]
        == state.run_id
    )

    assert (
        history[1]["run_id"]
        == state.run_id
    )
