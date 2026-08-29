from pathlib import Path

import pytest

from app.agents.tester import (
    TesterAgent as NexusTesterAgent,
    TestingError as NexusTestingError,
)
from app.core.models import (
    AgentRole,
    AgentTask,
    Artifact,
    ArtifactType,
)
from app.core.state import NexusState
from app.tools.executor import ExecutionResult


class FakeWorkspaceWriter:
    def __init__(self, root: Path):
        self.root = root

    def write_code_artifact(
        self,
        artifact,
        state,
    ):
        run_directory = (
            self.root
            / state.run_id
        )

        run_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        app_file = (
            run_directory
            / "app.py"
        )

        app_file.write_text(
            "print('hello')\n",
            encoding="utf-8",
        )

        return [
            app_file
        ]


class EmptyWorkspaceWriter:
    def __init__(self, root: Path):
        self.root = root

    def write_code_artifact(
        self,
        artifact,
        state,
    ):
        run_directory = (
            self.root
            / state.run_id
        )

        run_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        return []


class PassingExecutor:
    def execute(
        self,
        command,
        workspace,
    ):
        return ExecutionResult(
            command=command,
            exit_code=0,
            stdout="1 passed",
            stderr="",
            timed_out=False,
        )


class FailingExecutor:
    def execute(
        self,
        command,
        workspace,
    ):
        return ExecutionResult(
            command=command,
            exit_code=1,
            stdout="",
            stderr="AssertionError",
            timed_out=False,
        )


class TimeoutExecutor:
    def execute(
        self,
        command,
        workspace,
    ):
        return ExecutionResult(
            command=command,
            exit_code=-1,
            stdout="",
            stderr="",
            timed_out=True,
        )


class MixedExecutor:
    def __init__(self):
        self.calls = 0

    def execute(
        self,
        command,
        workspace,
    ):
        self.calls += 1

        if self.calls == 1:
            return ExecutionResult(
                command=command,
                exit_code=0,
                stdout="passed",
                stderr="",
                timed_out=False,
            )

        return ExecutionResult(
            command=command,
            exit_code=1,
            stdout="",
            stderr="failed",
            timed_out=False,
        )


def create_code_artifact(
    test_commands=None,
):
    if test_commands is None:
        test_commands = [
            "pytest -v"
        ]

    return Artifact(
        type=ArtifactType.CODE,
        name="generated_code_bundle",
        content={
            "project_name": "demo",
            "summary": "Demo project",
            "files": [
                {
                    "path": "app.py",
                    "content": "print('hello')\n",
                    "purpose": "Entry point",
                }
            ],
            "dependencies": [
                "pytest",
            ],
            "run_commands": [
                "python app.py",
            ],
            "test_commands": test_commands,
            "implementation_notes": [
                "Test project",
            ],
        },
        created_by=AgentRole.CODER,
    )


def build_state_and_task(
    test_commands=None,
):
    state = NexusState(
        user_request="Build application"
    )

    code_artifact = create_code_artifact(
        test_commands=test_commands
    )

    state.add_artifact(
        code_artifact
    )

    task = AgentTask(
        title="Test implementation",
        description="Run generated tests.",
        assigned_agent=AgentRole.TESTER,
        input_artifact_ids=[
            code_artifact.id
        ],
    )

    return state, task


def test_tester_agent_returns_test_artifact(
    tmp_path,
):
    state, task = build_state_and_task()

    agent = NexusTesterAgent(
        workspace_writer=(
            FakeWorkspaceWriter(
                tmp_path
            )
        ),
        executor=PassingExecutor(),
    )

    artifact = agent.execute(
        task,
        state,
    )

    assert (
        artifact.type
        == ArtifactType.TEST_RESULT
    )

    assert (
        artifact.created_by
        == AgentRole.TESTER
    )

    assert (
        artifact.content["passed"]
        is True
    )

    assert (
        artifact.content["failed_commands"]
        == 0
    )


def test_tester_agent_records_failed_command(
    tmp_path,
):
    state, task = build_state_and_task()

    agent = NexusTesterAgent(
        workspace_writer=(
            FakeWorkspaceWriter(
                tmp_path
            )
        ),
        executor=FailingExecutor(),
    )

    artifact = agent.execute(
        task,
        state,
    )

    assert (
        artifact.content["passed"]
        is False
    )

    assert (
        artifact.content["failed_commands"]
        == 1
    )

    assert (
        artifact.content["results"][0]["exit_code"]
        == 1
    )

    assert (
        "AssertionError"
        in artifact.content["results"][0]["stderr"]
    )


def test_tester_agent_treats_timeout_as_failure(
    tmp_path,
):
    state, task = build_state_and_task()

    agent = NexusTesterAgent(
        workspace_writer=(
            FakeWorkspaceWriter(
                tmp_path
            )
        ),
        executor=TimeoutExecutor(),
    )

    artifact = agent.execute(
        task,
        state,
    )

    assert (
        artifact.content["passed"]
        is False
    )

    assert (
        artifact.content["results"][0]["timed_out"]
        is True
    )


def test_tester_agent_handles_multiple_commands(
    tmp_path,
):
    state, task = build_state_and_task(
        test_commands=[
            "python app.py",
            "pytest -v",
        ]
    )

    agent = NexusTesterAgent(
        workspace_writer=(
            FakeWorkspaceWriter(
                tmp_path
            )
        ),
        executor=MixedExecutor(),
    )

    artifact = agent.execute(
        task,
        state,
    )

    assert (
        artifact.content["total_commands"]
        == 2
    )

    assert (
        artifact.content["passed_commands"]
        == 1
    )

    assert (
        artifact.content["failed_commands"]
        == 1
    )

    assert (
        artifact.content["passed"]
        is False
    )


def test_tester_agent_fails_without_code_artifact(
    tmp_path,
):
    state = NexusState(
        user_request="Build application"
    )

    task = AgentTask(
        title="Test implementation",
        description="Test code.",
        assigned_agent=AgentRole.TESTER,
    )

    agent = NexusTesterAgent(
        workspace_writer=(
            FakeWorkspaceWriter(
                tmp_path
            )
        ),
        executor=PassingExecutor(),
    )

    with pytest.raises(
        NexusTestingError,
        match="CODE artifact not found",
    ):
        agent.execute(
            task,
            state,
        )


def test_tester_agent_fails_without_test_commands(
    tmp_path,
):
    state, task = build_state_and_task(
        test_commands=[]
    )

    agent = NexusTesterAgent(
        workspace_writer=(
            FakeWorkspaceWriter(
                tmp_path
            )
        ),
        executor=PassingExecutor(),
    )

    with pytest.raises(
        NexusTestingError,
        match="No valid test commands",
    ):
        agent.execute(
            task,
            state,
        )


def test_tester_agent_fails_when_no_files_written(
    tmp_path,
):
    state, task = build_state_and_task()

    agent = NexusTesterAgent(
        workspace_writer=(
            EmptyWorkspaceWriter(
                tmp_path
            )
        ),
        executor=PassingExecutor(),
    )

    with pytest.raises(
        NexusTestingError,
        match="No generated files were written",
    ):
        agent.execute(
            task,
            state,
        )


def test_tester_agent_records_workspace_metadata(
    tmp_path,
):
    state, task = build_state_and_task()

    agent = NexusTesterAgent(
        workspace_writer=(
            FakeWorkspaceWriter(
                tmp_path
            )
        ),
        executor=PassingExecutor(),
    )

    artifact = agent.execute(
        task,
        state,
    )

    assert (
        artifact.metadata["written_file_count"]
        == 1
    )

    assert (
        state.run_id
        in artifact.metadata["workspace"]
    )
class PolicyRejectingExecutor:

    def execute(
        self,
        command,
        workspace,
    ):
        from app.tools.executor import (
            ExecutionError,
        )

        raise ExecutionError(
            "Executable is not allowed: mvn"
        )


def test_tester_records_execution_policy_rejection(
    tmp_path,
):
    state, task = build_state_and_task(
        test_commands=[
            "mvn test",
        ]
    )

    agent = NexusTesterAgent(
        workspace_writer=(
            FakeWorkspaceWriter(
                tmp_path
            )
        ),
        executor=(
            PolicyRejectingExecutor()
        ),
    )

    artifact = agent.execute(
        task,
        state,
    )

    assert (
        artifact.type
        == ArtifactType.TEST_RESULT
    )

    assert (
        artifact.content["passed"]
        is False
    )

    assert (
        artifact.content[
            "failed_commands"
        ]
        == 1
    )

    result = (
        artifact.content[
            "results"
        ][0]
    )

    assert result["command"] == (
        "mvn test"
    )

    assert (
        result["exit_code"]
        == -1
    )

    assert (
        result["timed_out"]
        is False
    )

    assert (
        result["passed"]
        is False
    )

    assert (
        "Executable is not allowed: mvn"
        in result["stderr"]
    )

    assert (
        artifact.metadata[
            "execution_policy_rejections"
        ]
        == 1
    )
