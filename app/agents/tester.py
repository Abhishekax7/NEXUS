from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from app.agents.base import BaseAgent
from app.core.models import (
    AgentRole,
    AgentTask,
    Artifact,
    ArtifactType,
)
from app.core.state import NexusState
from app.tools.executor import CommandExecutor
from app.tools.workspace import WorkspaceWriter


class CommandTestResult(BaseModel):
    command: str
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool
    passed: bool


class TestReport(BaseModel):
    passed: bool

    total_commands: int = Field(
        ge=1
    )

    passed_commands: int = Field(
        ge=0
    )

    failed_commands: int = Field(
        ge=0
    )

    results: list[CommandTestResult] = Field(
        min_length=1
    )

    failed_command_names: list[str]

    summary: str = Field(
        min_length=1
    )


class TestingError(Exception):
    """Raised when generated code cannot be tested."""


class TesterAgent(BaseAgent):
    role = AgentRole.TESTER

    def __init__(
        self,
        workspace_writer: Optional[WorkspaceWriter] = None,
        executor: Optional[CommandExecutor] = None,
    ):
        self.workspace_writer = (
            workspace_writer
            or WorkspaceWriter()
        )

        self.executor = (
            executor
            or CommandExecutor()
        )

    def _get_code_artifact(
        self,
        task: AgentTask,
        state: NexusState,
    ) -> Artifact:
        for artifact_id in task.input_artifact_ids:
            artifact = state.artifacts.get(
                artifact_id
            )

            if (
                artifact
                and artifact.type
                == ArtifactType.CODE
            ):
                return artifact

        for artifact in state.artifacts.values():
            if artifact.type == ArtifactType.CODE:
                return artifact

        raise TestingError(
            "CODE artifact not found."
        )

    def _get_test_commands(
        self,
        code_artifact: Artifact,
    ) -> list[str]:
        commands = code_artifact.content.get(
            "test_commands"
        )

        if not isinstance(commands, list):
            raise TestingError(
                "CODE artifact does not contain "
                "a valid test_commands list."
            )

        commands = [
            command
            for command in commands
            if isinstance(command, str)
            and command.strip()
        ]

        if not commands:
            raise TestingError(
                "No valid test commands were provided."
            )

        return commands

    def _get_workspace(
        self,
        state: NexusState,
    ) -> Path:
        return (
            self.workspace_writer.root
            / state.run_id
        ).resolve()

    def _workspace_needs_materialization(
        self,
        code_artifact: Artifact,
        state: NexusState,
    ) -> bool:
        """
        Materialize generated code only when the run workspace
        does not already contain the generated files.

        This prevents a retest from overwriting debugger patches.
        """

        workspace = self._get_workspace(
            state
        )

        if not workspace.exists():
            return True

        files = code_artifact.content.get(
            "files",
            []
        )

        if not isinstance(files, list):
            return True

        for file_data in files:
            if not isinstance(file_data, dict):
                return True

            relative_path = file_data.get(
                "path"
            )

            if not isinstance(
                relative_path,
                str,
            ):
                return True

            target = (
                workspace
                / relative_path
            )

            if not target.exists():
                return True

        return False

    def execute(
        self,
        task: AgentTask,
        state: NexusState,
    ) -> Artifact:
        code_artifact = self._get_code_artifact(
            task,
            state,
        )

        workspace = self._get_workspace(
            state
        )

        materialized = False
        written_file_count = 0

        if self._workspace_needs_materialization(
            code_artifact,
            state,
        ):
            written_files = (
                self.workspace_writer
                .write_code_artifact(
                    code_artifact,
                    state,
                )
            )

            if not written_files:
                raise TestingError(
                    "No generated files were written."
                )

            written_file_count = len(
                written_files
            )

            materialized = True

        if not workspace.exists():
            raise TestingError(
                "Generated workspace does not exist."
            )

        commands = self._get_test_commands(
            code_artifact
        )

        results = []

        for command in commands:
            execution = self.executor.execute(
                command,
                workspace,
            )

            passed = (
                execution.exit_code == 0
                and not execution.timed_out
            )

            results.append(
                CommandTestResult(
                    command=execution.command,
                    exit_code=execution.exit_code,
                    stdout=execution.stdout,
                    stderr=execution.stderr,
                    timed_out=execution.timed_out,
                    passed=passed,
                )
            )

        passed_commands = sum(
            1
            for result in results
            if result.passed
        )

        failed_commands = (
            len(results)
            - passed_commands
        )

        overall_passed = (
            failed_commands == 0
        )

        failed_command_names = [
            result.command
            for result in results
            if not result.passed
        ]

        if overall_passed:
            summary = (
                "All generated test commands passed."
            )
        else:
            summary = (
                f"{failed_commands} of "
                f"{len(results)} test commands failed."
            )

        report = TestReport(
            passed=overall_passed,
            total_commands=len(results),
            passed_commands=passed_commands,
            failed_commands=failed_commands,
            results=results,
            failed_command_names=failed_command_names,
            summary=summary,
        )

        return Artifact(
            type=ArtifactType.TEST_RESULT,
            name="generated_code_test_report",
            content=report.model_dump(),
            created_by=self.role,
            metadata={
                "workspace": str(
                    workspace
                ),
                "written_file_count":
                    written_file_count,
                "workspace_materialized":
                    materialized,
            },
        )
