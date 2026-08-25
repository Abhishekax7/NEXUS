from pathlib import Path

from app.core.models import (
    AgentRole,
    Artifact,
    ArtifactType,
)
from app.core.repair_loop import (
    RepairLoop,
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

        written_files = []

        for file_data in artifact.content["files"]:
            target = (
                run_directory
                / file_data["path"]
            )

            target.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            target.write_text(
                file_data["content"],
                encoding="utf-8",
            )

            written_files.append(
                target
            )

        return written_files


class StateAwareExecutor:
    def __init__(self):
        self.calls = 0

    def execute(
        self,
        command,
        workspace,
    ):
        self.calls += 1

        app_file = (
            workspace
            / "app.py"
        )

        content = app_file.read_text(
            encoding="utf-8"
        )

        if "return True" in content:
            return ExecutionResult(
                command=command,
                exit_code=0,
                stdout="1 passed",
                stderr="",
                timed_out=False,
            )

        return ExecutionResult(
            command=command,
            exit_code=1,
            stdout="",
            stderr="AssertionError",
            timed_out=False,
        )


class FakeDebugger:
    def execute(
        self,
        task,
        state,
    ):
        return Artifact(
            type=ArtifactType.DEBUG_REPORT,
            name="debug_repair_report",
            content={
                "root_cause": "Incorrect return value",
                "failure_summary": "Expected True",
                "patches": [
                    {
                        "path": "app.py",
                        "new_content": (
                            "def run():\n"
                            "    return True\n"
                        ),
                        "reason": "Fix failing return value",
                    }
                ],
                "retry_test_commands": [
                    "pytest -v"
                ],
                "confidence": 0.95,
                "notes": [
                    "Minimal fix"
                ],
            },
            created_by=AgentRole.DEBUGGER,
        )


class NonFixingDebugger:
    def execute(
        self,
        task,
        state,
    ):
        return Artifact(
            type=ArtifactType.DEBUG_REPORT,
            name="debug_repair_report",
            content={
                "root_cause": "Unknown issue",
                "failure_summary": "Still failing",
                "patches": [
                    {
                        "path": "app.py",
                        "new_content": (
                            "def run():\n"
                            "    return False\n"
                        ),
                        "reason": "Ineffective patch",
                    }
                ],
                "retry_test_commands": [
                    "pytest -v"
                ],
                "confidence": 0.2,
                "notes": [
                    "Likely incomplete"
                ],
            },
            created_by=AgentRole.DEBUGGER,
        )


def create_code_artifact():
    return Artifact(
        type=ArtifactType.CODE,
        name="generated_code_bundle",
        content={
            "project_name": "demo",
            "summary": "Demo project",
            "files": [
                {
                    "path": "app.py",
                    "content": (
                        "def run():\n"
                        "    return False\n"
                    ),
                    "purpose": "Application logic",
                },
                {
                    "path": "tests/test_app.py",
                    "content": (
                        "from app import run\n"
                        "\n"
                        "def test_run():\n"
                        "    assert run() is True\n"
                    ),
                    "purpose": "Tests",
                },
            ],
            "dependencies": [
                "pytest"
            ],
            "run_commands": [
                "python app.py"
            ],
            "test_commands": [
                "pytest -v"
            ],
            "implementation_notes": [
                "Demo"
            ],
        },
        created_by=AgentRole.CODER,
    )


def test_repair_loop_fixes_failing_code(
    tmp_path,
):
    from app.agents.tester import TesterAgent
    from app.tools.patcher import PatchApplicator

    state = NexusState(
        user_request="Build app"
    )

    code_artifact = create_code_artifact()

    state.add_artifact(
        code_artifact
    )

    tester = TesterAgent(
        workspace_writer=FakeWorkspaceWriter(
            tmp_path
        ),
        executor=StateAwareExecutor(),
    )

    loop = RepairLoop(
        tester=tester,
        debugger=FakeDebugger(),
        patcher=PatchApplicator(
            root=str(tmp_path)
        ),
        max_repairs=2,
    )

    result = loop.run(
        state
    )

    assert result.passed is True
    assert result.attempts == 1
    assert len(
        result.debug_artifacts
    ) == 1

    assert (
        result.final_test_artifact.content["passed"]
        is True
    )


def test_repair_loop_stops_after_retry_limit(
    tmp_path,
):
    from app.agents.tester import TesterAgent
    from app.tools.patcher import PatchApplicator

    state = NexusState(
        user_request="Build app"
    )

    code_artifact = create_code_artifact()

    state.add_artifact(
        code_artifact
    )

    tester = TesterAgent(
        workspace_writer=FakeWorkspaceWriter(
            tmp_path
        ),
        executor=StateAwareExecutor(),
    )

    loop = RepairLoop(
        tester=tester,
        debugger=NonFixingDebugger(),
        patcher=PatchApplicator(
            root=str(tmp_path)
        ),
        max_repairs=2,
    )

    result = loop.run(
        state
    )

    assert result.passed is False
    assert result.attempts == 2

    assert len(
        result.debug_artifacts
    ) == 2

    assert (
        result.final_test_artifact.content["passed"]
        is False
    )

