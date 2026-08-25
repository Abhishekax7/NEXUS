import json

import pytest

from app.agents.debugger import (
    DebugGenerationError,
    DebuggerAgent,
)
from app.core.models import (
    AgentRole,
    AgentTask,
    Artifact,
    ArtifactType,
)
from app.core.state import NexusState


class FakeDebuggerLLM:
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool = False,
    ) -> str:
        return json.dumps(
            {
                "root_cause": "Incorrect return value in app.py",
                "failure_summary": (
                    "The generated function returns False "
                    "while the test expects True."
                ),
                "patches": [
                    {
                        "path": "app.py",
                        "new_content": (
                            "def run():\n"
                            "    return True\n"
                        ),
                        "reason": (
                            "Update the implementation to match "
                            "the expected behavior."
                        ),
                    }
                ],
                "retry_test_commands": [
                    "pytest -v",
                ],
                "confidence": 0.95,
                "notes": [
                    "The patch changes only the failing logic.",
                ],
            }
        )


class RepairingDebuggerLLM:
    def __init__(self):
        self.calls = 0

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool = False,
    ) -> str:
        self.calls += 1

        if self.calls == 1:
            return json.dumps(
                {
                    "root_cause": "Broken logic",
                    "failure_summary": "Test failed",
                    "patches": [],
                }
            )

        return FakeDebuggerLLM().generate(
            system_prompt,
            user_prompt,
            json_mode,
        )


class UnknownFileDebuggerLLM:
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool = False,
    ) -> str:
        data = json.loads(
            FakeDebuggerLLM().generate(
                system_prompt,
                user_prompt,
                json_mode,
            )
        )

        data["patches"][0]["path"] = "unknown.py"

        return json.dumps(data)


class TraversalDebuggerLLM:
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool = False,
    ) -> str:
        data = json.loads(
            FakeDebuggerLLM().generate(
                system_prompt,
                user_prompt,
                json_mode,
            )
        )

        data["patches"][0]["path"] = "../app.py"

        return json.dumps(data)


class AbsolutePathDebuggerLLM:
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool = False,
    ) -> str:
        data = json.loads(
            FakeDebuggerLLM().generate(
                system_prompt,
                user_prompt,
                json_mode,
            )
        )

        data["patches"][0]["path"] = "/tmp/app.py"

        return json.dumps(data)


class DuplicatePatchDebuggerLLM:
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool = False,
    ) -> str:
        data = json.loads(
            FakeDebuggerLLM().generate(
                system_prompt,
                user_prompt,
                json_mode,
            )
        )

        data["patches"].append(
            {
                "path": "app.py",
                "new_content": (
                    "def run():\n"
                    "    return True\n"
                ),
                "reason": "Duplicate patch",
            }
        )

        return json.dumps(data)


class AlwaysInvalidDebuggerLLM:
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool = False,
    ) -> str:
        return "{}"


def create_code_artifact() -> Artifact:
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
                    "purpose": "Test suite",
                },
            ],
            "dependencies": [
                "pytest",
            ],
            "run_commands": [
                "python app.py",
            ],
            "test_commands": [
                "pytest -v",
            ],
            "implementation_notes": [
                "Demo code bundle",
            ],
        },
        created_by=AgentRole.CODER,
    )


def create_failed_test_artifact() -> Artifact:
    return Artifact(
        type=ArtifactType.TEST_RESULT,
        name="generated_code_test_report",
        content={
            "passed": False,
            "total_commands": 1,
            "passed_commands": 0,
            "failed_commands": 1,
            "results": [
                {
                    "command": "pytest -v",
                    "exit_code": 1,
                    "stdout": "",
                    "stderr": "AssertionError",
                    "timed_out": False,
                    "passed": False,
                }
            ],
            "failed_command_names": [
                "pytest -v",
            ],
            "summary": "1 of 1 test commands failed.",
        },
        created_by=AgentRole.TESTER,
    )


def create_passing_test_artifact() -> Artifact:
    return Artifact(
        type=ArtifactType.TEST_RESULT,
        name="generated_code_test_report",
        content={
            "passed": True,
            "total_commands": 1,
            "passed_commands": 1,
            "failed_commands": 0,
            "results": [
                {
                    "command": "pytest -v",
                    "exit_code": 0,
                    "stdout": "1 passed",
                    "stderr": "",
                    "timed_out": False,
                    "passed": True,
                }
            ],
            "failed_command_names": [],
            "summary": "All generated test commands passed.",
        },
        created_by=AgentRole.TESTER,
    )


def build_state_and_task(
    test_artifact=None,
):
    state = NexusState(
        user_request="Build application"
    )

    code_artifact = create_code_artifact()

    if test_artifact is None:
        test_artifact = create_failed_test_artifact()

    state.add_artifact(
        code_artifact
    )

    state.add_artifact(
        test_artifact
    )

    task = AgentTask(
        title="Debug implementation",
        description="Repair failing generated code.",
        assigned_agent=AgentRole.DEBUGGER,
        input_artifact_ids=[
            code_artifact.id,
            test_artifact.id,
        ],
    )

    return state, task


def test_debugger_returns_debug_report():
    state, task = build_state_and_task()

    agent = DebuggerAgent(
        llm_client=FakeDebuggerLLM()
    )

    artifact = agent.execute(
        task,
        state,
    )

    assert (
        artifact.type
        == ArtifactType.DEBUG_REPORT
    )

    assert (
        artifact.created_by
        == AgentRole.DEBUGGER
    )

    assert len(
        artifact.content["patches"]
    ) == 1

    assert (
        artifact.content["patches"][0]["path"]
        == "app.py"
    )

    assert (
        artifact.metadata["patch_count"]
        == 1
    )


def test_debugger_repairs_invalid_output():
    fake_llm = RepairingDebuggerLLM()

    state, task = build_state_and_task()

    agent = DebuggerAgent(
        llm_client=fake_llm
    )

    artifact = agent.execute(
        task,
        state,
    )

    assert fake_llm.calls == 2

    assert (
        artifact.metadata["validation_attempts"]
        == 2
    )


def test_debugger_rejects_unknown_file():
    state, task = build_state_and_task()

    agent = DebuggerAgent(
        llm_client=UnknownFileDebuggerLLM(),
        max_validation_retries=0,
    )

    with pytest.raises(
        DebugGenerationError,
        match="unknown file",
    ):
        agent.execute(
            task,
            state,
        )


def test_debugger_rejects_directory_traversal():
    state, task = build_state_and_task()

    agent = DebuggerAgent(
        llm_client=TraversalDebuggerLLM(),
        max_validation_retries=0,
    )

    with pytest.raises(
        DebugGenerationError,
        match="traversal",
    ):
        agent.execute(
            task,
            state,
        )


def test_debugger_rejects_absolute_path():
    state, task = build_state_and_task()

    agent = DebuggerAgent(
        llm_client=AbsolutePathDebuggerLLM(),
        max_validation_retries=0,
    )

    with pytest.raises(
        DebugGenerationError,
        match="Absolute patch path",
    ):
        agent.execute(
            task,
            state,
        )


def test_debugger_rejects_duplicate_patch_path():
    state, task = build_state_and_task()

    agent = DebuggerAgent(
        llm_client=DuplicatePatchDebuggerLLM(),
        max_validation_retries=0,
    )

    with pytest.raises(
        DebugGenerationError,
        match="Duplicate patch path",
    ):
        agent.execute(
            task,
            state,
        )


def test_debugger_fails_after_retry_limit():
    state, task = build_state_and_task()

    agent = DebuggerAgent(
        llm_client=AlwaysInvalidDebuggerLLM(),
        max_validation_retries=1,
    )

    with pytest.raises(
        DebugGenerationError
    ):
        agent.execute(
            task,
            state,
        )


def test_debugger_fails_without_code_artifact():
    state = NexusState(
        user_request="Build application"
    )

    test_artifact = create_failed_test_artifact()
    state.add_artifact(test_artifact)

    task = AgentTask(
        title="Debug implementation",
        description="Repair code.",
        assigned_agent=AgentRole.DEBUGGER,
        input_artifact_ids=[
            test_artifact.id
        ],
    )

    agent = DebuggerAgent(
        llm_client=FakeDebuggerLLM()
    )

    with pytest.raises(
        DebugGenerationError,
        match="code artifact not found",
    ):
        agent.execute(
            task,
            state,
        )


def test_debugger_fails_without_test_result():
    state = NexusState(
        user_request="Build application"
    )

    code_artifact = create_code_artifact()
    state.add_artifact(code_artifact)

    task = AgentTask(
        title="Debug implementation",
        description="Repair code.",
        assigned_agent=AgentRole.DEBUGGER,
        input_artifact_ids=[
            code_artifact.id
        ],
    )

    agent = DebuggerAgent(
        llm_client=FakeDebuggerLLM()
    )

    with pytest.raises(
        DebugGenerationError,
        match="test_result artifact not found",
    ):
        agent.execute(
            task,
            state,
        )


def test_debugger_does_not_run_when_tests_pass():
    state, task = build_state_and_task(
        test_artifact=create_passing_test_artifact()
    )

    agent = DebuggerAgent(
        llm_client=FakeDebuggerLLM()
    )

    with pytest.raises(
        DebugGenerationError,
        match="tests already pass",
    ):
        agent.execute(
            task,
            state,
        )
