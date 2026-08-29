import sys
from pathlib import Path

import pytest

from app.tools.executor import (
    CommandExecutor,
    ExecutionError,
)


def test_executor_runs_python_command(
    tmp_path,
):
    executor = CommandExecutor()

    result = executor.execute(
        "python -c \"print('NEXUS_OK')\"",
        tmp_path,
    )

    assert result.exit_code == 0
    assert result.stdout.strip() == "NEXUS_OK"
    assert result.stderr == ""
    assert result.timed_out is False


def test_executor_captures_failure(
    tmp_path,
):
    executor = CommandExecutor()

    result = executor.execute(
        "python -c \"raise RuntimeError('boom')\"",
        tmp_path,
    )

    assert result.exit_code != 0
    assert "RuntimeError" in result.stderr
    assert "boom" in result.stderr
    assert result.timed_out is False


def test_executor_runs_inside_workspace(
    tmp_path,
):
    target = tmp_path / "hello.py"

    target.write_text(
        "print('WORKSPACE_OK')\n",
        encoding="utf-8",
    )

    executor = CommandExecutor()

    result = executor.execute(
        "python hello.py",
        tmp_path,
    )

    assert result.exit_code == 0
    assert result.stdout.strip() == "WORKSPACE_OK"


def test_executor_rejects_unapproved_executable(
    tmp_path,
):
    executor = CommandExecutor()

    with pytest.raises(
        ExecutionError,
        match="Executable is not allowed",
    ):
        executor.execute(
            "rm -rf something",
            tmp_path,
        )


def test_executor_rejects_shell_chaining(
    tmp_path,
):
    executor = CommandExecutor()

    with pytest.raises(
        ExecutionError,
        match="Forbidden shell token",
    ):
        executor.execute(
            "python app.py && rm file.txt",
            tmp_path,
        )


def test_executor_rejects_pipe(
    tmp_path,
):
    executor = CommandExecutor()

    with pytest.raises(
        ExecutionError,
        match="Forbidden shell token",
    ):
        executor.execute(
            "python app.py | cat",
            tmp_path,
        )


def test_executor_rejects_redirection(
    tmp_path,
):
    executor = CommandExecutor()

    with pytest.raises(
        ExecutionError,
        match="Forbidden shell token",
    ):
        executor.execute(
            "python app.py > output.txt",
            tmp_path,
        )


def test_executor_rejects_missing_workspace(
    tmp_path,
):
    executor = CommandExecutor()

    missing = (
        tmp_path
        / "does-not-exist"
    )

    with pytest.raises(
        ExecutionError,
        match="Workspace does not exist",
    ):
        executor.execute(
            "python app.py",
            missing,
        )


def test_executor_rejects_empty_command(
    tmp_path,
):
    executor = CommandExecutor()

    with pytest.raises(
        ExecutionError,
        match="Command cannot be empty",
    ):
        executor.execute(
            "",
            tmp_path,
        )


def test_executor_times_out(
    tmp_path,
):
    executor = CommandExecutor(
        timeout_seconds=1
    )

    result = executor.execute(
        (
            "python -c "
            "\"import time; time.sleep(2)\""
        ),
        tmp_path,
    )

    assert result.exit_code == -1
    assert result.timed_out is True


def test_pytest_isolated_from_parent_project_config(
    tmp_path,
):
    workspace = tmp_path / "generated_project"
    workspace.mkdir()

    (workspace / "app.py").write_text(
        "def run():\n"
        "    return True\n",
        encoding="utf-8",
    )

    tests_dir = workspace / "tests"
    tests_dir.mkdir()

    (
        tests_dir / "test_app.py"
    ).write_text(
        "from app import run\n\n"
        "def test_run():\n"
        "    assert run() is True\n",
        encoding="utf-8",
    )

    executor = CommandExecutor(
        timeout_seconds=10
    )

    result = executor.execute(
        "pytest -q",
        workspace,
    )

    assert result.exit_code == 0
    assert result.timed_out is False
    assert "1 passed" in result.stdout
