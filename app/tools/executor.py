import os
import shlex
import subprocess

from pathlib import Path

from pydantic import BaseModel


class ExecutionError(Exception):
    """Raised when command execution is unsafe or invalid."""


class ExecutionResult(BaseModel):
    command: str
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False


class CommandExecutor:
    """
    Executes a restricted set of development commands
    inside a specific NEXUS run workspace.

    CommandExecutor is the authoritative runtime-policy
    boundary for executable command validation.
    """

    ALLOWED_EXECUTABLES = {
        "python",
        "python3",
        "pytest",
    }

    FORBIDDEN_TOKENS = {
        "&&",
        "||",
        ";",
        "|",
        ">",
        ">>",
        "<",
        "<<",
        "`",
    }

    def __init__(
        self,
        timeout_seconds: int = 20,
    ):
        if timeout_seconds <= 0:
            raise ValueError(
                "timeout_seconds must be greater than zero."
            )

        self.timeout_seconds = timeout_seconds

    @property
    def allowed_executables(
        self,
    ) -> frozenset[str]:
        """
        Return the executables supported by the
        current NEXUS runtime.

        A frozen set prevents callers from mutating
        the executor policy accidentally.
        """

        return frozenset(
            self.ALLOWED_EXECUTABLES
        )

    def validate_command(
        self,
        command: str,
    ) -> list[str]:
        """
        Validate a command against the NEXUS execution
        policy and return its parsed argument list.

        This method performs validation only.
        It never executes the command.
        """

        if not isinstance(command, str):
            raise ExecutionError(
                "Command must be a string."
            )

        command = command.strip()

        if not command:
            raise ExecutionError(
                "Command cannot be empty."
            )

        try:
            parts = shlex.split(command)

        except ValueError as exc:
            raise ExecutionError(
                f"Invalid command syntax: {exc}"
            ) from exc

        if not parts:
            raise ExecutionError(
                "Command cannot be empty."
            )

        for token in parts:
            if token in self.FORBIDDEN_TOKENS:
                raise ExecutionError(
                    f"Forbidden shell token: {token}"
                )

        executable = parts[0]

        if executable not in self.allowed_executables:
            raise ExecutionError(
                f"Executable is not allowed: {executable}"
            )

        return parts

    def _parse_command(
        self,
        command: str,
    ) -> list[str]:
        """
        Backward-compatible private wrapper.

        New NEXUS components should use
        validate_command() directly.
        """

        return self.validate_command(
            command
        )

    def _validate_workspace(
        self,
        workspace: Path,
    ) -> Path:
        workspace = workspace.resolve()

        if not workspace.exists():
            raise ExecutionError(
                f"Workspace does not exist: {workspace}"
            )

        if not workspace.is_dir():
            raise ExecutionError(
                f"Workspace is not a directory: {workspace}"
            )

        return workspace

    def execute(
        self,
        command: str,
        workspace: Path,
    ) -> ExecutionResult:
        workspace = self._validate_workspace(
            workspace
        )

        args = self.validate_command(
            command
        )

        env = os.environ.copy()

        existing_pythonpath = env.get(
            "PYTHONPATH"
        )

        workspace_pythonpath = str(
            workspace
        )

        if existing_pythonpath:
            env["PYTHONPATH"] = (
                workspace_pythonpath
                + os.pathsep
                + existing_pythonpath
            )
        else:
            env["PYTHONPATH"] = (
                workspace_pythonpath
            )

        is_direct_pytest = (
            args
            and args[0] == "pytest"
        )

        is_python_pytest = (
            len(args) >= 3
            and args[0]
            in {
                "python",
                "python3",
            }
            and args[1:3]
            == [
                "-m",
                "pytest",
            ]
        )

        if (
            is_direct_pytest
            or is_python_pytest
        ):
            local_pytest_configs = (
                "pytest.ini",
                "pyproject.toml",
                "setup.cfg",
                "tox.ini",
            )

            has_local_config = any(
                (
                    workspace
                    / config_name
                ).exists()
                for config_name
                in local_pytest_configs
            )

            if not has_local_config:
                args = [
                    *args,
                    "-c",
                    "/dev/null",
                    "--rootdir",
                    str(workspace),
                ]

        try:
            completed = subprocess.run(
                args,
                cwd=workspace,
                env=env,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                shell=False,
            )

            return ExecutionResult(
                command=command,
                exit_code=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
                timed_out=False,
            )

        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""

            if isinstance(stdout, bytes):
                stdout = stdout.decode(
                    "utf-8",
                    errors="replace",
                )

            if isinstance(stderr, bytes):
                stderr = stderr.decode(
                    "utf-8",
                    errors="replace",
                )

            return ExecutionResult(
                command=command,
                exit_code=-1,
                stdout=stdout,
                stderr=stderr,
                timed_out=True,
            )
