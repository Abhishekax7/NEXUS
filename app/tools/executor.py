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

    def _parse_command(
        self,
        command: str,
    ) -> list[str]:
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

        if executable not in self.ALLOWED_EXECUTABLES:
            raise ExecutionError(
                f"Executable is not allowed: {executable}"
            )

        return parts

    def execute(
        self,
        command: str,
        workspace: Path,
    ) -> ExecutionResult:
        workspace = self._validate_workspace(
            workspace
        )

        args = self._parse_command(
            command
        )

        try:
            completed = subprocess.run(
                args,
                cwd=workspace,
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
