from pathlib import Path
from typing import Iterable

from app.core.models import Artifact, ArtifactType
from app.core.state import NexusState


class WorkspaceError(Exception):
    """Raised when workspace operations are unsafe or invalid."""


class WorkspaceWriter:
    def __init__(
        self,
        root: str = "workspace",
    ):
        self.root = Path(root).resolve()

    def _validate_relative_path(
        self,
        relative_path: str,
    ) -> None:
        path = Path(relative_path)

        if path.is_absolute():
            raise WorkspaceError(
                f"Absolute path is forbidden: {relative_path}"
            )

        if ".." in path.parts:
            raise WorkspaceError(
                f"Directory traversal is forbidden: {relative_path}"
            )

        if relative_path.strip() == "":
            raise WorkspaceError(
                "Empty file path is not allowed."
            )

    def _safe_target(
        self,
        run_directory: Path,
        relative_path: str,
    ) -> Path:
        self._validate_relative_path(
            relative_path
        )

        target = (
            run_directory
            / relative_path
        ).resolve()

        try:
            target.relative_to(
                run_directory.resolve()
            )
        except ValueError as exc:
            raise WorkspaceError(
                f"Path escapes workspace: {relative_path}"
            ) from exc

        return target

    def create_run_directory(
        self,
        state: NexusState,
    ) -> Path:
        run_directory = (
            self.root
            / state.run_id
        ).resolve()

        run_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        return run_directory

    def write_code_artifact(
        self,
        artifact: Artifact,
        state: NexusState,
    ) -> list[Path]:
        if artifact.type != ArtifactType.CODE:
            raise WorkspaceError(
                "WorkspaceWriter only accepts CODE artifacts."
            )

        files = artifact.content.get(
            "files"
        )

        if not isinstance(files, list):
            raise WorkspaceError(
                "CODE artifact does not contain a valid files list."
            )

        run_directory = self.create_run_directory(
            state
        )

        written_files = []

        for file_data in files:
            relative_path = file_data.get(
                "path"
            )

            content = file_data.get(
                "content"
            )

            if not isinstance(
                relative_path,
                str,
            ):
                raise WorkspaceError(
                    "Generated file path must be a string."
                )

            if not isinstance(
                content,
                str,
            ):
                raise WorkspaceError(
                    f"Generated file content must be a string: "
                    f"{relative_path}"
                )

            target = self._safe_target(
                run_directory,
                relative_path,
            )

            target.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            target.write_text(
                content,
                encoding="utf-8",
            )

            written_files.append(
                target
            )

        return written_files
