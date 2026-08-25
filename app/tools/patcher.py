from pathlib import Path

from app.core.models import Artifact, ArtifactType
from app.core.state import NexusState


class PatchError(Exception):
    """Raised when a debug patch is unsafe or invalid."""


class PatchApplicator:
    def __init__(
        self,
        root: str = "workspace",
    ):
        self.root = Path(root).resolve()

    def _get_run_directory(
        self,
        state: NexusState,
    ) -> Path:
        run_directory = (
            self.root
            / state.run_id
        ).resolve()

        if not run_directory.exists():
            raise PatchError(
                f"Run workspace does not exist: {run_directory}"
            )

        if not run_directory.is_dir():
            raise PatchError(
                f"Run workspace is not a directory: {run_directory}"
            )

        return run_directory

    def _safe_target(
        self,
        run_directory: Path,
        relative_path: str,
    ) -> Path:
        if not isinstance(
            relative_path,
            str,
        ):
            raise PatchError(
                "Patch path must be a string."
            )

        relative_path = relative_path.strip()

        if not relative_path:
            raise PatchError(
                "Patch path cannot be empty."
            )

        path = Path(
            relative_path
        )

        if path.is_absolute():
            raise PatchError(
                f"Absolute patch path is forbidden: "
                f"{relative_path}"
            )

        if ".." in path.parts:
            raise PatchError(
                f"Patch path traversal is forbidden: "
                f"{relative_path}"
            )

        target = (
            run_directory
            / path
        ).resolve()

        try:
            target.relative_to(
                run_directory
            )
        except ValueError as exc:
            raise PatchError(
                f"Patch escapes workspace: "
                f"{relative_path}"
            ) from exc

        return target

    def apply_debug_artifact(
        self,
        artifact: Artifact,
        state: NexusState,
    ) -> list[Path]:
        if (
            artifact.type
            != ArtifactType.DEBUG_REPORT
        ):
            raise PatchError(
                "PatchApplicator only accepts "
                "DEBUG_REPORT artifacts."
            )

        patches = artifact.content.get(
            "patches"
        )

        if not isinstance(
            patches,
            list,
        ):
            raise PatchError(
                "DEBUG_REPORT does not contain "
                "a valid patches list."
            )

        if not patches:
            raise PatchError(
                "DEBUG_REPORT contains no patches."
            )

        run_directory = (
            self._get_run_directory(
                state
            )
        )

        applied_paths = []
        seen_paths = set()

        for patch in patches:
            if not isinstance(
                patch,
                dict,
            ):
                raise PatchError(
                    "Each patch must be an object."
                )

            relative_path = patch.get(
                "path"
            )

            new_content = patch.get(
                "new_content"
            )

            target = self._safe_target(
                run_directory,
                relative_path,
            )

            if relative_path in seen_paths:
                raise PatchError(
                    f"Duplicate patch path: "
                    f"{relative_path}"
                )

            seen_paths.add(
                relative_path
            )

            if not target.exists():
                raise PatchError(
                    f"Patch target does not exist: "
                    f"{relative_path}"
                )

            if not target.is_file():
                raise PatchError(
                    f"Patch target is not a file: "
                    f"{relative_path}"
                )

            if not isinstance(
                new_content,
                str,
            ):
                raise PatchError(
                    f"Patch content must be a string: "
                    f"{relative_path}"
                )

            if not new_content:
                raise PatchError(
                    f"Patch content cannot be empty: "
                    f"{relative_path}"
                )

            target.write_text(
                new_content,
                encoding="utf-8",
            )

            applied_paths.append(
                target
            )

        return applied_paths
