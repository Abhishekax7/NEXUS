from pathlib import Path

import pytest

from app.core.models import (
    AgentRole,
    Artifact,
    ArtifactType,
)
from app.core.state import NexusState
from app.tools.workspace import (
    WorkspaceError,
    WorkspaceWriter,
)


def create_code_artifact():
    return Artifact(
        type=ArtifactType.CODE,
        name="generated_code_bundle",
        content={
            "project_name": "demo_project",
            "summary": "Demo project",
            "files": [
                {
                    "path": "app.py",
                    "content": "print('hello')\n",
                    "purpose": "Entry point",
                },
                {
                    "path": "src/service.py",
                    "content": (
                        "def run():\n"
                        "    return True\n"
                    ),
                    "purpose": "Service module",
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
                "Test bundle",
            ],
        },
        created_by=AgentRole.CODER,
    )


def test_workspace_writer_creates_files(
    tmp_path,
):
    state = NexusState(
        user_request="Build application"
    )

    writer = WorkspaceWriter(
        root=str(tmp_path)
    )

    artifact = create_code_artifact()

    written = writer.write_code_artifact(
        artifact,
        state,
    )

    assert len(written) == 2

    run_directory = (
        tmp_path
        / state.run_id
    )

    assert (
        run_directory
        / "app.py"
    ).exists()

    assert (
        run_directory
        / "src"
        / "service.py"
    ).exists()


def test_workspace_writer_preserves_content(
    tmp_path,
):
    state = NexusState(
        user_request="Build application"
    )

    writer = WorkspaceWriter(
        root=str(tmp_path)
    )

    artifact = create_code_artifact()

    writer.write_code_artifact(
        artifact,
        state,
    )

    target = (
        tmp_path
        / state.run_id
        / "app.py"
    )

    assert target.read_text(
        encoding="utf-8"
    ) == "print('hello')\n"


def test_workspace_rejects_non_code_artifact(
    tmp_path,
):
    state = NexusState(
        user_request="Build application"
    )

    writer = WorkspaceWriter(
        root=str(tmp_path)
    )

    artifact = Artifact(
        type=ArtifactType.RESEARCH,
        name="research",
        content={},
        created_by=AgentRole.RESEARCH,
    )

    with pytest.raises(
        WorkspaceError,
        match="only accepts CODE artifacts",
    ):
        writer.write_code_artifact(
            artifact,
            state,
        )


def test_workspace_rejects_directory_traversal(
    tmp_path,
):
    state = NexusState(
        user_request="Build application"
    )

    writer = WorkspaceWriter(
        root=str(tmp_path)
    )

    artifact = create_code_artifact()

    artifact.content["files"][0][
        "path"
    ] = "../secret.txt"

    with pytest.raises(
        WorkspaceError,
        match="Directory traversal",
    ):
        writer.write_code_artifact(
            artifact,
            state,
        )


def test_workspace_rejects_absolute_path(
    tmp_path,
):
    state = NexusState(
        user_request="Build application"
    )

    writer = WorkspaceWriter(
        root=str(tmp_path)
    )

    artifact = create_code_artifact()

    artifact.content["files"][0][
        "path"
    ] = "/tmp/secret.txt"

    with pytest.raises(
        WorkspaceError,
        match="Absolute path",
    ):
        writer.write_code_artifact(
            artifact,
            state,
        )


def test_workspace_files_stay_inside_run_directory(
    tmp_path,
):
    state = NexusState(
        user_request="Build application"
    )

    writer = WorkspaceWriter(
        root=str(tmp_path)
    )

    artifact = create_code_artifact()

    written_files = writer.write_code_artifact(
        artifact,
        state,
    )

    run_directory = (
        tmp_path
        / state.run_id
    ).resolve()

    for written_file in written_files:
        written_file.resolve().relative_to(
            run_directory
        )
