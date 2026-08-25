import pytest

from app.core.models import (
    AgentRole,
    Artifact,
    ArtifactType,
)
from app.core.state import NexusState
from app.tools.patcher import (
    PatchApplicator,
    PatchError,
)


def create_debug_artifact():
    return Artifact(
        type=ArtifactType.DEBUG_REPORT,
        name="debug_repair_report",
        content={
            "root_cause": (
                "Incorrect return value."
            ),
            "failure_summary": (
                "Test expected True."
            ),
            "patches": [
                {
                    "path": "app.py",
                    "new_content": (
                        "def run():\n"
                        "    return True\n"
                    ),
                    "reason": (
                        "Fix incorrect return value."
                    ),
                }
            ],
            "retry_test_commands": [
                "pytest -v",
            ],
            "confidence": 0.95,
            "notes": [
                "Minimal repair",
            ],
        },
        created_by=AgentRole.DEBUGGER,
    )


def create_workspace(
    tmp_path,
    state,
):
    run_directory = (
        tmp_path
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
        (
            "def run():\n"
            "    return False\n"
        ),
        encoding="utf-8",
    )

    return run_directory


def test_patcher_applies_debug_patch(
    tmp_path,
):
    state = NexusState(
        user_request="Build app"
    )

    create_workspace(
        tmp_path,
        state,
    )

    patcher = PatchApplicator(
        root=str(tmp_path)
    )

    artifact = create_debug_artifact()

    applied = (
        patcher.apply_debug_artifact(
            artifact,
            state,
        )
    )

    assert len(applied) == 1

    target = (
        tmp_path
        / state.run_id
        / "app.py"
    )

    assert target.read_text(
        encoding="utf-8"
    ) == (
        "def run():\n"
        "    return True\n"
    )


def test_patcher_rejects_non_debug_artifact(
    tmp_path,
):
    state = NexusState(
        user_request="Build app"
    )

    create_workspace(
        tmp_path,
        state,
    )

    patcher = PatchApplicator(
        root=str(tmp_path)
    )

    artifact = Artifact(
        type=ArtifactType.CODE,
        name="code",
        content={},
        created_by=AgentRole.CODER,
    )

    with pytest.raises(
        PatchError,
        match="only accepts",
    ):
        patcher.apply_debug_artifact(
            artifact,
            state,
        )


def test_patcher_rejects_missing_workspace(
    tmp_path,
):
    state = NexusState(
        user_request="Build app"
    )

    patcher = PatchApplicator(
        root=str(tmp_path)
    )

    artifact = create_debug_artifact()

    with pytest.raises(
        PatchError,
        match="does not exist",
    ):
        patcher.apply_debug_artifact(
            artifact,
            state,
        )


def test_patcher_rejects_unknown_file(
    tmp_path,
):
    state = NexusState(
        user_request="Build app"
    )

    create_workspace(
        tmp_path,
        state,
    )

    artifact = create_debug_artifact()

    artifact.content[
        "patches"
    ][0]["path"] = (
        "unknown.py"
    )

    patcher = PatchApplicator(
        root=str(tmp_path)
    )

    with pytest.raises(
        PatchError,
        match="target does not exist",
    ):
        patcher.apply_debug_artifact(
            artifact,
            state,
        )


def test_patcher_rejects_directory_traversal(
    tmp_path,
):
    state = NexusState(
        user_request="Build app"
    )

    create_workspace(
        tmp_path,
        state,
    )

    artifact = create_debug_artifact()

    artifact.content[
        "patches"
    ][0]["path"] = (
        "../secret.txt"
    )

    patcher = PatchApplicator(
        root=str(tmp_path)
    )

    with pytest.raises(
        PatchError,
        match="traversal",
    ):
        patcher.apply_debug_artifact(
            artifact,
            state,
        )


def test_patcher_rejects_absolute_path(
    tmp_path,
):
    state = NexusState(
        user_request="Build app"
    )

    create_workspace(
        tmp_path,
        state,
    )

    artifact = create_debug_artifact()

    artifact.content[
        "patches"
    ][0]["path"] = (
        "/tmp/app.py"
    )

    patcher = PatchApplicator(
        root=str(tmp_path)
    )

    with pytest.raises(
        PatchError,
        match="Absolute patch path",
    ):
        patcher.apply_debug_artifact(
            artifact,
            state,
        )


def test_patcher_rejects_duplicate_paths(
    tmp_path,
):
    state = NexusState(
        user_request="Build app"
    )

    create_workspace(
        tmp_path,
        state,
    )

    artifact = create_debug_artifact()

    artifact.content[
        "patches"
    ].append(
        {
            "path": "app.py",
            "new_content": (
                "print('duplicate')\n"
            ),
            "reason": "Duplicate",
        }
    )

    patcher = PatchApplicator(
        root=str(tmp_path)
    )

    with pytest.raises(
        PatchError,
        match="Duplicate patch path",
    ):
        patcher.apply_debug_artifact(
            artifact,
            state,
        )


def test_patcher_rejects_empty_content(
    tmp_path,
):
    state = NexusState(
        user_request="Build app"
    )

    create_workspace(
        tmp_path,
        state,
    )

    artifact = create_debug_artifact()

    artifact.content[
        "patches"
    ][0]["new_content"] = ""

    patcher = PatchApplicator(
        root=str(tmp_path)
    )

    with pytest.raises(
        PatchError,
        match="cannot be empty",
    ):
        patcher.apply_debug_artifact(
            artifact,
            state,
        )
